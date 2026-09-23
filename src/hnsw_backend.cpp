#include "feedsearch/backend.hpp"
#include "hnswlib.h"
#include <cstring>

namespace feedanns::adapters {
class Hnsw final : public Backend {
    std::unique_ptr<hnswlib::SpaceInterface<float>> space;
    using Index = hnswlib::HierarchicalNSW<float>;
    std::unique_ptr<Index> index;
    struct CounterContext {
        hnswlib::DISTFUNC<float> fn;
        const void *param;
    } counter;
    static float counted(const void *a, const void *b, const void *p) {
        ++distance_calls;
        const auto &c = *static_cast<const CounterContext *>(p);
        return c.fn(a, b, c.param);
    }

  public:
    Hnsw(const std::string &path, Metric m, std::size_t dim) : Backend(m) {
        d = dim;
        if (m == Metric::l2)
            space = std::make_unique<hnswlib::L2Space>(d);
        else
            space = std::make_unique<hnswlib::InnerProductSpace>(d);
        index = std::make_unique<Index>(space.get(), path, false);
        n = index->cur_element_count;
        if (index->num_deleted_)
            throw std::runtime_error("Deleted HNSW nodes are not supported by this adapter");
        counter = {index->fstdistfunc_, index->dist_func_param_};
#ifdef FEED_COUNT_DISTANCES
        index->fstdistfunc_ = counted;
        index->dist_func_param_ = &counter;
#endif
        for (const auto &x : index->label_lookup_)
            if (x.first >= n)
                throw std::runtime_error("Expected dense external base IDs");
    }
    std::unique_ptr<Workspace> workspace() const override {
        return std::make_unique<Workspace>();
    }
    void set_width(std::size_t width) override {
        index->setEf(width);
    }
    std::string description() const override {
        return "hnswlib searchKnn; M=" + std::to_string(index->M_) +
               " efConstruction=" + std::to_string(index->ef_construction_);
    }
    void vector_at(std::uint32_t id, float *out) const override {
        std::memcpy(out, index->getDataByInternalId(index->label_lookup_.at(id)),
                    d * sizeof(float));
    }
    Output search(const float *query, std::size_t k, double lambda, bool feed,
                  Workspace &ws) override {
        const float *q = prepare(query, ws);
        Output out;
        out.hits.reserve(k);
        if (!feed) {
            auto heap = index->searchKnn(q, k);
            while (!heap.empty()) {
                auto x = heap.top();
                heap.pop();
                out.hits.push_back({static_cast<std::uint32_t>(x.second), metric == Metric::mips
                                                                              ? 1 - double(x.first)
                                                                              : double(x.first)});
            }
            std::reverse(out.hits.begin(), out.hits.end());
            return out;
        }
        QueryMetric conversion(metric, q, d, max_base_norm());
        auto cur = index->enterpoint_node_;
        float best =
            index->fstdistfunc_(q, index->getDataByInternalId(cur), index->dist_func_param_);
        for (int level = index->maxlevel_; level > 0; --level) {
            bool changed = true;
            while (changed) {
                changed = false;
                auto *links = index->get_linklist(cur, level);
                const auto count = index->getListCount(links);
                ++index->metric_hops;
                index->metric_distance_computations += count;
                auto *ids = reinterpret_cast<const hnswlib::tableint *>(links + 1);
                for (unsigned j = 0; j < count; ++j) {
                    const auto id = ids[j];
                    if (id > index->max_elements_)
                        throw std::runtime_error("Invalid HNSW edge");
                    const float dist = index->fstdistfunc_(q, index->getDataByInternalId(id),
                                                           index->dist_func_param_);
                    if (dist < best) {
                        best = dist;
                        cur = id;
                        changed = true;
                    }
                }
            }
        }
        feedanns::FeedbackController controller(lambda);
        auto result = feedanns::FeedSearch(
            std::vector<std::uint32_t>{cur}, k, n, ws.feed,
            [&](std::uint32_t id, auto visit) {
                auto *links = index->get_linklist0(id);
                const auto count = index->getListCount(links);
                auto *ids = reinterpret_cast<const hnswlib::tableint *>(links + 1);
                for (unsigned j = 0; j < count; ++j)
                    if (!visit(ids[j]))
                        break;
            },
            [&](std::uint32_t id) {
                return conversion.positive(
                    index->fstdistfunc_(q, index->getDataByInternalId(id), index->dist_func_param_),
                    1);
            },
            controller);
        for (const auto &x : result.neighbors)
            out.hits.push_back({static_cast<std::uint32_t>(index->getExternalLabel(x.id)),
                                conversion.public_score(x.distance)});
        out.layer_distances = result.distance_computations;
        out.expansions = result.expansions;
        out.stop = result.stop_reason;
        return out;
    }
};
std::unique_ptr<Backend> load_hnsw(const std::string &p, Metric m, std::size_t d) {
    return std::make_unique<Hnsw>(p, m, d);
}
} // namespace feedanns::adapters
