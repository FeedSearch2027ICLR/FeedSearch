#include "feedsearch/backend.hpp"
#include <faiss/IndexNSG.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/index_io.h>

namespace feedanns::adapters {
struct CountingComputer : faiss::DistanceComputer {
    std::unique_ptr<faiss::DistanceComputer> inner;
    explicit CountingComputer(faiss::DistanceComputer *p) : inner(p) {}
    void set_query(const float *q) override { inner->set_query(q); }
    float operator()(faiss::idx_t id) override {
        ++distance_calls;
        return (*inner)(id);
    }
    float symmetric_dis(faiss::idx_t a, faiss::idx_t b) override {
        ++distance_calls;
        return inner->symmetric_dis(a, b);
    }
};
class Nsg final : public Backend {
    std::unique_ptr<faiss::Index> owned;
    faiss::IndexNSG *index;
    struct State : Workspace {
        faiss::VisitedTable visited;
        std::unique_ptr<faiss::DistanceComputer> distance;
        std::vector<faiss::idx_t> ids;
        std::vector<float> values;
        State(std::size_t n, faiss::DistanceComputer *dc) : visited(n), distance(dc) {}
    };

  public:
    Nsg(const std::string &path, Metric m, std::size_t dim) : Backend(m) {
        owned.reset(faiss::read_index(path.c_str()));
        index = dynamic_cast<faiss::IndexNSG *>(owned.get());
        if (!index || !index->nsg.final_graph)
            throw std::runtime_error("Expected built Faiss NSG");
        n = index->ntotal;
        d = index->d;
        if (d != dim || index->metric_type !=
                            (m == Metric::l2 ? faiss::METRIC_L2 : faiss::METRIC_INNER_PRODUCT))
            throw std::runtime_error("NSG metric/dimension mismatch");
    }
    std::unique_ptr<Workspace> workspace() const override {
        auto *dc = faiss::nsg::storage_distance_computer(index->storage);
#ifdef FEED_COUNT_DISTANCES
        dc = new CountingComputer(dc);
#endif
        return std::make_unique<State>(n, dc);
    }
    void set_width(std::size_t w) override {
        index->nsg.search_L = w;
    }
    std::string description() const override {
        return "Faiss NSG::search (native batch worker); R=" + std::to_string(index->nsg.R);
    }
    void vector_at(std::uint32_t id, float *out) const override {
        index->reconstruct(id, out);
    }
    Output search(const float *query, std::size_t k, double lambda, bool feed,
                  Workspace &ws) override {
        auto &state = static_cast<State &>(ws);
        const float *q = prepare(query, ws);
        auto &dc = *state.distance;
        dc.set_query(q);
        Output out;
        out.hits.reserve(k);
        if (!feed) {
            state.ids.resize(k);
            state.values.resize(k);
            index->nsg.search(dc, k, state.ids.data(), state.values.data(), state.visited);
            state.visited.advance();
            for (std::size_t j = 0; j < k; ++j)
                out.hits.push_back({static_cast<std::uint32_t>(state.ids[j]),
                                    metric == Metric::mips     ? -double(state.values[j])
                                    : metric == Metric::cosine ? 1 + double(state.values[j])
                                                               : double(state.values[j])});
            return out;
        }
        QueryMetric conversion(metric, q, d, max_base_norm());
        feedanns::FeedbackController controller(lambda);
        const auto &graph = *index->nsg.final_graph;
        auto result = feedanns::FeedSearch(
            std::vector<std::uint32_t>{static_cast<std::uint32_t>(index->nsg.enterpoint)}, k, n,
            ws.feed,
            [&](std::uint32_t id, auto visit) {
                for (int j = 0; j < graph.K; ++j) {
                    auto nb = graph.at(id, j);
                    if (nb < 0)
                        break;
                    if (!visit(static_cast<std::uint32_t>(nb)))
                        break;
                }
            },
            [&](std::uint32_t id) { return conversion.positive(dc(id), 0); }, controller);
        for (const auto &x : result.neighbors)
            out.hits.push_back({x.id, conversion.public_score(x.distance)});
        out.layer_distances = result.distance_computations;
        out.expansions = result.expansions;
        out.stop = result.stop_reason;
        return out;
    }
};
std::unique_ptr<Backend> load_nsg(const std::string &p, Metric m, std::size_t d) {
    return std::make_unique<Nsg>(p, m, d);
}
} // namespace feedanns::adapters
