#include "feedsearch/backend.hpp"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include "in_mem_data_store.h"
#include "in_mem_graph_store.h"
#include "index.h"
#include "index_config.h"
#include "utils.h"
#include <cstring>
#include <fstream>

namespace feedanns::adapters {
class CountingDiskDistance : public diskann::Distance<float> {
    std::unique_ptr<diskann::Distance<float>> inner;

  public:
    explicit CountingDiskDistance(std::unique_ptr<diskann::Distance<float>> p)
        : Distance(p->get_metric()), inner(std::move(p)) {}
    float compare(const float *a, const float *b, std::uint32_t d) const override {
        ++distance_calls;
        return inner->compare(a, b, d);
    }
    float compare(const float *a, const float *b, float an, float bn,
                  std::uint32_t d) const override {
        ++distance_calls;
        return inner->compare(a, b, an, bn, d);
    }
    bool preprocessing_required() const override { return inner->preprocessing_required(); }
    std::uint32_t post_normalization_dimension(std::uint32_t d) const override {
        return inner->post_normalization_dimension(d);
    }
    void preprocess_base_points(float *a, std::size_t d, std::size_t n) override {
        inner->preprocess_base_points(a, d, n);
    }
    void preprocess_query(const float *a, std::size_t d, float *b) override {
        inner->preprocess_query(a, d, b);
    }
    std::size_t get_required_alignment() const override { return inner->get_required_alignment(); }
};
class Vamana final : public Backend {
    std::shared_ptr<diskann::InMemDataStore<float>> data;
    diskann::InMemGraphStore *graph = nullptr;
    std::unique_ptr<diskann::Index<float>> index;
    std::uint32_t entry = 0, width = 10;
    struct State : Workspace {
        float *aligned = nullptr;
        std::vector<std::uint32_t> ids;
        std::vector<float> values;
        explicit State(std::size_t d) {
            diskann::alloc_aligned(reinterpret_cast<void **>(&aligned), d * sizeof(float), 32);
            std::fill(aligned, aligned + d, 0);
        }
        ~State() { diskann::aligned_free(aligned); }
    };

  public:
    Vamana(const std::string &path, Metric m, std::size_t dim, std::size_t threads,
           std::size_t max_l)
        : Backend(m) {
        d = dim;
        std::uint32_t rows, cols, degree;
        std::uint64_t bytes, frozen;
        std::ifstream df(path + ".data", std::ios::binary);
        df.read(reinterpret_cast<char *>(&rows), 4);
        df.read(reinterpret_cast<char *>(&cols), 4);
        std::ifstream gf(path, std::ios::binary);
        gf.read(reinterpret_cast<char *>(&bytes), 8);
        gf.read(reinterpret_cast<char *>(&degree), 4);
        gf.read(reinterpret_cast<char *>(&entry), 4);
        gf.read(reinterpret_cast<char *>(&frozen), 8);
        if (!df || !gf || cols != d || frozen)
            throw std::runtime_error("Unsupported Vamana graph/data header or frozen points");
        n = rows;
        const auto dm = m == Metric::l2     ? diskann::L2
                        : m == Metric::mips ? diskann::INNER_PRODUCT
                                            : diskann::COSINE;
        std::unique_ptr<diskann::Distance<float>> dist;
        if (m == Metric::cosine)
            dist = std::make_unique<diskann::AVXNormalizedCosineDistanceFloat>();
        else
            dist.reset(diskann::get_distance_function<float>(dm));
#ifdef FEED_COUNT_DISTANCES
        dist = std::make_unique<CountingDiskDistance>(std::move(dist));
#endif
        data = std::make_shared<diskann::InMemDataStore<float>>(n, d, std::move(dist));
        auto gs = std::make_unique<diskann::InMemGraphStore>(n, degree);
        graph = gs.get();
        auto cfg = diskann::IndexConfigBuilder()
                       .with_metric(dm)
                       .with_dimension(d)
                       .with_max_points(n)
                       .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                       .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                       .with_data_type("float")
                       .with_num_frozen_pts(0)
                       .build();
        // Match IndexFactory: without PQ, both distance-store slots refer to
        // the same exact-vector store (the optional ctor argument is not inferred).
        index = std::make_unique<diskann::Index<float>>(cfg, data, std::move(gs), data);
        index->load(path.c_str(), threads, max_l);
        if (index->get_num_points() != n)
            throw std::runtime_error("Vamana size mismatch");
    }
    std::unique_ptr<Workspace> workspace() const override {
        return std::make_unique<State>(data->get_aligned_dim());
    }
    void set_width(std::size_t l) override {
        width = l;
    }
    std::string description() const override {
        return "DiskANN Index<float>::search; native graph/data stores and native metric";
    }
    void vector_at(std::uint32_t id, float *out) const override {
        data->get_vector(id, out);
    }
    Output search(const float *query, std::size_t k, double lambda, bool feed,
                  Workspace &ws) override {
        auto &state = static_cast<State &>(ws);
        const float *q = prepare(query, ws);
        Output out;
        out.hits.reserve(k);
        if (!feed) {
            state.ids.assign(k, std::numeric_limits<std::uint32_t>::max());
            state.values.resize(k);
            index->search(q, k, width, state.ids.data(), state.values.data());
            for (std::size_t j = 0; j < k; ++j)
                out.hits.push_back({state.ids[j], metric == Metric::cosine
                                                      ? 1 + double(state.values[j])
                                                      : double(state.values[j])});
            return out;
        }
        std::memcpy(state.aligned, q, d * sizeof(float));
        QueryMetric conversion(metric, q, d, max_base_norm());
        feedanns::FeedbackController controller(lambda);
        auto result = feedanns::FeedSearch(
            std::vector<std::uint32_t>{entry}, k, n, ws.feed,
            [&](std::uint32_t id, auto visit) {
                for (const auto nb : graph->get_neighbours(id))
                    if (!visit(nb))
                        break;
            },
            [&](std::uint32_t id) {
                return conversion.positive(data->get_distance(state.aligned, id), 0);
            },
            controller);
        for (const auto &x : result.neighbors)
            out.hits.push_back({x.id, conversion.public_score(x.distance)});
        out.layer_distances = result.distance_computations;
        out.expansions = result.expansions;
        out.stop = result.stop_reason;
        return out;
    }
};
std::unique_ptr<Backend> load_vamana(const std::string &p, Metric m, std::size_t d, std::size_t t,
                                     std::size_t l) {
    return std::make_unique<Vamana>(p, m, d, t, l);
}
} // namespace feedanns::adapters
