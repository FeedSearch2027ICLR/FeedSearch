#pragma once
#include "core.hpp"
#include "metrics.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace feedanns::adapters {
inline thread_local std::uint64_t distance_calls = 0;
struct Hit {
    std::uint32_t id;
    double value;
}; // MIPS: similarity; L2/cosine: distance.
struct Output {
    std::vector<Hit> hits;
    std::uint64_t layer_distances = 0, expansions = 0;
    feedanns::StopReason stop = feedanns::StopReason::frontier_exhausted;
};
struct Workspace {
    feedanns::FeedWorkspace<std::uint32_t> feed;
    std::vector<float> normalized;
    virtual ~Workspace() = default;
};
class Backend {
    double max_base_norm_ = std::numeric_limits<double>::quiet_NaN();

  public:
    Metric metric;
    std::size_t n = 0, d = 0;
    explicit Backend(Metric m) : metric(m) {}
    virtual ~Backend() = default;
    virtual std::unique_ptr<Workspace> workspace() const = 0;
    virtual void set_width(std::size_t width) = 0; // outside parallel searches
    virtual Output search(const float *query, std::size_t k, double lambda, bool feed,
                          Workspace &ws) = 0;
    virtual void vector_at(std::uint32_t external_id, float *out) const = 0;
    virtual std::string description() const = 0;
    void set_max_base_norm(double value) {
        if (!std::isfinite(value) || value < 0)
            throw std::invalid_argument("MIPS base norm bound must be finite and nonnegative");
        max_base_norm_ = value;
    }
    double max_base_norm() const noexcept { return max_base_norm_; }
    Output feed_search(const float *query, std::size_t k, double lambda, Workspace &ws) {
        return search(query, k, lambda, true, ws);
    }
    Output native_search(const float *query, std::size_t k, Workspace &ws) {
        return search(query, k, 0, false, ws);
    }
    const float *prepare(const float *q, Workspace &ws) const {
        if (metric != Metric::cosine)
            return q;
        ws.normalized.resize(d);
        normalize_query(q, d, ws.normalized.data());
        return ws.normalized.data();
    }
};
enum class Graph { hnsw, nsg, vamana };
struct IndexOptions {
    Graph graph = Graph::hnsw;
    Metric metric = Metric::l2;
    std::size_t dimension = 0;
    std::size_t threads = 1;
    std::size_t maximum_native_width = 2048;
    double max_base_norm = std::numeric_limits<double>::quiet_NaN();
};
std::unique_ptr<Backend> load_index(const std::string &path, const IndexOptions &options);
std::unique_ptr<Backend> load_hnsw(const std::string &, Metric, std::size_t);
std::unique_ptr<Backend> load_nsg(const std::string &, Metric, std::size_t);
std::unique_ptr<Backend> load_vamana(const std::string &, Metric, std::size_t, std::size_t,
                                     std::size_t);
} // namespace feedanns::adapters
