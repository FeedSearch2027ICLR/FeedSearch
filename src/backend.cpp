#include "feedsearch/backend.hpp"

namespace feedanns::adapters {
std::unique_ptr<Backend> load_index(const std::string &path, const IndexOptions &options) {
    if (options.dimension == 0 || options.threads == 0 || options.maximum_native_width == 0)
        throw std::invalid_argument(
            "Index dimension, thread count and native width must be positive");
    std::unique_ptr<Backend> result;
    switch (options.graph) {
    case Graph::hnsw:
        result = load_hnsw(path, options.metric, options.dimension);
        break;
    case Graph::nsg:
        result = load_nsg(path, options.metric, options.dimension);
        break;
    case Graph::vamana:
        result = load_vamana(path, options.metric, options.dimension, options.threads,
                             options.maximum_native_width);
        break;
    default:
        throw std::invalid_argument("Unknown graph type");
    }
    if (std::isfinite(options.max_base_norm))
        result->set_max_base_norm(options.max_base_norm);
    return result;
}
} // namespace feedanns::adapters
