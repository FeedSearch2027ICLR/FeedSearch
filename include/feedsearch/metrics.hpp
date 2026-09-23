#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace feedanns::adapters {
enum class Metric { l2, mips, cosine };
inline Metric parse_metric(const std::string &value) {
    if (value == "l2")
        return Metric::l2;
    if (value == "mips" || value == "ip")
        return Metric::mips;
    if (value == "cosine" || value == "cos")
        return Metric::cosine;
    throw std::invalid_argument("Metric must be l2, mips or cosine");
}
inline double norm(const float *x, std::size_t d) {
    double sum = 0;
    for (std::size_t j = 0; j < d; ++j)
        sum += double(x[j]) * x[j];
    return std::sqrt(sum);
}
// Dataset-level max_base_norm is fitted without queries or ground truth.
// Positive translation preserves MIPS ordering; it DOES affect log gains.
struct QueryMetric {
    Metric metric;
    double shift = 0;
    QueryMetric(Metric m, const float *q, std::size_t d, double max_base_norm) : metric(m) {
        if (m == Metric::mips) {
            if (!std::isfinite(max_base_norm) || max_base_norm < 0)
                throw std::invalid_argument(
                    "Set the maximum base-vector norm before MIPS FeedSearch");
            shift =
                std::max(1e-30, max_base_norm * norm(q, d) *
                                    (1 + 8 * d * double(std::numeric_limits<float>::epsilon())));
        }
    }
    double positive(float raw, double native_offset) const {
        if (metric == Metric::l2)
            return raw;
        const double x = metric == Metric::mips ? shift + (double(raw) - native_offset)
                                                : 1 + (double(raw) - native_offset);
        if (metric == Metric::cosine && x >= -1e-4)
            return std::max(0.0, x);
        if (x < 0 || !std::isfinite(x))
            throw std::domain_error("Invalid metric conversion / insufficient MIPS norm bound");
        return x;
    }
    double public_score(double positive_distance) const {
        return metric == Metric::mips ? shift - positive_distance : positive_distance;
    }
};
inline void normalize_query(const float *q, std::size_t d, float *out) {
    const double n = norm(q, d);
    if (!(n > 0) || !std::isfinite(n))
        throw std::domain_error("Cosine requires a finite nonzero query");
    const float inv = static_cast<float>(1 / n);
    for (std::size_t j = 0; j < d; ++j)
        out[j] = q[j] * inv;
}
} // namespace feedanns::adapters
