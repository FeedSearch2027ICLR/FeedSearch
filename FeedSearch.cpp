#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace feedanns {

// -----------------------------------------------------------------------------
// Feed / Adam-style termination controller.
// After the visible top-k is initialized, every exact distance computation is
// one feedback step:
//   - positive top-k potential improvement -> positive gain
//   - no improvement (including zero-gain replacement) -> zero-gain decay
// Search stops when the normalized progress falls below 10^{-Lambda}.
// -----------------------------------------------------------------------------
class FeedbackController {
public:
    explicit FeedbackController(double lambda_exponent,
                                double beta1 = 0.9,
                                double beta2 = 0.99,
                                double delta = 1e-30)
        : lambda_exponent_(lambda_exponent),
          threshold_(std::pow(10.0, -lambda_exponent)),
          beta1_(beta1),
          beta2_(beta2),
          one_minus_beta1_(1.0 - beta1),
          delta_(delta),
          lazy_quiet_supported_(beta1 < beta2),
          log_beta1_(std::log(beta1)),
          log_beta2_(std::log(beta2)),
          log_threshold_(std::log(threshold_)) {
        validate_parameters();
        reset();
    }

    void reset() noexcept {
        step_ = 0;
        no_change_steps_ = 0;
        pending_quiet_steps_ = 0;
        quiet_stop_after_ = std::numeric_limits<std::size_t>::max();
        momentum_ = 0.0;
        scale_ = 0.0;
        beta1_pow_ = 1.0;
    }

    // A no-change distance remains a full logical Adam step, but for the
    // default beta1 < beta2 case we defer the floating-point decay.  The hot
    // path is therefore only integer increments plus one comparison.
    bool on_no_change() noexcept {
        ++step_;
        ++no_change_steps_;

        if (!lazy_quiet_supported_) {
            // Exact eager fallback for unusual beta1 >= beta2 settings, where
            // the quiet-step score is not guaranteed to decrease monotonically.
            beta1_pow_ *= beta1_;
            momentum_ *= beta1_;
            scale_ *= beta2_;

            if (scale_ <= 0.0 &&
                no_change_steps_ > zero_scale_no_change_limit()) {
                return true;
            }
            return should_stop_materialized();
        }

        // Even before the first positive gain, zero-gain steps must advance the
        // bias-correction clock. We record them lazily as well; when scale is
        // still zero the only stopping rule is the original 10000-step guard.
        ++pending_quiet_steps_;
        if (scale_ <= 0.0) {
            return no_change_steps_ > zero_scale_no_change_limit();
        }

        return pending_quiet_steps_ >= quiet_stop_after_;
    }

    bool on_replacement(double removed_distance,
                        double inserted_distance,
                        std::size_t k) {
        return on_gain(replacement_gain(
            removed_distance, inserted_distance, k, delta_));
    }

    bool on_gain(double gain) {
        if (!std::isfinite(gain) || gain < 0.0) {
            throw std::invalid_argument(
                "FeedbackController gain must be finite and nonnegative");
        }

        // Membership changes with tied or floor-clipped distances are still
        // zero progress, including for the zero-scale safeguard.
        if (gain == 0.0) {
            return on_no_change();
        }

        // Materialize all deferred zero-gain steps once, immediately before a
        // non-quiet event. This is algebraically equivalent to applying the
        // original per-step recurrences one quiet step at a time.
        flush_pending_quiet();

        ++step_;
        no_change_steps_ = 0;
        beta1_pow_ *= beta1_;
        momentum_ = beta1_ * momentum_ + one_minus_beta1_ * gain;
        scale_ = std::max(beta2_ * scale_, gain);

        if (should_stop_materialized()) {
            quiet_stop_after_ = 0;
            return true;
        }

        recompute_quiet_stop_after();
        return false;
    }

    static double replacement_gain(double removed_distance,
                                   double inserted_distance,
                                   std::size_t k,
                                   double delta = 1e-30) {
        if (k == 0) {
            throw std::invalid_argument(
                "FeedbackController replacement requires k > 0");
        }
        if (!std::isfinite(removed_distance) ||
            !std::isfinite(inserted_distance) ||
            removed_distance < 0.0 || inserted_distance < 0.0 ||
            !std::isfinite(delta) || delta <= 0.0) {
            throw std::invalid_argument(
                "FeedbackController ranking values must be finite and nonnegative");
        }
        if (inserted_distance > removed_distance) {
            throw std::invalid_argument(
                "inserted_distance must be <= removed_distance");
        }

        const double d_out = std::max(removed_distance, delta);
        const double d_in  = std::max(inserted_distance, delta);
        const double gain =
            (std::log(d_out) - std::log(d_in)) / static_cast<double>(k);
        return std::max(0.0, gain);
    }

    double lambda_exponent() const noexcept { return lambda_exponent_; }
    double threshold() const noexcept { return threshold_; }
    double beta1() const noexcept { return beta1_; }
    double beta2() const noexcept { return beta2_; }
    double delta() const noexcept { return delta_; }

    std::size_t step() const noexcept { return step_; }
    std::size_t no_change_steps() const noexcept { return no_change_steps_; }
    static constexpr std::size_t zero_scale_no_change_limit() noexcept {
        return 10000;
    }

    // These accessors expose the logical current state, including any deferred
    // quiet-step decay. They may use pow(), but they are diagnostics rather than
    // the search hot path.
    double momentum() const noexcept {
        if (!lazy_quiet_supported_ || pending_quiet_steps_ == 0)
            return momentum_;
        return momentum_ * std::pow(beta1_,
                                    static_cast<double>(pending_quiet_steps_));
    }

    double scale() const noexcept {
        if (!lazy_quiet_supported_ || pending_quiet_steps_ == 0)
            return scale_;
        return scale_ * std::pow(beta2_,
                                 static_cast<double>(pending_quiet_steps_));
    }

    double beta1_power() const noexcept {
        if (!lazy_quiet_supported_ || pending_quiet_steps_ == 0)
            return beta1_pow_;
        return beta1_pow_ * std::pow(beta1_,
                                     static_cast<double>(pending_quiet_steps_));
    }

    double score() const noexcept {
        const double s = scale();
        if (s <= 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        const double b1p = beta1_power();
        const double correction = 1.0 - b1p;
        if (correction <= 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        return momentum() / (correction * s);
    }

private:
    bool should_stop_materialized() const noexcept {
        if (scale_ <= 0.0) return false;
        const double correction = 1.0 - beta1_pow_;
        return momentum_ < threshold_ * correction * scale_;
    }

    void flush_pending_quiet() noexcept {
        if (!lazy_quiet_supported_ || pending_quiet_steps_ == 0) return;

        const double n = static_cast<double>(pending_quiet_steps_);
        const double b1n = std::pow(beta1_, n);
        const double b2n = std::pow(beta2_, n);

        beta1_pow_ *= b1n;
        momentum_ *= b1n;
        scale_ *= b2n;
        pending_quiet_steps_ = 0;
    }

    // Evaluate the original stop test after exactly n additional zero-gain
    // steps, without mutating state. For beta1 < beta2 this predicate is
    // monotone, which lets us precompute the exact quiet-step stopping point.
    bool would_stop_after_quiet(std::size_t n) const noexcept {
        if (scale_ <= 0.0) return false;
        if (momentum_ <= 0.0) return true;

        const double nd = static_cast<double>(n);
        const double log_b1n = nd * log_beta1_;
        const double b1n = std::exp(log_b1n);
        const double correction = 1.0 - beta1_pow_ * b1n;
        if (correction <= 0.0) return false;

        // Compare in log space to avoid underflow for long quiet runs:
        // m*beta1^n < threshold * correction * u*beta2^n.
        const double lhs = std::log(momentum_) + nd * log_beta1_;
        const double rhs = log_threshold_ + std::log(correction) +
                           std::log(scale_) + nd * log_beta2_;
        return lhs < rhs;
    }

    void recompute_quiet_stop_after() noexcept {
        pending_quiet_steps_ = 0;
        quiet_stop_after_ = std::numeric_limits<std::size_t>::max();

        if (!lazy_quiet_supported_ || scale_ <= 0.0) return;
        if (should_stop_materialized()) {
            quiet_stop_after_ = 0;
            return;
        }

        // Find an upper bound exponentially. In normal FeedSearch settings the
        // answer is small (typically tens to hundreds of distance steps), so
        // this work is paid only on relatively rare top-k replacements.
        std::size_t hi = 1;
        const std::size_t maxv = std::numeric_limits<std::size_t>::max();
        while (!would_stop_after_quiet(hi)) {
            if (hi > maxv / 2) return;
            hi *= 2;
        }

        std::size_t lo = 0;
        while (lo + 1 < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (would_stop_after_quiet(mid))
                hi = mid;
            else
                lo = mid;
        }
        quiet_stop_after_ = hi;
    }

    void validate_parameters() const {
        if (!std::isfinite(lambda_exponent_)) {
            throw std::invalid_argument(
                "FeedbackController lambda_exponent must be finite");
        }
        if (!std::isfinite(threshold_) || threshold_ <= 0.0) {
            throw std::invalid_argument(
                "FeedbackController threshold must be finite and positive");
        }
        if (!std::isfinite(beta1_) || !std::isfinite(beta2_) ||
            beta1_ <= 0.0 || beta1_ >= 1.0 ||
            beta2_ <= 0.0 || beta2_ >= 1.0) {
            throw std::invalid_argument(
                "FeedbackController requires 0 < beta1,beta2 < 1");
        }
        if (!std::isfinite(delta_) || delta_ <= 0.0) {
            throw std::invalid_argument(
                "FeedbackController delta must be finite and positive");
        }
    }

    double lambda_exponent_;
    double threshold_;
    double beta1_;
    double beta2_;
    double one_minus_beta1_;
    double delta_;

    bool lazy_quiet_supported_ = true;
    double log_beta1_ = 0.0;
    double log_beta2_ = 0.0;
    double log_threshold_ = 0.0;

    std::size_t step_ = 0;
    std::size_t no_change_steps_ = 0;
    std::size_t pending_quiet_steps_ = 0;
    std::size_t quiet_stop_after_ = std::numeric_limits<std::size_t>::max();

    // Materialized state at the last non-quiet event. Deferred quiet steps are
    // represented only by pending_quiet_steps_.
    double momentum_ = 0.0;
    double scale_ = 0.0;
    double beta1_pow_ = 1.0;
};

enum class StopReason {
    adaptive_feedback,
    frontier_exhausted,
    comparison_budget,
    zero_k
};

template<class Id>
struct Neighbor {
    Id id{};
    double distance = 0.0;
};

template<class Id>
struct FeedSearchResult {
    std::vector<Neighbor<Id>> neighbors;

    std::size_t distance_computations = 0;
    std::size_t expansions = 0;
    std::size_t topk_replacements = 0;

    // Admission statistics; all discovered vertices are admitted.
    // max_c_size retains the cumulative historical-C size for API compatibility.
    std::size_t c_admissions = 0;
    std::size_t c_rejections = 0;
    std::size_t max_c_size = 0;

    StopReason stop_reason = StopReason::frontier_exhausted;
};

struct FeedSearchConfig {
    // Safety guard only. There is no W and no ef/width parameter.
    std::size_t max_comparisons = std::numeric_limits<std::size_t>::max();
};

namespace detail {

template<class Id>
struct CandidateItem {
    double distance = 0.0;
    Id id{};
};

template<class Id>
struct ResultItem {
    double distance = 0.0;
    Id id{};
};

// Pending subset of C: nearest unexpanded node at heap front.
template<class Id>
struct CandidateMin {
    bool operator()(const CandidateItem<Id>& a,
                    const CandidateItem<Id>& b) const noexcept {
        if (a.distance != b.distance) return a.distance > b.distance;
        return static_cast<std::size_t>(a.id) > static_cast<std::size_t>(b.id);
    }
};

// Visible top-k: worst result at heap front.
template<class Id>
struct ResultMax {
    bool operator()(const ResultItem<Id>& a,
                    const ResultItem<Id>& b) const noexcept {
        if (a.distance != b.distance) return a.distance < b.distance;
        return static_cast<std::size_t>(a.id) < static_cast<std::size_t>(b.id);
    }
};

template<class T, class Comp>
inline void heap_push(std::vector<T>& h, const T& x, Comp comp) {
    h.push_back(x);
    std::push_heap(h.begin(), h.end(), comp);
}

template<class T, class Comp>
inline T heap_pop(std::vector<T>& h, Comp comp) {
    std::pop_heap(h.begin(), h.end(), comp);
    T x = h.back();
    h.pop_back();
    return x;
}

} // namespace detail

template<class Id>
class FeedWorkspace {
    static_assert(std::is_integral<Id>::value,
                  "FeedWorkspace requires integral dense graph IDs");

public:
    FeedWorkspace() = default;
    explicit FeedWorkspace(std::size_t node_count) { resize(node_count); }

    void resize(std::size_t node_count) {
        seen_stamp_.resize(node_count, 0);
    }

private:
    template<class I, class E, class D>
    friend FeedSearchResult<I> FeedSearch(
        const std::vector<I>&,
        std::size_t,
        std::size_t,
        FeedWorkspace<I>&,
        E&&,
        D&&,
        FeedbackController&,
        const FeedSearchConfig&);

    void begin_query(std::size_t node_count, std::size_t k) {
        if (seen_stamp_.size() < node_count) resize(node_count);

        ++query_stamp_;
        if (query_stamp_ == 0) {
            std::fill(seen_stamp_.begin(), seen_stamp_.end(), 0);
            query_stamp_ = 1;
        }

        pending_.clear();
        topk_.clear();

        pending_.reserve(std::max<std::size_t>(128, k * 8));
        topk_.reserve(k + 1);
    }

    std::vector<std::uint32_t> seen_stamp_;
    std::uint32_t query_stamp_ = 0;

    // All discovered, unexpanded nodes; no distance or width admission bound.
    std::vector<detail::CandidateItem<Id>> pending_;

    // Visible top-k observer used for return values and Feed termination.
    std::vector<detail::ResultItem<Id>> topk_;
};

template<class Id, class EnumerateNeighbors, class DistanceToQuery>
FeedSearchResult<Id> FeedSearch(
    const std::vector<Id>& entry_points,
    std::size_t k,
    std::size_t node_count,
    FeedWorkspace<Id>& ws,
    EnumerateNeighbors&& enumerate_neighbors,
    DistanceToQuery&& distance_to_query,
    FeedbackController& feedback,
    const FeedSearchConfig& cfg = {}) {

    static_assert(std::is_integral<Id>::value,
                  "FeedSearch requires integral dense graph IDs");

    FeedSearchResult<Id> out;
    out.neighbors.reserve(k);

    if (k == 0) {
        out.stop_reason = StopReason::zero_k;
        return out;
    }
    if (node_count == 0 || entry_points.empty()) {
        out.stop_reason = StopReason::frontier_exhausted;
        return out;
    }

    ws.begin_query(node_count, k);

    using Cand = detail::CandidateItem<Id>;
    using Res  = detail::ResultItem<Id>;
    const detail::CandidateMin<Id> cand_comp{};
    const detail::ResultMax<Id> res_comp{};

    auto& pending = ws.pending_;
    auto& topk = ws.topk_;

    const auto idx = [&](Id id) -> std::size_t {
        if constexpr (std::is_signed<Id>::value) {
            if (id < 0) throw std::out_of_range("negative graph ID");
        }
        const std::size_t x = static_cast<std::size_t>(id);
        if (x >= node_count) throw std::out_of_range("graph ID >= node_count");
        return x;
    };

    bool topk_ready = false;
    bool feedback_stop = false;
    bool budget_hit = false;

    const auto update_feedback_topk = [&](Id id, double d) {
        bool changed = false;
        double removed_distance = 0.0;

        if (topk.size() < k) {
            detail::heap_push(topk, Res{d, id}, res_comp);
            changed = true;

            if (topk.size() == k) {
                topk_ready = true;
                feedback.reset();
                return;
            }
        } else {
            const Res worst = topk.front();
            const bool better =
                d < worst.distance ||
                (d == worst.distance &&
                 static_cast<std::size_t>(id) <
                     static_cast<std::size_t>(worst.id));

            if (better) {
                removed_distance = worst.distance;
                (void)detail::heap_pop(topk, res_comp);
                detail::heap_push(topk, Res{d, id}, res_comp);
                changed = true;
                ++out.topk_replacements;
            }
        }

        if (!topk_ready) return;

        bool stop_now = false;
        if (changed) {
            stop_now = feedback.on_replacement(removed_distance, d, k);
        } else {
            stop_now = feedback.on_no_change();
        }

        if (stop_now) feedback_stop = true; // sticky stop
    };

    // Every newly evaluated vertex remains eligible for later expansion, even
    // when it is farther than the current top-k or every pending candidate.
    const auto admit_to_c = [&](Id id, double d) {
        detail::heap_push(pending, Cand{d, id}, cand_comp);
        ++out.c_admissions;
        // Preserve the public historical-C count without storing its members.
        out.max_c_size = out.c_admissions;
    };

    const auto discover = [&](Id id) -> bool {
        if (feedback_stop || budget_hit) return false;

        const std::size_t x = idx(id);
        if (ws.seen_stamp_[x] == ws.query_stamp_) return true;

        if (out.distance_computations >= cfg.max_comparisons) {
            budget_hit = true;
            return false;
        }

        const double d = static_cast<double>(distance_to_query(id));
        ++out.distance_computations;
        if (!std::isfinite(d) || d < 0.0) {
            throw std::domain_error(
                "FeedSearch requires finite nonnegative ranking values");
        }

        // A vertex is evaluated and inserted at most once per query.
        ws.seen_stamp_[x] = ws.query_stamp_;

        admit_to_c(id, d);
        // This distance contributes one feedback step after top-k initialization.
        // Stop immediately, including in the middle of a neighbor list.
        update_feedback_topk(id, d);
        return !feedback_stop;
    };

    // Seed the frontier; duplicate entry points are ignored.
    for (Id ep : entry_points) {
        if (!discover(ep)) break;
    }

    // Best-first expansion. There is no ef pool or worst-distance cutoff.
    while (!feedback_stop && !budget_hit && !pending.empty()) {
        const Cand current = detail::heap_pop(pending, cand_comp);
        ++out.expansions;

        enumerate_neighbors(current.id, [&](Id nid) -> bool {
            return discover(nid);
        });
    }

    if (feedback_stop) {
        out.stop_reason = StopReason::adaptive_feedback;
    } else if (budget_hit || out.distance_computations >= cfg.max_comparisons) {
        out.stop_reason = StopReason::comparison_budget;
    } else {
        out.stop_reason = StopReason::frontier_exhausted;
    }

    while (!topk.empty()) {
        const Res r = detail::heap_pop(topk, res_comp);
        out.neighbors.push_back(Neighbor<Id>{r.id, r.distance});
    }
    std::reverse(out.neighbors.begin(), out.neighbors.end());

    return out;
}

template<class Id, class EnumerateNeighbors, class DistanceToQuery>
FeedSearchResult<Id> FeedSearch(
    const std::vector<Id>& entry_points,
    std::size_t k,
    std::size_t node_count,
    EnumerateNeighbors&& enumerate_neighbors,
    DistanceToQuery&& distance_to_query,
    FeedbackController& feedback,
    const FeedSearchConfig& cfg = {}) {

    thread_local FeedWorkspace<Id> ws;
    return FeedSearch(
        entry_points,
        k,
        node_count,
        ws,
        std::forward<EnumerateNeighbors>(enumerate_neighbors),
        std::forward<DistanceToQuery>(distance_to_query),
        feedback,
        cfg);
}

} // namespace feedanns
