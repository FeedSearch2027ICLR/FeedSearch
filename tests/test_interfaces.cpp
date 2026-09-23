#include "feedsearch/core.hpp"
#include "feedsearch/metrics.hpp"
#include <iostream>

void test_unrestricted_frontier() {
    // Query=0 in 1D squared L2. The nearest answer is reachable only through
    // a point farther than both the entry and the already pending local point.
    const std::vector<double> distance{4, 1, 9, .01};
    for (bool reverse : {false, true}) {
        std::vector<std::vector<unsigned>> graph{{1, 2}, {}, {3}, {}};
        if (reverse) graph[0] = {2, 1};
        feedanns::FeedbackController controller(12);
        std::size_t calls = 0;
        const auto out = feedanns::FeedSearch(
            std::vector<unsigned>{0, 0}, 1, graph.size(),
            [&](unsigned id, auto visit) {
                for (auto nb : graph[id]) if (!visit(nb)) break;
            },
            [&](unsigned id) { ++calls; return distance[id]; }, controller);
        if (out.neighbors.size() != 1 || out.neighbors[0].id != 3 ||
            calls != 4 || out.distance_computations != calls ||
            out.c_admissions != calls || out.c_rejections != 0 ||
            out.stop_reason != feedanns::StopReason::frontier_exhausted)
            throw std::runtime_error("Unrestricted frontier / duplicate entry regression");
    }
}

void test_immediate_stop() {
    // After one positive improvement, the next non-improving comparison makes
    // the Lambda=0 controller stop. No later neighbor may be evaluated.
    const std::vector<std::vector<unsigned>> graph{{1, 2, 3}, {}, {}, {}};
    const double distance[]{4, 1, 2, .01};
    feedanns::FeedbackController controller(0);
    std::size_t calls = 0;
    const auto out = feedanns::FeedSearch(
        std::vector<unsigned>{0}, 1, graph.size(),
        [&](unsigned id, auto visit) {
            for (auto nb : graph[id]) if (!visit(nb)) break;
        },
        [&](unsigned id) {
            if (id == 3) throw std::runtime_error("Distance evaluated after online stop");
            ++calls; return distance[id];
        }, controller);
    if (out.neighbors[0].id != 1 || calls != 3 ||
        out.stop_reason != feedanns::StopReason::adaptive_feedback)
        throw std::runtime_error("Immediate online termination regression");
}

void test_zero_gain_safeguard() {
    // Exercise both lazy decay and the eager fallback.
    for (double beta1 : {0.9, 0.995}) {
        feedanns::FeedbackController c(12, beta1, 0.99);
        for (std::size_t i = 1; i <= 10000; ++i) {
            bool stopped = i % 3 == 0 ? c.on_no_change() :
                (i % 3 == 1 ? c.on_replacement(1.0, 1.0, 10) :
                              c.on_replacement(1e-31, 1e-32, 10));
            if (stopped || c.step() != i || c.no_change_steps() != i)
                throw std::runtime_error("Zero-gain replacement reset safeguard");
        }
        if (!c.on_gain(0.0) || c.no_change_steps() != 10001)
            throw std::runtime_error("Zero-scale safeguard boundary regression");
        c.reset();
        for (int i = 0; i < 100; ++i) c.on_gain(0.0);
        c.on_gain(1.0);
        if (c.no_change_steps() != 0 || c.step() != 101)
            throw std::runtime_error("Positive gain failed to reset safeguard");
        auto reference = c;
        for (int i = 0; i < 100; ++i) {
            const bool actual = c.on_replacement(2.0, 2.0, 10);
            const bool expected = reference.on_no_change();
            if (actual != expected || c.momentum() != reference.momentum() ||
                c.scale() != reference.scale() || c.no_change_steps() != reference.no_change_steps())
                throw std::runtime_error("Zero-gain decay differs from quiet step");
            if (actual) break;
        }
    }
}

int main() {
    test_zero_gain_safeguard();
    test_unrestricted_frontier();
    test_immediate_stop();
    using namespace feedanns::adapters;
    const float q[]{2, -1};
    QueryMetric metric(Metric::mips, q, 2, 10);
    if (!(metric.positive(-4, 0) < metric.positive(3, 0)))
        throw std::runtime_error("MIPS ranking reversal");
    if (std::abs(metric.public_score(metric.positive(-4, 0)) - 4) > 1e-12)
        throw std::runtime_error("MIPS score conversion");
    if (metric.positive(-3, 1) != metric.positive(-4, 0))
        throw std::runtime_error("HNSW inner product offset");
    float normalized[2];
    normalize_query(q, 2, normalized);
    if (std::abs(norm(normalized, 2) - 1) > 1e-6)
        throw std::runtime_error("Cosine normalization");
    const std::vector<std::vector<unsigned>> graph{{1, 2}, {}, {3}, {}};
    const std::vector<float> neg_dot{0, -1, 2, -4};
    feedanns::FeedbackController controller(12);
    auto out = feedanns::FeedSearch(
        // Request all reachable points to check MIPS conversion and bridging.
        std::vector<unsigned>{0}, graph.size(), graph.size(),
        [&](unsigned id, auto visit) {
            for (auto nb : graph[id])
                if (!visit(nb))
                    break;
        },
        [&](unsigned id) { return metric.positive(neg_dot[id], 0); }, controller);
    if (out.neighbors.size() != graph.size() || out.neighbors[0].id != 3 ||
        out.distance_computations != 4)
        throw std::runtime_error("MIPS graph bridge search");
    std::cout << "PASS: signed MIPS ordering/score, offset, cosine, graph bridge, "
                 "unrestricted frontier in both neighbor orders, duplicate entries, immediate stop\n";
}
