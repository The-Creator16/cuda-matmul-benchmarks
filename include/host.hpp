#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {
struct Shape {
    int m, n, k;
};

inline int positive_int(const std::string& text) {
    if(text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("Expected a positive integer: " + text);
    }
    std::size_t used = 0;
    long long value = std::stoll(text, &used);
    if(used != text.size() || value < 1 || value > 16384) {
        throw std::runtime_error("Integer must be in [1, 16384]: " + text);
    }
    return static_cast<int>(value);
}

inline std::size_t elements(int rows, int cols) {
    return static_cast<std::size_t>(rows) * cols;
}

inline double flops(Shape s) {
    return 2.0 * s.m * s.n * s.k;
}

// Map mt19937's specified integer output explicitly; avoid implementation-specific
// uniform_real_distribution sequences. Values are in [-1, 1).
inline void fill_inputs(std::vector<float>& a, std::vector<float>& b,
                        Shape s, unsigned seed, const std::string& pattern) {
    std::mt19937 rng(seed);
    if(pattern == "random") {
        for(auto* v : {&a, &b}) {
            for(float& x : *v) {
                x = static_cast<float>(rng() >> 8) / 8388608.0f - 1.0f;
            }
        }
    } else if(pattern == "zero" || pattern == "ones") {
        float value = pattern == "zero" ? 0.0f : 1.0f;
        std::fill(a.begin(), a.end(), value);
        std::fill(b.begin(), b.end(), value);
    } else if(pattern == "identity") {
        std::fill(a.begin(), a.end(), 0.0f);
        for(int i=0; i<std::min(s.m, s.k); i=i+1) {
            a[static_cast<std::size_t>(i)*s.k+i] = 1.0f;
        }
        for(float& x : b) {
            x = static_cast<float>(rng() >> 8) / 8388608.0f - 1.0f;
        }
    } else if(pattern == "alternating") {
        for(int i=0; i<s.m; i=i+1) {
            for(int k=0; k<s.k; k=k+1) {
                a[static_cast<std::size_t>(i)*s.k+k] = k%2 ? -1.0f : 1.0f;
            }
        }
        std::fill(b.begin(), b.end(), 1.0f);
    } else {
        throw std::runtime_error("Unknown input pattern: " + pattern);
    }
}

inline double reference_at(const std::vector<float>& a, const std::vector<float>& b,
                           Shape s, std::size_t index) {
    std::size_t row = index/s.n, col = index%s.n;
    double sum = 0.0;
    for(int k=0; k<s.k; k=k+1) {
        sum += static_cast<double>(a[row*s.k+k]) * b[static_cast<std::size_t>(k)*s.n+col];
    }
    return sum;
}

struct Error {
    std::size_t checked = 0, failed = 0;
    double max_abs = 0.0, max_scaled = 0.0;
    void add(double actual, double expected) {
        ++checked;
        if(!std::isfinite(actual) || !std::isfinite(expected)) {
            ++failed;
            max_abs = max_scaled = INFINITY;
            return;
        }
        double delta = std::abs(actual-expected);
        // An absolute term is necessary for results close to zero.
        double scaled = delta/(1e-3 + 1e-4*std::abs(expected));
        max_abs = std::max(max_abs, delta);
        max_scaled = std::max(max_scaled, scaled);
        failed += scaled > 1.0;
    }
};

inline std::vector<std::size_t> cpu_indices(Shape s, unsigned seed) {
    std::size_t count = elements(s.m, s.n);
    std::vector<std::size_t> indices;
    if(static_cast<double>(s.m)*s.n*s.k <= 4'000'000) {
        for(std::size_t i=0; i<count; i=i+1) indices.push_back(i);
    } else {
        // Include corners and central positions, plus a reproducible sample.
        indices = {0, count-1, static_cast<std::size_t>(s.n-1),
                   static_cast<std::size_t>(s.m-1)*s.n, count/2};
        std::mt19937 rng(seed);
        while(indices.size() < std::min<std::size_t>(256, count)) {
            indices.push_back(rng()%count);
            std::sort(indices.begin(), indices.end());
            indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        }
    }
    return indices;
}
} // namespace bench
