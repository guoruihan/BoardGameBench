// Exact public-state encoder, shared by prototype and native PPO collector.
// Layout/version is pinned by v129.encoding. No fast-math or hidden supply.
#include <algorithm>
#include <cstddef>
#include <cstdint>

extern "C" int harmonies_encode(std::size_t count, const std::int16_t* states,
                               const float* card_features, const std::int16_t* cubes,
                               int max_cubes, float* features) {
    if (max_cubes <= 0) return -1;
    constexpr int width = 161, dimension = 1346, card_dimension = 83;
    for (std::size_t row = 0; row < count; ++row) {
        const auto* s = states + row * width;
        auto* x = features + row * dimension;
        std::fill(x, x + dimension, 0.f);
        int p = 0;
        for (int i = 0; i < 23; ++i) {
            if (s[i] < 0 || s[i] >= 15 || s[23+i] < 0 || s[23+i] > 32) return -2;
            x[p+s[i]] = 1.f;
            x[p+15] = s[23+i] != 0;
            x[p+16] = static_cast<double>(s[23+i]) / 32;
            p += 17;
        }
        for (int i = 46; i < 70; ++i) x[p++] = static_cast<double>(s[i]) / 3;
        for (int i = 70; i < 82; ++i) x[p++] = static_cast<double>(s[i]) / 23;
        for (int i = 82; i < 85; ++i) {
            if (s[i] < 0 || s[i] > 32) return -3;
            x[p+s[i]] = 1.f;
            p += 33;
        }
        for (int i = 0; i < 32; ++i) {
            const int placed = s[93+i];
            if (cubes[i+1] <= 0) return -4;
            x[p++] = placed >= 0;
            x[p++] = placed >= 0 ? static_cast<double>(placed) / cubes[i+1] : 0.;
            x[p++] = s[125+i];
        }
        x[p++] = static_cast<double>(s[157]) / 13;
        x[p++] = s[158]; x[p++] = s[159]; x[p++] = s[160];
        for (int i = 0; i < 4; ++i) {
            const int card = s[85+i];
            if (card < 0 || card > 32) return -5;
            x[p+card] = 1.f;
            p += 33;
            x[p++] = card ? static_cast<double>(s[89+i]) / max_cubes : 0.;
        }
        for (int i = 0; i < 7; ++i) {
            const int card = i < 4 ? s[85+i] : s[82+i-4];
            const auto* start = card_features + card * card_dimension;
            std::copy(start, start + card_dimension, x+p);
            p += card_dimension;
        }
        // Last three time-conditioning fields remain zero, as in current PPO.
        if (p + 3 != dimension) return -6;
    }
    return 0;
}
