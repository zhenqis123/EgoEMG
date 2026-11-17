// Camera-space local-region rasterization, double intermediates/float32 output.
// Intentionally no fast-math or FMA: matches the NumPy reference arithmetic.
#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" void rasterize_regions_cpu(
    const double* uv, const double* z, const int64_t* faces,
    const int16_t* labels, int64_t n_faces, int height, int width, float* depth) {
    for (int64_t i = 0; i < n_faces; ++i) {
        const int64_t ia = faces[3*i], ib = faces[3*i+1], ic = faces[3*i+2];
        const double ax = uv[2*ia], ay = uv[2*ia+1];
        const double bx = uv[2*ib], by = uv[2*ib+1];
        const double cx = uv[2*ic], cy = uv[2*ic+1];
        const double min_x = std::floor(std::min({ax, bx, cx}));
        const double max_x = std::ceil(std::max({ax, bx, cx}));
        const double min_y = std::floor(std::min({ay, by, cy}));
        const double max_y = std::ceil(std::max({ay, by, cy}));
        if (min_x >= width || max_x < 0 || min_y >= height || max_y < 0) continue;
        const int x0 = static_cast<int>(std::max(0.0, min_x));
        const int x1 = static_cast<int>(std::min(double(width-1), max_x));
        const int y0 = static_cast<int>(std::max(0.0, min_y));
        const int y1 = static_cast<int>(std::min(double(height-1), max_y));
        const double denom = (by-cy)*(ax-cx) + (cx-bx)*(ay-cy);
        if (std::abs(denom) < 1e-10) continue;
        float* target = depth + int64_t(labels[i])*height*width;
        for (int y = y0; y <= y1; ++y) {
            const double py = y + 0.5;
            for (int x = x0; x <= x1; ++x) {
                const double px = x + 0.5;
                const double wa = ((by-cy)*(px-cx) + (cx-bx)*(py-cy))/denom;
                const double wb = ((cy-ay)*(px-cx) + (ax-cx)*(py-cy))/denom;
                const double wc = 1.0 - wa - wb;
                if (wa < -1e-7 || wb < -1e-7 || wc < -1e-7) continue;
                const double inv_z = wa/z[ia] + wb/z[ib] + wc/z[ic];
                if (inv_z <= 0) continue;
                const double value = 1.0/inv_z;
                float& pixel = target[int64_t(y)*width+x];
                if (value < pixel) pixel = static_cast<float>(value);
            }
        }
    }
}
