#include <metal_stdlib>

using namespace metal;

kernel void rotate(device float* dst, device const float* src, device const float* q, device const float* N, uint index [[thread_position_in_grid]]) {
    int n = index;
    float w = q[0];
    float x = q[1];
    float y = q[2];
    float z = q[3];

    float vw = 0;
    float vx = src[3 * n + 0];
    float vy = src[3 * n + 1];
    float vz = src[3 * n + 2];
    
    float rw = w * vw - x * vx - y * vy - z * vz;
    float rx = w * vx + x * vw + y * vz - z * vy;
    float ry = w * vy - x * vz + y * vw + z * vx;
    float rz = w * vz + x * vy - y * vx + z * vw;

    dst[3 * n + 0] = rw * -x + rx * w + ry * -z - rz * -y;
    dst[3 * n + 1] = rw * -y - rx * -z + ry * w + rz * -x;
    dst[3 * n + 2] = rw * -z + rx * -y - ry * -x + rz * w;

}

kernel void translate(device float* dst, device const float* src, device const float* p, device const float* N, uint index [[thread_position_in_grid]]) {
    int n = index;
    for (int i = 0; i < 3; i++) {
        dst[3 * n + i] = src[3 * n + i] + p[i];
    }
}