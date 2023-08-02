#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <render/render.hpp>

__global__ void rotate(float* dst, float* src, float* q, uint N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
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

}


__global__ void translate(float* dst, float* src, float* p, uint N) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < N) {
        int n = index;
        for (size_t i = 0; i < 3; i++) {
            dst[3 * n + i] = src[3 * n + i] + p[i];
        }
    }
}

void rotate_helper(float* dst, float* src, float* q, uint N) {
    int numElements = N;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    rotate<<<blocksPerGrid, threadsPerBlock>>>(dst, src, q, N);
}


void translate_helper(float* dst, float* src, float* p, uint N) {
    int numElements = N;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    translate<<<blocksPerGrid, threadsPerBlock>>>(dst, src, p, N);
}
