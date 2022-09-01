#ifndef transform_render_hpp
#define transform_render_hpp

#ifdef __APPLE__
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

#ifdef __APPLE__

class MetalTransformer{
public:
    
    void rotate(float* dst, float* src, float* q, uint N);
    MetalTransformer(MTL::Device* device);
    int init();
       
private:

    MTL::Device* _device;
    MTL::CommandQueue* _CommandQueue;
    MTL::ComputePipelineState* _addFunctionPSO;
    
};
#endif

#ifdef __gnu_linux__
class CudaRenderer{
public:
    CudaRenderer();
    void render_vertices(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf);
};

void render_helper(uint8_t* data, float* zbuffer, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf);

#endif

#endif