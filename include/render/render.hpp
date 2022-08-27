#ifndef metal_render_hpp
#define metal_render_hpp

#ifdef __APPLE__
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#endif

#ifdef __APPLE__
class MetalRenderer{
public:
    
    void render_vertices(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf);
    MetalRenderer(MTL::Device* device);
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

void render_helper(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf);

#endif

#endif