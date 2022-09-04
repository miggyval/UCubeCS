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
class CudaTransformer{
public:
    CudaTransformer();
    void rotate(float* dst, float* src, float* q, uint N);
};

void rotate_helper(float* dst, float* src, float* q, uint N);

#endif

#endif