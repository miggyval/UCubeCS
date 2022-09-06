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
    void translate(float* dst, float* src, float* p, uint N);
    MetalTransformer(MTL::Device* device);
    int init();
       
private:

    MTL::Device* _device;
    MTL::CommandQueue* _CommandQueue;
    MTL::ComputePipelineState* _rotateFunctionPSO;
    MTL::ComputePipelineState* _translateFunctionPSO;
    
};
#endif

#ifdef __gnu_linux__
class CudaTransformer{
public:
    CudaTransformer();
    void rotate(float* dst, float* src, float* q, uint N);
    void translate(float* dst, float* src, float* p, uint N);
};

void rotate_helper(float* dst, float* src, float* q, uint N);
void translate_helper(float* dst, float* src, float* q, uint N);

#endif

#endif