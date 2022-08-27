#ifndef metal_render_hpp
#define metal_render_hpp

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

class metal_renderer{
public:
    
    void helper_render(uint8_t* data, float* vertices, float* colors, uint32_t* faces, uint Nv, uint Nf);
    metal_renderer(MTL::Device* device);
    int init();
       
private:

    MTL::Device* _device;
    MTL::CommandQueue* _CommandQueue;
    MTL::ComputePipelineState* _addFunctionPSO;
    
};

#endif /* metal_adder_hpp */