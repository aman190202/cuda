#include <iostream>
#include <vector>
#include <cmath>
#include "src/vec.h"
#include "src/renderer.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "src/stb_image_write.h"

struct color{
    float r, g, b;
};


// CUDA kernel to generate the image
__global__ void generateImage(color* image, int width, int height) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        // Convert pixel coordinates to normalized device coordinates (-1 to 1)
        float x = 2.0f * (static_cast<float>(idx % width) / width) - 1.0f;
        float y = 2.0f * (static_cast<float>(idx / width) / height) - 1.0f;

        vec3 ray_origin = vec3{0, 0, 0};
        // Create a proper view plane at z = -1
        vec3 ray_direction = vec3{x, y, -1};
        normalize(ray_direction);

        vec3 pixel_color = vec3{0, 0, 0};
        pixel_color = trace_ray(ray_origin, ray_direction);

        image[idx].r = pixel_color.x;       
        image[idx].g = pixel_color.y;
        image[idx].b = pixel_color.z;
    }
}

void saveImage(const std::vector<color>& image, int width, int height, const char* filename)
 {
    // Convert float colors to unsigned char (0-255)
    std::vector<unsigned char> data(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        data[i * 3] = static_cast<unsigned char>(image[i].r * 255.0f);
        data[i * 3 + 1] = static_cast<unsigned char>(image[i].g * 255.0f);
        data[i * 3 + 2] = static_cast<unsigned char>(image[i].b * 255.0f);
    }
    
    // Save as PNG
    stbi_write_png(filename, width, height, 3, data.data(), width * 3);
}

int main()
{
    int width = 1000;
    int height = 1000;
    int total_pixels = width * height;
    
    // Allocate memory on host
    std::vector<color> Image(total_pixels);
    
    // Allocate memory on device
    color* d_image;
    cudaMalloc(&d_image, total_pixels * sizeof(color));
    
    // Calculate grid and block dimensions
    int blockSize = 256;
    int numBlocks = (total_pixels + blockSize - 1) / blockSize;
    
    // Launch kernel
    generateImage<<<numBlocks, blockSize>>>(d_image, width, height);
    
    // Copy result back to host
    cudaMemcpy(Image.data(), d_image, total_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    
    // Save the image
    saveImage(Image, width, height, "output/output.png");
    
    // Cleanup
    cudaFree(d_image);
    
    return 0;
}