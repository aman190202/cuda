#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "src/vec.h"
#include "src/renderer.h"
#include "src/light.h"
#include "src/sampler.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "src/stb_image_write.h"

constexpr float PI = 3.14159265358979323846f;

// Function to generate random lights
std::vector<light> generateRandomLights(int num_lights) {
    std::vector<light> lights;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-9.0f, 9.0f);  // Keep within box bounds
    std::uniform_real_distribution<float> color_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> intensity_dist(0.1f, 1.0f);

    for (int i = 0; i < num_lights; ++i) {
        light l;
        // Random position within the box
        l.position = vec3{
            pos_dist(gen),
            -9,
            pos_dist(gen)
        };
        
        // Random color
        l.col = vec3{
            1.f,
            color_dist(gen),
            color_dist(gen)
        };
        
        // Small random intensity
        l.intensity = intensity_dist(gen) * 0.01f;  // Scale down intensity
        
        lights.push_back(l);
    }
    
    return lights;
}

// Simple Reinhard tone mapping
__device__ __host__ vec3 toneMap(const vec3& color) {
    // Reinhard tone mapping
    vec3 mapped = color / (color + vec3{1.0f, 1.0f, 1.0f});
    // Gamma correction
    const float gamma = 2.2f;
    mapped.x = powf(mapped.x, 1.0f / gamma);
    mapped.y = powf(mapped.y, 1.0f / gamma);
    mapped.z = powf(mapped.z, 1.0f / gamma);
    return mapped;
}

// CUDA kernel to generate the image with a simple pinhole camera
__global__ void generateImage(color* image, int width, int height, light* lights, int num_lights) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    int i = idx % width;
    int j = idx / width;

    // set up camera
    float aspect = float(width) / float(height);
    float fov = PI / 3.0f;           // 60° field of view
    float scale = tanf(fov * 0.5f);

    // normalized device coordinates [−1,1]
    float u = (2.0f * (i + 0.5f) / float(width)  - 1.0f) * aspect * scale;
    float v = (1.0f - 2.0f * (j + 0.5f) / float(height)) * scale;

    vec3 ray_origin = vec3{0.0f, 0.0f, 30.0f};
    vec3 ray_direction = normalize(vec3{u, v, -1.0f});

    vec3 pixel_color = trace_ray(ray_origin, ray_direction, lights, num_lights);
    
    // Apply tone mapping
    pixel_color = toneMap(pixel_color);

    image[idx].r = pixel_color.x;
    image[idx].g = pixel_color.y;
    image[idx].b = pixel_color.z;
}

void saveImage(const std::vector<color>& image, int width, int height, const char* filename)
{
    // Convert float colors to unsigned char (0-255)
    std::vector<unsigned char> data(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        data[i * 3    ] = static_cast<unsigned char>(fminf(fmaxf(image[i].r, 0.0f), 1.0f) * 255.0f);
        data[i * 3 + 1] = static_cast<unsigned char>(fminf(fmaxf(image[i].g, 0.0f), 1.0f) * 255.0f);
        data[i * 3 + 2] = static_cast<unsigned char>(fminf(fmaxf(image[i].b, 0.0f), 1.0f) * 255.0f);
    }

    // Save as PNG
    stbi_write_png(filename, width, height, 3, data.data(), width * 3);
}


int main()
{
    int width  = 1000;
    int height = 1000;
    int total_pixels = width * height;
    std::string vdbFilePath = "v1/untitled.filecache1_v1.0092.vdb";

    initializeSampler(vdbFilePath);
    std::vector<light> lights = generateRandomLights(100000);
    int num_lights = static_cast<int>(lights.size());

    // Host image buffer
    std::vector<color> Image(total_pixels);

    

    // Allocate device memory
    color* d_image;
    light* d_lights;
    cudaMalloc(&d_image, total_pixels * sizeof(color));
    cudaMalloc(&d_lights, num_lights * sizeof(light));
    cudaMemcpy(d_lights, lights.data(), num_lights * sizeof(light), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (total_pixels + blockSize - 1) / blockSize;
    generateImage<<<numBlocks, blockSize>>>(d_image, width, height, d_lights, num_lights);
    cudaDeviceSynchronize();

    // Copy back and save
    cudaMemcpy(Image.data(), d_image, total_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    saveImage(Image, width, height, "output/output.png");

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_lights);
    return 0;
}