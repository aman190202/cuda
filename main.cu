#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <filesystem>
#include "src/vec.h"
#include "src/light.h"
#include "src/renderer.h"
// #include "src/vdb_reader.cu"
#include "src/vdb_reader.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "src/stb_image_write.h"

constexpr float PI = 3.14159265358979323846f;



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
__global__ void generateImage(color* image, int width, int height, light* lights, int num_lights, vec3 min, vec3 max, const char* vdb_file, density_sample* d_density_samples, int num_density_samples) 
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

    vec3 ray_origin = vec3{0.0f, 0.0f, 100.0f};
    vec3 ray_direction = normalize(vec3{u, v, -1.0f});

    // Sample density along the ray
    float step_size = 1.0f;
    float max_distance = 200.0f;
    float total_density = 0.0f;
    

    vec3 pixel_color = trace_ray(ray_origin, ray_direction, lights, num_lights, min, max, vdb_file, d_density_samples, num_density_samples);
    
    // Apply density-based attenuation
    float attenuation = exp(-total_density);
    pixel_color = pixel_color * attenuation;
    
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


int main(int argc, char* argv[])
{
    if (argc != 2) {    
        std::cerr << "Usage: " << argv[0] << " <vdb_file>" << std::endl;
        return 1;
    }

    const char* vdb_file = argv[1];
    readAndPrintVDB(vdb_file);
    int width  = 1000;
    int height = 1000;
    int total_pixels = width * height;

    std::vector<light> lights = getLightsFromVDB(vdb_file);
    std::cout << "Number of lights: " << lights.size() << std::endl;
    // float density = getDensityAtPosition(std::string(vdb_file), 0.0f, 0.0f, 0.0f);
    // std::cout << "Density at (0,0,0): " << density << std::endl;


    std::vector<density_sample> density_samples = getDensitySamplesFromVDB(vdb_file);
    std::cout << "Number of density samples: " << density_samples.size() << std::endl;  

    // Extract base filename without extension
    std::string input_file(vdb_file);
    size_t last_slash = input_file.find_last_of("/\\");
    std::string filename = (last_slash != std::string::npos) ? input_file.substr(last_slash + 1) : input_file;
    size_t last_dot = filename.find_last_of(".");
    std::string base_name = (last_dot != std::string::npos) ? filename.substr(0, last_dot) : filename;
    
    // Ensure output directory exists
    std::filesystem::create_directories("output");
    std::string output_file = "scenes/" + base_name + ".png";

    vec3 min, max;
    getScaledBoundingBox(vdb_file, min, max);
    std::cout << "Min: " << min << std::endl;
    std::cout << "Max: " << max << std::endl;   

    // Get density range
    float minDensity, maxDensity;
    vec3 minPos, maxPos;
    getDensityRange(vdb_file, minDensity, maxDensity, minPos, maxPos);
    std::cout << "Density range: [" << minDensity << ", " << maxDensity << "]" << std::endl;
    std::cout << "Min density position: (" << minPos.x << ", " << minPos.y << ", " << minPos.z << ")" << std::endl;
    std::cout << "Max density position: (" << maxPos.x << ", " << maxPos.y << ", " << maxPos.z << ")" << std::endl;


    int num_lights = static_cast<int>(lights.size());
    int num_density_samples = static_cast<int>(density_samples.size());
    std::cout << "Number of lights: " << num_lights << std::endl;
    std::cout << "Number of density samples: " << num_density_samples << std::endl;

    vec3 testPos = vec3{0, -49, 0}; // replace with known world-space center
    float d = getDensityAtPosition(vdb_file, testPos);
    std::cout << "Density at (0,0,0): " << d << std::endl;
    // Host image buffer
    std::vector<color> Image(total_pixels);


    // Allocate device memory
    color* d_image;
    light* d_lights;
    density_sample* d_density_samples;
    // VDBReader* d_vdb_reader;
    cudaMalloc(&d_image, total_pixels * sizeof(color));
    cudaMalloc(&d_lights, num_lights * sizeof(light));
    cudaMalloc(&d_density_samples, density_samples.size() * sizeof(density_sample));
    // cudaMalloc(&d_vdb_reader, sizeof(VDBReader));
    cudaMemcpy(d_lights, lights.data(), num_lights * sizeof(light), cudaMemcpyHostToDevice);
    cudaMemcpy(d_density_samples, density_samples.data(), density_samples.size() * sizeof(density_sample), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_vdb_reader, vdb_reader, sizeof(VDBReader), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (total_pixels + blockSize - 1) / blockSize;

    generateImage<<<numBlocks, blockSize>>>(d_image, width, height, d_lights, num_lights, min, max, vdb_file, d_density_samples, num_density_samples);
    cudaDeviceSynchronize();



    // Copy back and save
    cudaMemcpy(Image.data(), d_image, total_pixels * sizeof(color), cudaMemcpyDeviceToHost);
    saveImage(Image, width, height, output_file.c_str());
    std::cout << "Image saved to: " << output_file << std::endl;

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_lights);
    cudaFree(d_density_samples);
    // cudaFree(d_vdb_reader);
    return 0;
}