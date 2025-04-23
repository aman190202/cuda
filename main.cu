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
// #include "vdb_to_lights.cu" 
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

    vec3 ray_origin = vec3{0.0f, 0.0f, 100.0f};
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


int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <lights_file>" << std::endl;
        return 1;
    }

    const char* lights_file = argv[1];
    int width  = 1000;
    int height = 1000;
    int total_pixels = width * height;

    // Extract base filename without extension
    std::string input_file(lights_file);
    size_t last_slash = input_file.find_last_of("/\\");
    std::string filename = (last_slash != std::string::npos) ? input_file.substr(last_slash + 1) : input_file;
    size_t last_dot = filename.find_last_of(".");
    std::string base_name = (last_dot != std::string::npos) ? filename.substr(0, last_dot) : filename;
    
    // Ensure output directory exists
    std::filesystem::create_directories("output");
    std::string output_file = "output/" + base_name + ".png";

    std::vector<light> lights;

    // Load from binary file
    std::ifstream lightFile(lights_file, std::ios::binary);
    if (!lightFile) {
        std::cerr << "Failed to open " << lights_file << std::endl;
        return 1;
    }

    // Determine file size
    lightFile.seekg(0, std::ios::end);
    std::streamsize size = lightFile.tellg();
    lightFile.seekg(0, std::ios::beg);

    if (size <= 0 || size % sizeof(light) != 0) {
        std::cerr << "Invalid file size or format" << std::endl;
        return 1;
    }

    // Resize and read into vector
    lights.resize(size / sizeof(light));
    if (!lightFile.read(reinterpret_cast<char*>(lights.data()), size)) {
        std::cerr << "Error reading from light.bin" << std::endl;
        return 1;
    }

    int num_lights = static_cast<int>(lights.size());
    std::cout << "Number of lights: " << num_lights << std::endl;

    // Get volume bounds if needed

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
    saveImage(Image, width, height, output_file.c_str());
    std::cout << "Image saved to: " << output_file << std::endl;

    // Cleanup
    cudaFree(d_image);
    cudaFree(d_lights);
    return 0;
}