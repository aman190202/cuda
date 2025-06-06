#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <filesystem>
#include <thread>
#include <chrono>
#include <iomanip>

#include "src/vec.h"
#include "src/light.h"
#include "src/renderer.h"
#include "src/kdtree.h"
#include "src/LGH.h"
#include "src/vdb_reader.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "src/stb_image_write.h"

constexpr float PI = 3.14159265358979323846f;

// Simple Reinhard tone mapping
__device__ __host__ vec3 toneMap(const vec3& color)
{
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
__global__ void generateImage(color* image, 
                                int width, int height, 
                                light* lights,  int num_lights, 
                                vec3 min, vec3 max, vec3 center, float* d_density_grid, int nx, int ny, int nz, int* d_progress)//, KDNode** d_lighting_grids, int num_lighting_grids, float voxel_size) 
{
    // Use 2D grid layout
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= width || j >= height) return;
    
    int idx = j * width + i;

    // set up camera
    float aspect = float(width) / float(height);
    float fov = PI / 3.0f;           // 60° field of view
    float scale = tanf(fov * 0.5f);

    // Initialize random state for this thread
    curandState_t state;
    curand_init(clock64(), idx, 0, &state);

    // Take multiple samples per pixel
    const int num_samples = 1;
    vec3 accumulated_color = vec3{0, 0, 0};

    for (int s = 0; s < num_samples; s++) {
        // Add jitter to pixel coordinates
        float u = (2.0f * (i + 0.5f + curand_uniform(&state)) / float(width)  - 1.0f) * aspect * scale;
        float v = (1.0f - 2.0f * (j + 0.5f + curand_uniform(&state)) / float(height)) * scale;

        vec3 ray_origin = vec3{0.0f, 0.0f, 100.0f};
        vec3 ray_direction = normalize(vec3{u, v, -1.0f});
        
        vec3 sample_color = trace_ray(ray_origin, 
                                    ray_direction, 
                                    lights, num_lights,
                                    min, max, center, d_density_grid, nx, ny, nz);//, d_lighting_grids, num_lighting_grids, voxel_size);

        accumulated_color = accumulated_color + sample_color;
    }

    // Average the samples
    vec3 pixel_color = accumulated_color * (1.0f / num_samples);
    
    // Apply tone mapping
    pixel_color = toneMap(pixel_color);

    image[idx].r = pixel_color.x;
    image[idx].g = pixel_color.y;
    image[idx].b = pixel_color.z;

    // Update progress counter
    atomicAdd(d_progress, 1);
}

void saveImage(const std::vector<color>& image, 
                int width, int height, 
                const char* filename)
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

    // Input file shenanigans
    if (argc != 2) {    
        std::cerr << "Usage: " << argv[0] << " <vdb_file>" << std::endl;
        return 1;
    }

    const char* vdb_file = argv[1];
    int width  = 500;
    int height = 500;
    int total_pixels = width * height;



    // Light and density grid setup 
    float voxel_size;
    vec3 world_min, world_max;
    std::vector<light> lights = getLightsFromVDB(vdb_file, &voxel_size, &world_min, &world_max); //getTopNLightsFromVDB(vdb_file, &voxel_size, &world_min, &world_max, 100000);
    std::vector<light> lights_all = getLightsFromVDB(vdb_file, &voxel_size, &world_min, &world_max);
    std::vector<KDNode*> lighting_grids = LGH(lights_all, 1, voxel_size, world_min, world_max);
    int num_lighting_grids = lighting_grids.size();
    


    int num_lights = static_cast<int>(lights.size());
    std::cout << "Number of lights: " << num_lights << std::endl;
    std::cout << "Number of actual lights: " << lights_all.size() << std::endl;

    vec3 min, max;
    getScaledBoundingBox(vdb_file, min, max);
    vec3 center = (min + max) / 2;

    int nx, ny, nz;
    std::vector<float> denseGrid = getDenseGridFromVDB(vdb_file, nx, ny, nz);
    denseGrid = gaussianSmooth(denseGrid, nx, ny, nz, 3.0f);

    float* d_density_grid;
    size_t gridSize = nx * ny * nz * sizeof(float);

    // sanity check
    std::cout << "Grid dimensions: " << nx << " " << ny << " " << nz << std::endl;
    std::cout << "Grid size (bytes): " << gridSize << std::endl;

    
    // Allocate pinned memory for host arrays
    color* h_image;
    cudaHostAlloc(&h_image, total_pixels * sizeof(color), cudaHostAllocDefault);
    
    // Allocate device memory
    color* d_image;
    light* d_lights;
    int* d_progress;
    KDNode** d_lighting_grids;
    
    cudaMalloc(&d_image, total_pixels * sizeof(color));
    cudaMalloc(&d_lights, num_lights * sizeof(light));
    cudaMalloc(&d_density_grid, gridSize);
    cudaMalloc(&d_progress, sizeof(int));
    cudaMemset(d_progress, 0, sizeof(int));
    
    // Use asynchronous memory transfers
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemcpyAsync(d_lights, lights.data(), num_lights * sizeof(light), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_density_grid, denseGrid.data(), gridSize, cudaMemcpyHostToDevice, stream);
    cudaMalloc(&d_lighting_grids, lighting_grids.size() * sizeof(KDNode*));
    cudaMemcpyAsync(d_lighting_grids, lighting_grids.data(), lighting_grids.size() * sizeof(KDNode*), cudaMemcpyHostToDevice, stream);
    
    // Wait for transfers to complete
    cudaStreamSynchronize(stream);

    // Print memory usage information
    size_t total_memory = 0;
    size_t free_memory = 0;
    cudaMemGetInfo(&free_memory, &total_memory);
    
    std::cout << "\nMemory Usage Information:" << std::endl;
    std::cout << "------------------------" << std::endl;
    std::cout << "Total GPU Memory: " << total_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Free GPU Memory: " << free_memory / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Used GPU Memory: " << (total_memory - free_memory) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "\nAllocated Memory Breakdown:" << std::endl;
    std::cout << "Image Buffer: " << (total_pixels * sizeof(color)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Lights Buffer: " << (num_lights * sizeof(light)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Density Grid: " << gridSize / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Progress Counter: " << sizeof(int) / 1024.0 << " KB" << std::endl;
    std::cout << "------------------------" << std::endl;

    // Launch kernel with 2D grid
    dim3 blockSize(16, 16);  // 256 threads per block in 2D
    dim3 numBlocks((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    std::vector<color> Image(total_pixels);

    // Start progress tracking
    std::cout << "Rendering started..." << std::endl;
    generateImage<<<numBlocks, blockSize>>>(d_image, width, height, d_lights, num_lights, min, max, center, d_density_grid, nx, ny, nz, d_progress);//, d_lighting_grids, num_lighting_grids, voxel_size);
    
    // Monitor progress
    int last_progress = 0;
    while (true) {
        int current_progress;
        cudaMemcpy(&current_progress, d_progress, sizeof(int), cudaMemcpyDeviceToHost);
        float progress_percent = (float)current_progress / total_pixels * 100.0f;
        
        if (progress_percent > last_progress + 1.0f) {
            std::cout << "\rProgress: " << std::fixed << std::setprecision(1) << progress_percent << "%" << std::flush;
            last_progress = (int)progress_percent;
        }
        
        if (current_progress >= total_pixels) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << std::endl;

    cudaDeviceSynchronize();

    std::string input_file(vdb_file);
    size_t last_slash = input_file.find_last_of("/\\");
    std::string filename = (last_slash != std::string::npos) ? input_file.substr(last_slash + 1) : input_file;
    size_t last_dot = filename.find_last_of(".");
    std::string base_name = (last_dot != std::string::npos) ? filename.substr(0, last_dot) : filename;
    std::string output_file = "final_demo_new/" + base_name + ".png";

    // Copy back using pinned memory
    cudaMemcpyAsync(h_image, d_image, total_pixels * sizeof(color), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Save image using pinned memory
    saveImage(std::vector<color>(h_image, h_image + total_pixels), width, height, output_file.c_str());
    
    // Cleanup
    cudaFreeHost(h_image);
    cudaFree(d_image);
    cudaFree(d_lights);
    cudaFree(d_density_grid);
    cudaFree(d_lighting_grids);
    cudaStreamDestroy(stream);
    
    return 0;
}