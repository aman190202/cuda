#pragma once
#include <string>

void readAndPrintVDB(const char* filename);
std::vector<light> getLightsFromVDB(const char* filename, float* voxel_size, vec3* world_min, vec3* world_max);
void getScaledBoundingBox(const char* filename, vec3& outMin, vec3& outMax);
std::vector<float> getDenseGridFromVDB(const char* vdb_file, int& nx, int& ny, int& nz);
std::vector<float> gaussianSmooth(const std::vector<float>& grid, int nx, int ny, int nz, float sigma);
std::vector<light> getTopNLightsFromVDB(
    const char* filename,
    float* voxel_size,
    vec3* world_min,
    vec3* world_max,
    size_t maxLights);