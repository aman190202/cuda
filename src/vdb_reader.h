#pragma once
#include <string>

void readAndPrintVDB(const std::string& filename);
// float getDensityAtPosition(const std::string& filename, float x, float y, float z);
std::vector<light> getLightsFromVDB(const std::string& filename);
// __host__ __device__ vec3 temperatureToColor(float temp);
float getDensityAtPosition(const std::string& filename, const vec3& position);
