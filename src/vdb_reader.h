#pragma once
#include <string>

void readAndPrintVDB(const char* filename);
std::vector<light> getLightsFromVDB(const char* filename);
float getDensityAtPosition(const char* filename, const vec3& position);
std::vector<density_sample> getDensitySamplesFromVDB(const char* filename);
void getScaledBoundingBox(const char* filename, vec3& outMin, vec3& outMax);
void getDensityRange(const char* filename, float& outMin, float& outMax, vec3& outMinPos, vec3& outMaxPos);
