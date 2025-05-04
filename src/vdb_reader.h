#pragma once
#include <string>

void readAndPrintVDB(const char* filename);
std::vector<light> getLightsFromVDB(const char* filename);
void getScaledBoundingBox(const char* filename, vec3& outMin, vec3& outMax);
std::vector<float> getDenseGridFromVDB(const char* vdb_file, int& nx, int& ny, int& nz);