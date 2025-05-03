#include <openvdb/openvdb.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "light.h"

struct DensityDebugInfo {
    vec3 inputPos;
    vec3 worldPos;
    vec3 gridCoord;
    float density;
    bool foundDensity;
};

__device__ DensityDebugInfo* debugBuffer = nullptr;
__device__ int debugBufferIndex = 0;

DensityDebugInfo* hostDebugBuffer = nullptr;
int debugBufferSize = 0;

void allocateDebugBuffer(int size) {
    debugBufferSize = size;
    cudaMalloc(&hostDebugBuffer, size * sizeof(DensityDebugInfo));
    cudaMemcpyToSymbol(debugBuffer, &hostDebugBuffer, sizeof(DensityDebugInfo*));
    cudaMemcpyToSymbol(debugBufferIndex, &debugBufferSize, sizeof(int));
}

void freeDebugBuffer() {
    if (hostDebugBuffer) {
        cudaFree(hostDebugBuffer);
        hostDebugBuffer = nullptr;
    }
}

void printDebugInfo() {
    if (!hostDebugBuffer) return;
    
    DensityDebugInfo* localBuffer = new DensityDebugInfo[debugBufferSize];
    cudaMemcpy(localBuffer, hostDebugBuffer, debugBufferSize * sizeof(DensityDebugInfo), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < debugBufferSize; i++) {
        if (localBuffer[i].foundDensity) {
            std::cout << "Debug Info " << i << ":" << std::endl;
            std::cout << "  Input Pos: (" << localBuffer[i].inputPos.x << ", " 
                      << localBuffer[i].inputPos.y << ", " 
                      << localBuffer[i].inputPos.z << ")" << std::endl;
            std::cout << "  World Pos: (" << localBuffer[i].worldPos.x << ", " 
                      << localBuffer[i].worldPos.y << ", " 
                      << localBuffer[i].worldPos.z << ")" << std::endl;
            std::cout << "  Grid Coord: (" << localBuffer[i].gridCoord.x << ", " 
                      << localBuffer[i].gridCoord.y << ", " 
                      << localBuffer[i].gridCoord.z << ")" << std::endl;
            std::cout << "  Density: " << localBuffer[i].density << std::endl;
            std::cout << std::endl;
        }
    }
    
    delete[] localBuffer;
}

void readAndPrintVDB(const char* filename) {
    openvdb::initialize();

    openvdb::io::File file(filename);



    try {
        file.open();
        for (auto nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
            std::cout << "Found grid: " << nameIter.gridName() << std::endl;

            openvdb::GridBase::Ptr baseGrid = file.readGrid(nameIter.gridName());
            std::cout << "Grid bbox: " << baseGrid->evalActiveVoxelBoundingBox() << std::endl;
        }
        file.close();
    } catch (const openvdb::IoError& e) {
        std::cerr << "Failed to read VDB: " << e.what() << std::endl;
    }
}



// Function to map temperature to RGB color based on Kelvin scale
__host__ __device__ vec3 temperatureToColor(float temp) {
    // Scale to Kelvin (e.g., 1000K to 7000K)
    temp = temp * 4500.0f;
    float kelvin = std::clamp(temp, 1000.0f, 7000.0f);
    
    // Normalize for interpolation (1000K to 7000K)
    float t = (kelvin - 1000.0f) / (7000.0f - 1000.0f);
    
    // Smooth interpolation for black-body color ramp
    float r = 1.0f;
    float g = std::clamp(0.0f + t * 1.2f, 0.0f, 1.0f);  // from dark red to yellow
    float b = std::clamp((t - 0.5f) * 2.0f, 0.0f, 1.0f); // blue kicks in after midpoint
    
    // Optional: Apply simple gamma correction
    float gamma = 2.2f;
    r = pow(r, 1.0f / gamma);
    g = pow(g, 1.0f / gamma);
    b = pow(b, 1.0f / gamma);
    
    return vec3(r, g, b);
}


std::vector<light> getLightsFromVDB(const char* filename) {
    std::vector<light> lights;
    openvdb::initialize();

    openvdb::io::File file(filename);
    try {
        file.open();
        
        // Find and process only the flames grid
        openvdb::GridBase::Ptr baseGrid = file.readGrid("temperature");
        openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        
        if (floatGrid) {
            float minValue = std::numeric_limits<float>::max();
            float maxValue = std::numeric_limits<float>::lowest();
            
            // Get the bounding box in world coordinates
            openvdb::CoordBBox bbox = floatGrid->evalActiveVoxelBoundingBox();
            openvdb::Vec3d worldMin = floatGrid->indexToWorld(bbox.min());
            openvdb::Vec3d worldMax = floatGrid->indexToWorld(bbox.max());
            
            std::cout << "Original bounding box (world coordinates):" << std::endl;
            std::cout << "  Min: (" << worldMin.x() << ", " << worldMin.y() << ", " << worldMin.z() << ")" << std::endl;
            std::cout << "  Max: (" << worldMax.x() << ", " << worldMax.y() << ", " << worldMax.z() << ")" << std::endl;
            
            // Calculate the height of the bounding box
            double height = worldMax.y() - worldMin.y();
            
            // Target base position
            openvdb::Vec3d targetBase(0.0, -50.0, 0.0);
            
            // Scale factor for the bounding box
            const float scaleFactor = 2.0f;
            
            // Calculate the new height after scaling
            double scaledHeight = height * scaleFactor;
            
            // Iterate through all active voxels
            for (openvdb::FloatGrid::ValueOnIter iter = floatGrid->beginValueOn(); iter; ++iter) {
                float value = iter.getValue();
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
                

                if((value * 4500) < 1000) continue;
                // Get world position of the voxel
                openvdb::Vec3d worldPos = floatGrid->indexToWorld(iter.getCoord());
                
                // Calculate relative position within the bounding box (0 to 1)
                double relY = (worldPos.y() - worldMin.y()) / height;
                
                // Calculate new position
                // X and Z are scaled and centered around targetBase
                // Y is positioned relative to the new base
                openvdb::Vec3d scaledPos(
                    targetBase.x() + (worldPos.x() - (worldMin.x() + worldMax.x()) * 0.5) * scaleFactor,
                    targetBase.y() + scaledHeight * relY,
                    targetBase.z() + (worldPos.z() - (worldMin.z() + worldMax.z()) * 0.5) * scaleFactor
                );
                
                // Create a new light
                light l;
                l.position = vec3(scaledPos.x(), scaledPos.y(), scaledPos.z());
                l.col = temperatureToColor(value);

                l.intensity = value/5; // Default intensity

                if(l.intensity < 0.05) continue;
                
                lights.push_back(l);
            }
            
            std::cout << "Flames grid value range: [" << minValue << ", " << maxValue << "]" << std::endl;
            std::cout << "Volume scaled by factor: " << scaleFactor << std::endl;
            std::cout << "New base position: (" << targetBase.x() << ", " << targetBase.y() << ", " << targetBase.z() << ")" << std::endl;
        } else {
            std::cerr << "Could not find or cast flames grid" << std::endl;
        }
        
        file.close();
    } catch (const openvdb::IoError& e) {
        std::cerr << "Failed to read VDB: " << e.what() << std::endl;
    }
    
    return lights;
}

float getDensityAtPosition(const char* filename, const vec3& position) {
    openvdb::initialize();
    openvdb::io::File file(filename);
    file.open();
    auto baseGrid  = file.readGrid("density");
    auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    // original VDB world bounds and center
    auto bbox     = floatGrid->evalActiveVoxelBoundingBox();
    auto worldMin = floatGrid->indexToWorld(bbox.min());
    auto worldMax = floatGrid->indexToWorld(bbox.max());
    auto center   = (worldMin + worldMax) * 0.5;

    // scene→VDB inverse transform
    const double scale = 2.0;
    const openvdb::Vec3d targetBase(0.0, -50.0, 0.0);
    openvdb::Vec3d worldPos(
        (position.x - targetBase.x())/scale + center.x(),
        (position.y - targetBase.y())/scale + center.y(),
        (position.z - targetBase.z())/scale + center.z()
    );

    // clamp inside VDB
    worldPos = openvdb::Vec3d(
        std::clamp(worldPos.x(), worldMin.x(), worldMax.x()),
        std::clamp(worldPos.y(), worldMin.y(), worldMax.y()),
        std::clamp(worldPos.z(), worldMin.z(), worldMax.z())
    );

    // trilinear gather
    auto idxPos   = floatGrid->worldToIndex(worldPos);
    auto ijk0     = openvdb::Coord::floor(idxPos);
    auto accessor = floatGrid->getAccessor();
    float sum = 0.0f;
    for (int dx = 0; dx <= 1; ++dx)
     for (int dy = 0; dy <= 1; ++dy)
      for (int dz = 0; dz <= 1; ++dz)
        sum += accessor.getValue(ijk0.offsetBy(dx,dy,dz));

    file.close();
    return sum;
}


// Function to get the scaled and transformed bounding box coordinates
void getScaledBoundingBox(const char* filename, vec3& outMin, vec3& outMax) {
    openvdb::initialize();
    outMin = vec3(0.0f);
    outMax = vec3(0.0f);

    openvdb::io::File file(filename);
    try {
        file.open();
        
        // Find and process the density grid
        openvdb::GridBase::Ptr baseGrid = file.readGrid("density");
        openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        
        if (floatGrid) {
            // Get the grid's bounding box
            openvdb::CoordBBox bbox = floatGrid->evalActiveVoxelBoundingBox();
            openvdb::Vec3d worldMin = floatGrid->indexToWorld(bbox.min());
            openvdb::Vec3d worldMax = floatGrid->indexToWorld(bbox.max());
            
            // Calculate the height of the bounding box
            double height = worldMax.y() - worldMin.y();
            
            // Target base position
            openvdb::Vec3d targetBase(0.0, -50.0, 0.0);
            
            // Scale factor for the bounding box
            const float scaleFactor = 2.0f;
            
            // Calculate the new height after scaling
            double scaledHeight = height * scaleFactor;
            
            // Calculate scaled min and max positions
            openvdb::Vec3d scaledMin(
                targetBase.x() + (worldMin.x() - (worldMin.x() + worldMax.x()) * 0.5) * scaleFactor,
                targetBase.y(),
                targetBase.z() + (worldMin.z() - (worldMin.z() + worldMax.z()) * 0.5) * scaleFactor
            );
            
            openvdb::Vec3d scaledMax(
                targetBase.x() + (worldMax.x() - (worldMin.x() + worldMax.x()) * 0.5) * scaleFactor,
                targetBase.y() + scaledHeight,
                targetBase.z() + (worldMax.z() - (worldMin.z() + worldMax.z()) * 0.5) * scaleFactor
            );
            
            // Convert to vec3
            outMin = vec3(scaledMin.x(), scaledMin.y(), scaledMin.z());
            outMax = vec3(scaledMax.x(), scaledMax.y(), scaledMax.z());
        }
        
        file.close();
    } catch (const openvdb::IoError& e) {
        std::cerr << "Failed to read VDB: " << e.what() << std::endl;
    }
}

void getDensityRange(const char* filename, float& outMin, float& outMax, vec3& outMinPos, vec3& outMaxPos) {
    openvdb::initialize();
    outMin = std::numeric_limits<float>::max();
    outMax = std::numeric_limits<float>::lowest();
    outMinPos = vec3(0.0f);
    outMaxPos = vec3(0.0f);

    openvdb::io::File file(filename);
    try {
        file.open();
        
        // Find and process the density grid
        openvdb::GridBase::Ptr baseGrid = file.readGrid("density");
        openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        
        if (floatGrid) {
            // Get the grid's bounding box
            openvdb::CoordBBox bbox = floatGrid->evalActiveVoxelBoundingBox();
            openvdb::Vec3d worldMin = floatGrid->indexToWorld(bbox.min());
            openvdb::Vec3d worldMax = floatGrid->indexToWorld(bbox.max());
            
            // Calculate the height of the bounding box
            double height = worldMax.y() - worldMin.y();
            
            // Target base position
            openvdb::Vec3d targetBase(0.0, -50.0, 0.0);
            
            // Scale factor for the bounding box
            const float scaleFactor = 2.0f;
            
            // Calculate the new height after scaling
            double scaledHeight = height * scaleFactor;
            
            // Iterate through all active voxels to find min and max values
            for (openvdb::FloatGrid::ValueOnIter iter = floatGrid->beginValueOn(); iter; ++iter) {
                float value = iter.getValue();
                
                if (value < outMin) {
                    outMin = value;
                    // Get world position of the voxel
                    openvdb::Vec3d worldPos = floatGrid->indexToWorld(iter.getCoord());
                    
                    // Calculate relative position within the bounding box (0 to 1)
                    double relY = (worldPos.y() - worldMin.y()) / height;
                    
                    // Calculate new position
                    openvdb::Vec3d scaledPos(
                        targetBase.x() + (worldPos.x() - (worldMin.x() + worldMax.x()) * 0.5) * scaleFactor,
                        targetBase.y() + scaledHeight * relY,
                        targetBase.z() + (worldPos.z() - (worldMin.z() + worldMax.z()) * 0.5) * scaleFactor
                    );
                    
                    outMinPos = vec3(scaledPos.x(), scaledPos.y(), scaledPos.z());
                }
                
                if (value > outMax) {
                    outMax = value;
                    // Get world position of the voxel
                    openvdb::Vec3d worldPos = floatGrid->indexToWorld(iter.getCoord());
                    
                    // Calculate relative position within the bounding box (0 to 1)
                    double relY = (worldPos.y() - worldMin.y()) / height;
                    
                    // Calculate new position
                    openvdb::Vec3d scaledPos(
                        targetBase.x() + (worldPos.x() - (worldMin.x() + worldMax.x()) * 0.5) * scaleFactor,
                        targetBase.y() + scaledHeight * relY,
                        targetBase.z() + (worldPos.z() - (worldMin.z() + worldMax.z()) * 0.5) * scaleFactor
                    );
                    
                    outMaxPos = vec3(scaledPos.x(), scaledPos.y(), scaledPos.z());
                }
            }
        }
        
        file.close();
    } catch (const openvdb::IoError& e) {
        std::cerr << "Failed to read VDB: " << e.what() << std::endl;
    }
}

std::vector<density_sample> getDensitySamplesFromVDB(const char* filename) {
    std::vector<density_sample> samples;
    openvdb::initialize();

    openvdb::io::File file(filename);
    try {
        file.open();
        
        // Find and process the density grid
        openvdb::GridBase::Ptr baseGrid = file.readGrid("density");
        openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        
        if (floatGrid) {
            float minValue = std::numeric_limits<float>::max();
            float maxValue = std::numeric_limits<float>::lowest();
            
            // Get the bounding box in world coordinates
            openvdb::CoordBBox bbox = floatGrid->evalActiveVoxelBoundingBox();
            openvdb::Vec3d worldMin = floatGrid->indexToWorld(bbox.min());
            openvdb::Vec3d worldMax = floatGrid->indexToWorld(bbox.max());
            
            std::cout << "Original bounding box (world coordinates):" << std::endl;
            std::cout << "  Min: (" << worldMin.x() << ", " << worldMin.y() << ", " << worldMin.z() << ")" << std::endl;
            std::cout << "  Max: (" << worldMax.x() << ", " << worldMax.y() << ", " << worldMax.z() << ")" << std::endl;
            
            // Calculate the height of the bounding box
            double height = worldMax.y() - worldMin.y();
            
            // Target base position
            openvdb::Vec3d targetBase(0.0, -50.0, 0.0);
            
            // Scale factor for the bounding box
            const float scaleFactor = 2.0f;
            
            // Calculate the new height after scaling
            double scaledHeight = height * scaleFactor;
            
            // Iterate through all active voxels
            for (openvdb::FloatGrid::ValueOnIter iter = floatGrid->beginValueOn(); iter; ++iter) {
                float value = iter.getValue();
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
                if(value < 0.2) continue;
                
                // Get world position of the voxel
                openvdb::Vec3d worldPos = floatGrid->indexToWorld(iter.getCoord());
                
                // Calculate relative position within the bounding box (0 to 1)
                double relY = (worldPos.y() - worldMin.y()) / height;
                
                // Calculate new position
                // X and Z are scaled and centered around targetBase
                // Y is positioned relative to the new base
                openvdb::Vec3d scaledPos(
                    targetBase.x() + (worldPos.x() - (worldMin.x() + worldMax.x()) * 0.5) * scaleFactor,
                    targetBase.y() + scaledHeight * relY,
                    targetBase.z() + (worldPos.z() - (worldMin.z() + worldMax.z()) * 0.5) * scaleFactor
                );
                
                // Create a new density sample
                density_sample sample;
                sample.position = vec3(scaledPos.x(), scaledPos.y(), scaledPos.z());
                sample.value = value;
                
                samples.push_back(sample);
            }
            
            std::cout << "Density grid value range: [" << minValue << ", " << maxValue << "]" << std::endl;
            std::cout << "Volume scaled by factor: " << scaleFactor << std::endl;
            std::cout << "New base position: (" << targetBase.x() << ", " << targetBase.y() << ", " << targetBase.z() << ")" << std::endl;
        } else {
            std::cerr << "Could not find or cast density grid" << std::endl;
        }
        
        file.close();
    } catch (const openvdb::IoError& e) {
        std::cerr << "Failed to read VDB: " << e.what() << std::endl;
    }
    
    return samples;
}