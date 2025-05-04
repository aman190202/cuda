#include <openvdb/openvdb.h>
#include <openvdb/tools/Filter.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include "light.h"




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
            
            // std::cout << "Flames grid value range: [" << minValue << ", " << maxValue << "]" << std::endl;
            // std::cout << "Volume scaled by factor: " << scaleFactor << std::endl;
            // std::cout << "New base position: (" << targetBase.x() << ", " << targetBase.y() << ", " << targetBase.z() << ")" << std::endl;
        } else {
            std::cerr << "Could not find or cast flames grid" << std::endl;
        }
        
        file.close();
    } catch (const openvdb::IoError& e) {
        std::cerr << "Failed to read VDB: " << e.what() << std::endl;
    }
    
    return lights;
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
/*
std::vector<float> getDenseGridFromVDB(const char* vdb_file, int& nx, int& ny, int& nz) 
{
    openvdb::initialize();
    openvdb::io::File file(vdb_file);
    file.open();
    auto baseGrid = file.readGrid("density");
    auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    // Null grid check
    if (!floatGrid) {
        std::cerr << "ERROR: 'density' grid not found in VDB file. Aborting." << std::endl;
        exit(1);
    }

    auto bbox = floatGrid->evalActiveVoxelBoundingBox();

    // Empty bbox check
    if (bbox.empty()) {
        std::cerr << "ERROR: 'density' grid has empty bounding box (no active voxels). Aborting." << std::endl;
        exit(1);
    }

    auto dim = bbox.dim();
    nx = dim.x();
    ny = dim.y();
    nz = dim.z();

    // Sanity check
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        std::cerr << "ERROR: Invalid grid dimensions: " << nx << " " << ny << " " << nz << std::endl;
        exit(1);
    }

    std::vector<float> denseGrid(nx * ny * nz, 0.0f);
    auto accessor = floatGrid->getAccessor();
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                auto coord = bbox.min().offsetBy(x, y, z);
                denseGrid[x + y * nx + z * nx * ny] = accessor.getValue(coord);
            }
        }
    }

    file.close();
    return denseGrid;
}
*/

std::vector<float> getDenseGridFromVDB(const char* vdb_file, int& nx, int& ny, int& nz) 
{
    openvdb::initialize();
    openvdb::io::File file(vdb_file);
    file.open();
    auto baseGrid = file.readGrid("density");
    auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    if (!floatGrid) {
        std::cerr << "ERROR: 'density' grid not found in VDB file." << std::endl;
        exit(1);
    }

    auto bbox = floatGrid->evalActiveVoxelBoundingBox();
    if (bbox.empty()) {
        std::cerr << "ERROR: Bounding box is empty." << std::endl;
        exit(1);
    }

    auto dim = bbox.dim();
    nx = dim.x();
    ny = dim.y();
    nz = dim.z();

    std::vector<float> denseGrid(nx * ny * nz, 0.0f);

    // Iterate only active voxels
    for (auto iter = floatGrid->cbeginValueOn(); iter.test(); ++iter) {
        openvdb::Coord coord = iter.getCoord();
        openvdb::Coord local = coord - bbox.min();

        // Bounds check
        if (local.x() >= 0 && local.x() < nx &&
            local.y() >= 0 && local.y() < ny &&
            local.z() >= 0 && local.z() < nz) 
        {
            denseGrid[local.x() + local.y() * nx + local.z() * nx * ny] = iter.getValue();
        }
    }

    file.close();
    return denseGrid;
}


// std::vector<float> getDenseGridFromVDB(const char* vdb_file, int& nx, int& ny, int& nz) 
// {
//     openvdb::initialize();
//     openvdb::io::File file(vdb_file);
//     file.open();

//     auto baseGrid = file.readGrid("density");
//     auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

//     // Null grid check
//     if (!floatGrid) {
//         std::cerr << "ERROR: 'density' grid not found in VDB file. Aborting." << std::endl;
//         exit(1);
//     }

//     // --------->>> ADD HERE <<<<<--------
//     // Apply Gaussian blur to smooth density grid (important for mushroom shape)
//     openvdb::tools::GaussianFilter<openvdb::FloatGrid> filter(*floatGrid);
//     filter.setSigma(1.0);  // Sigma controls smoothness
//     filter.apply();
//     // -----------------------------------

//     auto bbox = floatGrid->evalActiveVoxelBoundingBox();
//     auto dim = bbox.dim();
//     nx = dim.x();
//     ny = dim.y();
//     nz = dim.z();

//     std::vector<float> denseGrid(nx * ny * nz, 0.0f);
//     auto accessor = floatGrid->getAccessor();
//     for (int z = 0; z < nz; ++z) {
//         for (int y = 0; y < ny; ++y) {
//             for (int x = 0; x < nx; ++x) {
//                 auto coord = bbox.min().offsetBy(x, y, z);
//                 denseGrid[x + y * nx + z * nx * ny] = accessor.getValue(coord);
//             }
//         }
//     }

//     file.close();
//     return denseGrid;
// }