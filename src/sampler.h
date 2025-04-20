#ifndef SAMPLER_H
#define SAMPLER_H

#include <string>
#include <memory>
#include <optional>
#include <stdexcept>


#include <openvdb/openvdb.h>
#include "vec.h"

class VDBVolumeSampler 
{
    private:

    openvdb::FloatGrid::Ptr densityGrid;
    openvdb::FloatGrid::Ptr temperatureGrid;
    std::optional<openvdb::FloatGrid::ConstAccessor> densityAccessor;
    std::optional<openvdb::FloatGrid::ConstAccessor> temperatureAccessor;
    openvdb::math::Transform::Ptr transform;

    bool initialized;

public:
    VDBVolumeSampler(const std::string& vdbFilePath) 
    {

        // Initialize OpenVDB
        openvdb::initialize();
        
        // Open the VDB file
        openvdb::io::File file(vdbFilePath);
        file.open();
        
        // Get the grids
        openvdb::GridPtrVecPtr grids = file.getGrids();
        
        // Find density and temperature grids
        for (openvdb::GridBase::Ptr grid : *grids) {
            if (grid->getName() == "density") {
                densityGrid = openvdb::GridBase::grid<openvdb::FloatGrid>(grid);
                densityAccessor.emplace(densityGrid->getConstAccessor());
            }
            else if (grid->getName() == "temperature") {
                temperatureGrid = openvdb::GridBase::grid<openvdb::FloatGrid>(grid);
                temperatureAccessor.emplace(temperatureGrid->getConstAccessor());
            }
        }
        
        if (!densityGrid || !temperatureGrid) {
            throw std::runtime_error("Could not find density or temperature grid in VDB file");
        }
        
        transform = densityGrid->transformPtr();
        initialized = true;
    }

    // Sample density at world position
    float sampleDensity(const vec3& worldPos) const
    {

        if (!densityAccessor) {
            throw std::runtime_error("Density accessor not initialized");
        }
        openvdb::Vec3d pos(worldPos.x(), worldPos.y(), worldPos.z());
        openvdb::Coord coord = transform->worldToIndexCellCentered(pos);
        return densityAccessor->getValue(coord);
    }

    // Sample temperature at world position
    float sampleTemperature(const vec3& worldPos) const 
    {
        if (!temperatureAccessor) {
            throw std::runtime_error("Temperature accessor not initialized");
        }
        openvdb::Vec3d pos(worldPos.x(), worldPos.y(), worldPos.z());
        openvdb::Coord coord = transform->worldToIndexCellCentered(pos);
        return temperatureAccessor->getValue(coord);
    }

    bool isInitialized() const { return initialized; }
};


// Initialize the global sampler
std::unique_ptr<VDBVolumeSampler> volumeSampler;

void initializeSampler(const std::string& vdbFilePath) {
    try {
        volumeSampler = std::make_unique<VDBVolumeSampler>(vdbFilePath);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize sampler: " + std::string(e.what()));
    }
}

vec3 densitySampler(const vec3& position) 
{
    if (!volumeSampler || !volumeSampler->isInitialized()) {
        return vec3(0.0f, 0.0f, 0.0f);
    }
    float density = volumeSampler->sampleDensity(position);
    // Return density as a grayscale color
    return vec3(density, density, density);
}

vec3 colorSampler(const vec3& position) 
{
    if (!volumeSampler || !volumeSampler->isInitialized()) {
        return vec3(0.0f, 0.0f, 0.0f);
    }
    float temperature = volumeSampler->sampleTemperature(position);
    
    // Map temperature to color using a simple color ramp
    // Cold (blue) -> Hot (red)
    float t = std::min(std::max(temperature, 0.0f), 1.0f);
    
    // Blue to red color ramp
    vec3 color;
    color.x = t;                    // Red increases with temperature
    color.y = 0.0;                  // No green
    color.z = 1.0 - t;              // Blue decreases with temperature
    
    return color;
} 