#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include "vec.h"

class VDBReader {
private:
    openvdb::FloatGrid::Ptr densityGrid;
    openvdb::CoordBBox bbox;
    openvdb::Vec3d gridMin, gridMax;
    float voxelSize;

public:
    VDBReader() : densityGrid(nullptr), voxelSize(1.0f) {}

    bool loadFile(const std::string& filename) {
        // Initialize OpenVDB
        openvdb::initialize();

        // Open the file
        openvdb::io::File file(filename);
        if (!file.open()) {
            std::cerr << "Failed to open VDB file: " << filename << std::endl;
            return false;
        }

        // Read the density grid
        for (openvdb::io::File::NameIterator nameIter = file.beginName();
             nameIter != file.endName(); ++nameIter) {
            if (nameIter.gridName() == "density") {
                densityGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid(nameIter.gridName()));
                break;
            }
        }

        if (!densityGrid) {
            std::cerr << "No density grid found in VDB file" << std::endl;
            return false;
        }

        // Get grid information
        bbox = densityGrid->evalActiveVoxelBoundingBox();
        gridMin = densityGrid->indexToWorld(bbox.min());
        gridMax = densityGrid->indexToWorld(bbox.max());
        voxelSize = densityGrid->voxelSize()[0];

        file.close();
        return true;
    }

    __device__ float sampleDensity(const vec3& worldPos) const {
        if (!densityGrid) return 0.0f;

        // Convert world position to grid coordinates
        openvdb::Vec3d gridPos = densityGrid->worldToIndex(openvdb::Vec3d(worldPos.x, worldPos.y, worldPos.z));
        
        // Use trilinear interpolation to sample the density
        return openvdb::tools::BoxSampler::sample(densityGrid->tree(), gridPos);
    }

    __device__ bool isInsideVolume(const vec3& worldPos) const {
        if (!densityGrid) return false;
        
        return worldPos.x >= gridMin.x() && worldPos.x <= gridMax.x() &&
               worldPos.y >= gridMin.y() && worldPos.y <= gridMax.y() &&
               worldPos.z >= gridMin.z() && worldPos.z <= gridMax.z();
    }

    __device__ vec3 getMin() const {
        return vec3{static_cast<float>(gridMin.x()),
                   static_cast<float>(gridMin.y()),
                   static_cast<float>(gridMin.z())};
    }

    __device__ vec3 getMax() const {
        return vec3{static_cast<float>(gridMax.x()),
                   static_cast<float>(gridMax.y()),
                   static_cast<float>(gridMax.z())};
    }
}; 