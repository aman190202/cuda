#include <openvdb/openvdb.h>
#include <iostream>

int main() {
    openvdb::initialize();

    const std::string filename = "/users/aagar133/scratch/cuda/ground_explosion_0015.vdb";  // <-- change this
    openvdb::io::File file(filename);

    try {
        file.open();
        for (auto nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
            std::cout << "Found grid: " << nameIter.gridName() << std::endl;

            openvdb::GridBase::Ptr baseGrid = file.readGrid(nameIter.gridName());
            std::cout << "Grid bbox: " << baseGrid->evalActiveVoxelBoundingBox() << std::endl;
        }
        file.close();
    } catch (openvdb::IoError& e) {
        std::cerr << "Failed to read VDB: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}