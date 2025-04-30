# Volumetric Path-tracer illuminated by million lights in CUDA

writing a volumetric path-tracer in CUDA C++

cluster instruction
```bash
sbatch gpu_job.slurm
```

<img src="output/untitled.filecache1_v1.0015_lights.png" height='500px'></img>


```bash
cd nanovdb/nanovdb
rm -rf build
mkdir build
cd build

# Create the target directory first
mkdir -p /users/aagar133/.conda/envs/graphics/include/nanovdb

# Try with DESTDIR environment variable
DESTDIR=/users/aagar133/.conda/envs/graphics/include cmake .. \
    -DCMAKE_INSTALL_PREFIX="/nanovdb"

make -j$(nproc)
DESTDIR=/users/aagar133/.conda/envs/graphics/include make install
```


```
module load  gcc/10.1.0-mojgbnp 

nvcc -std=c++17 main.cu src/vdb_reader.cu -o render \
    -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib \
    -Xlinker -rpath -Xlinker $CONDA_PREFIX/lib \
    -lopenvdb -ltbb -lblosc -lboost_system -lboost_iostreams -lboost_filesystem

echo "Compilation complete!"

# Now compile CUDA code
# nvcc main.cu -o render
# nvprof ./render lights/untitled.filecache1_v1.0026_lights.bin

# for file in lights/*.bin; do
#     if [ -f "$file" ]; then
#         echo "Processing $file..."
#         ./render "$file"
#     fi
# done




export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
for file in /users/aagar133/scratch/cuda/v1/*.vdb; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        ./render "$file"
    fi
done

echo "All files processed!" 
```
