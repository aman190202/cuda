#ifndef KDTREE_H
#define KDTREE_H

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "vec.h"
#include "light.h"

struct KDNode {
    vec3 position;
    vec3 color;
    float intensity;
    KDNode* left;
    KDNode* right;
};

// Functor for sorting lights by axis
struct LightAxisComparator {
    int axis;
    __device__ bool operator()(const light& a, const light& b) const {
        return a.position[axis] < b.position[axis];
    }
};

// Simple bubble sort for device code
__device__ void sort_lights_by_axis(light* lights, int num_lights, int axis) {
    for (int i = 0; i < num_lights - 1; i++) {
        for (int j = 0; j < num_lights - i - 1; j++) {
            if (lights[j].position[axis] > lights[j + 1].position[axis]) {
                // Swap lights
                light temp = lights[j];
                lights[j] = lights[j + 1];
                lights[j + 1] = temp;
            }
        }
    }
}

__host__ __device__ KDNode* build_kdtree(light* lights, int num_lights, int depth) 
{
    if (num_lights == 0) return nullptr;

    int axis = depth % 3;
    size_t median = num_lights / 2;

    // Sort based on position
    #ifdef __CUDA_ARCH__
    sort_lights_by_axis(lights, num_lights, axis);
    #else
    thrust::device_ptr<light> d_lights_ptr(lights);
    LightAxisComparator comp{axis};
    thrust::sort(d_lights_ptr, d_lights_ptr + num_lights, comp);
    #endif

    // Allocate new node
    KDNode* node;
    #ifdef __CUDA_ARCH__
    cudaMalloc(&node, sizeof(KDNode));
    #else
    node = new KDNode();
    #endif
    
    // Set node properties
    node->position = lights[median].position;
    node->color = lights[median].col;  // Set the color from the light
    node->intensity = lights[median].intensity;
    node->left = nullptr;
    node->right = nullptr;
    
    // Recursively build left and right subtrees
    node->left = build_kdtree(lights, median, depth + 1);
    node->right = build_kdtree(lights + median + 1, num_lights - median - 1, depth + 1);

    return node;
}

// Helper function to free the tree
__host__ __device__ void free_kdtree(KDNode* root) {
    if (root == nullptr) return;
    free_kdtree(root->left);
    free_kdtree(root->right);
    #ifdef __CUDA_ARCH__
    cudaFree(root);
    #else
    delete root;
    #endif
}

// Device function to calculate squared distance between two points
__device__ float distanceSquared(const vec3& a, const vec3& b) {
    vec3 diff = a - b;
    return dot(diff, diff);
}

// Device function for radius search
__device__ void radiusSearchRecursive(KDNode* node, const vec3& target, float radiusSquared,
                                    int depth, KDNode** results, int* resultCount, int maxResults) 
{
    if (!node || *resultCount >= maxResults) return;

    if (distanceSquared(target, node->position) <= radiusSquared) {
        results[(*resultCount)++] = node;
    }

    int axis = depth % 3;
    float diff = target[axis] - node->position[axis];

    if (diff <= 0) {
        radiusSearchRecursive(node->left, target, radiusSquared, depth + 1, results, resultCount, maxResults);
        if (diff * diff <= radiusSquared)
            radiusSearchRecursive(node->right, target, radiusSquared, depth + 1, results, resultCount, maxResults);
    } else {
        radiusSearchRecursive(node->right, target, radiusSquared, depth + 1, results, resultCount, maxResults);
        if (diff * diff <= radiusSquared)
            radiusSearchRecursive(node->left, target, radiusSquared, depth + 1, results, resultCount, maxResults);
    }
}

// CUDA kernel for radius search
__global__ void radiusSearchKernel(KDNode* root, const vec3 target, float radius,
                                 KDNode** results, int* resultCount, int maxResults) 
{
    radiusSearchRecursive(root, target, radius * radius, 0, results, resultCount, maxResults);
}

// Host wrapper for radius search
void radiusSearch(KDNode* d_root, const vec3& target, float radius, 
                 KDNode** d_results, int* d_resultCount, int maxResults) 
{
    // Reset result count
    cudaMemset(d_resultCount, 0, sizeof(int));

    // Launch kernel
    radiusSearchKernel<<<1, 1>>>(d_root, target, radius, d_results, d_resultCount, maxResults);
}

#endif // KDTREE_H

