__device__ __forceinline__ float distanceSquaredCUDA(const vec3 a, const vec3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return dx*dx + dy*dy + dz*dz;
}

// device recursive search unchanged except use atomicAdd for count
__device__ void radiusSearchRecursive(
    KDNode* node, const vec3 target, const float r2,
    const int depth, KDNode** results, int* resultCount, const int maxResults)
{
    if (!node || *resultCount >= maxResults) return;
    if (distanceSquaredCUDA(target, node->position) <= r2) {
        int dst = atomicAdd(resultCount, 1);
        if (dst < maxResults) results[dst] = node;
    }
    int axis = depth % 3;
    float diff = ((&target.x)[axis]) - ((&node->position.x)[axis]);
    if (diff <= 0.0f) {
        radiusSearchRecursive(node->left,  target, r2, depth+1, results, resultCount, maxResults);
        if (diff*diff <= r2)
            radiusSearchRecursive(node->right, target, r2, depth+1, results, resultCount, maxResults);
    } else {
        radiusSearchRecursive(node->right, target, r2, depth+1, results, resultCount, maxResults);
        if (diff*diff <= r2)
            radiusSearchRecursive(node->left,  target, r2, depth+1, results, resultCount, maxResults);
    }
}


// host setup
/*
constexpr int MAX_RESULTS = 128;
KDNode** d_results;
int*     d_resultCount;
cudaMalloc(&d_results,    MAX_RESULTS * sizeof(KDNode*));
cudaMalloc(&d_resultCount, sizeof(int));
cudaMemset(d_resultCount, 0, sizeof(int));
radiusSearchKernel<<<1,1>>>(d_root, target, R, d_results, d_resultCount, MAX_RESULTS);
*/