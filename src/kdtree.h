#ifndef KDTREE_H
#define KDTREE_H


#include "vec.h"
#include "light.h"

struct KDNode {
    vec3 position;
    vec3 color;
    float intensity;
    KDNode* left;
    KDNode* right;

    KDNode(const vec3 pos, const vec3 col, const float intensity) : position(pos), color(col), intensity(intensity) {}
};

// Device function to calculate squared distance between two points
float distanceSquared(const vec3& a, const vec3& b) {
    vec3 diff = a - b;
    return dot(diff, diff);
}

void radiusSearchRecursive(KDNode* node, const vec3& target, float radiusSquared, int depth, std::vector<KDNode*>& results) {
    if (!node) return;

    if (distanceSquared(target, node->position) <= radiusSquared) {
        results.push_back(node);
    }

    int axis = depth % 3;
    float diff = target[axis] - node->position[axis];

    if (diff <= 0) {
        radiusSearchRecursive(node->left, target, radiusSquared, depth + 1, results);
        if (diff * diff <= radiusSquared)
            radiusSearchRecursive(node->right, target, radiusSquared, depth + 1, results);
    } else {
        radiusSearchRecursive(node->right, target, radiusSquared, depth + 1, results);
        if (diff * diff <= radiusSquared)
            radiusSearchRecursive(node->left, target, radiusSquared, depth + 1, results);
    }
}

void radiusSearch(KDNode* root, const vec3& target, float radius, std::vector<KDNode*>& results) {
        radiusSearchRecursive(root, target, radius * radius, 0, results);
}




#endif // KDTREE_H

