#ifndef LIGHTING_GRID_HIERARCHY_H
#define LIGHTING_GRID_HIERARCHY_H

#include "vec.h"
#include "kdtree.h"
#include <algorithm>

KDNode* deriveNewS(KDNode* lighting_grid_j, int l, float h, vec3 BBoxMin, vec3 BBoxMax);
KDNode* build_kdtree(const std::vector<light>& lights);
std::vector<KDNode*> LGH(std::vector<light> lights, int depth, float h, vec3 BBoxMin, vec3 BBoxMax);
float calcNewI(int l, vec3 target_light_pos, const std::vector<KDNode*>& j_lights, float h);
vec3 calcNewPos(int l, vec3 target_light_pos, const std::vector<KDNode*>& j_lights, float h);
vec3 calcNewColor(int l, vec3 target_light_pos, const std::vector<KDNode*>& j_lights, float h);

KDNode* build(std::vector<light> points, int depth) {
    if (points.empty()) return nullptr;

    int axis = depth % 3;
    size_t median = points.size() / 2;

    // Sorts based on position
    std::nth_element(points.begin(), points.begin() + median, points.end(),
        [axis](const light& a, const light& b) {
            return a.position[axis] < b.position[axis];
        });

    KDNode* node = new KDNode(points[median].position, points[median].col, points[median].intensity);

    std::vector<light> leftPoints(points.begin(), points.begin() + median);
    std::vector<light> rightPoints(points.begin() + median + 1, points.end());

    node->left = build(leftPoints, depth + 1);
    node->right = build(rightPoints, depth + 1);

    return node;
}

KDNode* build_kdtree(const std::vector<light>& lights) {
    return build(lights, 0);
}

std::vector<KDNode*> LGH(std::vector<light> lights, int depth, float h, vec3 BBoxMin, vec3 BBoxMax)
{
    std::vector<KDNode*> lighting_grids = std::vector<KDNode*>(depth + 1);

    // Initialize S0
    lighting_grids[0] = build_kdtree(lights);

    // Create rest of S_l
    for (int i=1; i<=depth; i++) {
        lighting_grids[i] = deriveNewS(lighting_grids[i-1], i, h, BBoxMin, BBoxMax);
    }

    return lighting_grids;
}

KDNode* deriveNewS(KDNode* lighting_grid_j, int l, float h, vec3 BBoxMin, vec3 BBoxMax)
{
    float h_l = h * pow(2, l);                 // spacing of level l
    std::vector<light> lights;

    // iterate over grid vertices of level l
    for (float x = BBoxMin.x; x < BBoxMax.x; x += h_l)
    for (float y = BBoxMin.y; y < BBoxMax.y; y += h_l)
    for (float z = BBoxMin.z; z < BBoxMax.z; z += h_l)
    {
        vec3 target_light_pos(x,y,z);
    
        float radius = sqrt(3) * h * pow(2, l-1);    
        std::vector<KDNode*> j_lights;
        constexpr int MAX_RESULTS = 10000;
        
        radiusSearch(lighting_grid_j, target_light_pos, radius, j_lights);

        if (!j_lights.empty()) {
            float I = calcNewI(l, target_light_pos, j_lights, h);
            vec3 p = calcNewPos(l, target_light_pos, j_lights, h);
            vec3 col = calcNewColor(l, target_light_pos, j_lights, h);
            if (I > 0.f) {
                lights.emplace_back(p, col, I);
            }
        }
    }

    return build_kdtree(lights);
}

float calcTrilinearWeight(vec3 p, vec3 q, float h_l) {
    vec3 v = vec3(1,1,1) - (p - q).abs() / h_l;
    float v_product = v.x * v.y * v.z;
    return std::max(0.f, std::min(1.f, v_product));
}

float calcNewI(int l, vec3 target_light_pos, const std::vector<KDNode*>& j_lights, float h)
{
    float I_sum = 0.0f;
    for (const auto& node : j_lights) {
        float w = calcTrilinearWeight(node->position, target_light_pos, h * pow(2, l-1));
        I_sum += w * node->intensity;
    }
    return I_sum;
}

vec3 calcNewPos(int l, vec3 target_light_pos, const std::vector<KDNode*>& j_lights, float h)
{
    vec3 p_num(0.0f, 0.0f, 0.0f);
    float p_denom = 0.0f;

    for (const auto& node : j_lights) {
        float w = calcTrilinearWeight(node->position, target_light_pos, h * pow(2, l-1));
        float v = w * node->intensity;
        p_num += v * node->position;
        p_denom += v;
    }

    return (p_denom > 0.0f) ? p_num / p_denom : target_light_pos;
}

vec3 calcNewColor(int l, vec3 target_light_pos, const std::vector<KDNode*>& j_lights, float h)
{
    vec3 color_num(0.0f, 0.0f, 0.0f);
    float color_denom = 0.0f;

    for (const auto& node : j_lights) {
        float w = calcTrilinearWeight(node->position, target_light_pos, h * pow(2, l-1));
        float v = w * node->intensity;
        color_num += v * node->color;
        color_denom += v;
    }

    return (color_denom > 0.0f) ? color_num / color_denom : j_lights[0]->color;
}

#endif // LIGHTING_GRID_HIERARCHY_H