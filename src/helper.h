#include "light.h"
#include "kdtree.h"
#include <algorithm>  // For std::sort


void sort_lights_by_position(light* lights, int num_lights, const vec3& reference_point) {
    std::sort(lights, lights + num_lights, 
        [&reference_point](const light& a, const light& b) {
            float dist_a = length(a.position - reference_point);
            float dist_b = length(b.position - reference_point);
            return dist_a < dist_b;
        });
}

/**
 * Device version of light sorting function
 * Uses a simple bubble sort since we can't use std::sort on device
 */
void sort_lights_by_position_device(light* lights, int num_lights, const vec3& reference_point) {
    for (int i = 0; i < num_lights - 1; i++) {
        for (int j = 0; j < num_lights - i - 1; j++) {
            float dist_j = length(lights[j].position - reference_point);
            float dist_j1 = length(lights[j + 1].position - reference_point);
            
            if (dist_j > dist_j1) {
                // Swap lights
                light temp = lights[j];
                lights[j] = lights[j + 1];
                lights[j + 1] = temp;
            }
        }
    }
}

/**
 * CPU version of light clustering function
 * Groups lights into clusters based on their positions
 */
void cluster_lights(light* lights, int num_lights, float cluster_radius) {
    // First sort lights by position
    vec3 center = vec3{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < num_lights; i++) {
        center = center + lights[i].position;
    }
    center = center / static_cast<float>(num_lights);
    
    sort_lights_by_position(lights, num_lights, center);
    
    // Then group nearby lights
    for (int i = 0; i < num_lights; i++) {
        if (lights[i].intensity == 0.0f) continue;  // Skip already processed lights
        
        vec3 cluster_center = lights[i].position;
        float cluster_intensity = lights[i].intensity;
        vec3 cluster_color = lights[i].col;
        int cluster_count = 1;
        
        // Look for nearby lights
        for (int j = i + 1; j < num_lights; j++) {
            if (lights[j].intensity == 0.0f) continue;
            
            float dist = length(lights[j].position - cluster_center);
            if (dist <= cluster_radius) {
                // Merge light into cluster
                cluster_center = (cluster_center * cluster_count + lights[j].position) / (cluster_count + 1);
                cluster_intensity += lights[j].intensity;
                cluster_color = cluster_color + lights[j].col;
                cluster_count++;
                
                // Mark light as processed
                lights[j].intensity = 0.0f;
            }
        }
        
        // Update cluster center light
        lights[i].position = cluster_center;
        lights[i].intensity = cluster_intensity;
        lights[i].col = cluster_color / static_cast<float>(cluster_count);
    }
}

// ... rest of existing code ...
