#ifndef VOLUME_H
#define VOLUME_H

#include "light.h"
#include "kdtree.h"



__host__ __device__ float getDensityAtPositionDevice(float* grid, int nx, int ny, int nz, vec3 grid_min, vec3 grid_max, vec3 grid_center, vec3 pos_scene) 
{
    // Hardcoded
    const vec3 targetBase = vec3{0.0f, -50.0f, 0.0f};
    const float scaleFactor = 2.0f;

    // Scene → Original world
    vec3 pos_world = (pos_scene - targetBase) / scaleFactor + grid_center;

    // World → Grid local
    float gx = ((pos_world.x - grid_min.x) / (grid_max.x - grid_min.x)) * (nx - 1);
    float gy = ((pos_world.y - grid_min.y) / (grid_max.y - grid_min.y)) * (ny - 1);
    float gz = ((pos_world.z - grid_min.z) / (grid_max.z - grid_min.z)) * (nz - 1);

    gx = fmaxf(0.0f, fminf(gx, nx - 1.001f));
    gy = fmaxf(0.0f, fminf(gy, ny - 1.001f));
    gz = fmaxf(0.0f, fminf(gz, nz - 1.001f));

    int x0 = floorf(gx);
    int y0 = floorf(gy);
    int z0 = floorf(gz);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    float xd = gx - x0;
    float yd = gy - y0;
    float zd = gz - z0;

    #define GRID(x,y,z) grid[(x) + (y) * nx + (z) * nx * ny]

    float c000 = GRID(x0, y0, z0);
    float c100 = GRID(x1, y0, z0);
    float c010 = GRID(x0, y1, z0);
    float c110 = GRID(x1, y1, z0);
    float c001 = GRID(x0, y0, z1);
    float c101 = GRID(x1, y0, z1);
    float c011 = GRID(x0, y1, z1);
    float c111 = GRID(x1, y1, z1);

    float c00 = c000 * (1 - xd) + c100 * xd;
    float c01 = c001 * (1 - xd) + c101 * xd;
    float c10 = c010 * (1 - xd) + c110 * xd;
    float c11 = c011 * (1 - xd) + c111 * xd;

    float c0 = c00 * (1 - yd) + c10 * yd;
    float c1 = c01 * (1 - yd) + c11 * yd;

    return c0 * (1 - zd) + c1 * zd;
}

__device__ __host__ int get_random(int seed, int index, int m)
{
    unsigned int hash = seed * 1664525u + index * 1013904223u;
    return hash % (m + 1);
}

__device__ __host__ void generate_random_array(int* out_array, int n, int m, int seed)
{
    for (int i = 0; i < n; ++i)
    {
        out_array[i] = get_random(seed, i, m);
    }
}


/**
 * Renders a realistic self-illuminating explosion with smooth billowing black clouds
 * The illumination comes from the passed light sources, creating dramatic lighting effects
 * through the volumetric smoke/explosion.
 */


 __device__ __host__ vec3 getDensityGradient(
    float* d_density_grid,
    int nx, int ny, int nz,
    const vec3& min, const vec3& max,
    const vec3& center,
    const vec3& p,
    float h)
{
    // Calculate gradient using central differences
    float dx = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p + vec3{h, 0, 0}) - 
               getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p - vec3{h, 0, 0});
    
    float dy = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p + vec3{0, h, 0}) - 
               getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p - vec3{0, h, 0});
    
    float dz = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p + vec3{0, 0, h}) - 
               getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p - vec3{0, 0, h});
    
    return vec3{dx, dy, dz} / (2.0f * h);
}


/**
 * Renders a realistic self-illuminating explosion with smooth billowing black clouds
 * The illumination comes from the passed light sources, creating dramatic lighting effects
 * through the volumetric smoke/explosion.
 */
__device__ __host__ vec3 render_volume_self(
    const vec3& o, const vec3& d,
    const vec3& min, const vec3& max,
    float t_near,
    light* lights, int num_lights,
    float* d_density_grid,
    int nx, int ny, int nz,
    const vec3& center)
{
    const float ds = 0.03f;            // Smaller step size for better detail
    const int Nsteps = 300;            // More steps for better quality
    const float sigma_s = 4.0f;        // Higher scattering for stronger light interaction
    const float sigma_a = 0.5f;        // Reduced absorption for more greyish appearance
    const float sigma_t = sigma_s + sigma_a;  // extinction coefficient
    const float phase_g = 0.4f;        // Adjusted anisotropy for better light distribution
    // const int Nsample = NUM;           // Random lights count
    const float light_radius = 1.8f ;   // Significantly increased light influence radius
    const float inv_r2 = 1.0f / (light_radius * light_radius);
    const float atten_k = 0.01f;     // Further reduced attenuation for brighter illumination
    
    // Color temperature adjustment for realistic fire/explosion
    const vec3 hot_color = vec3{1.0f, 0.8f, 0.4f};    // Brighter orange-yellow for hot spots
    const vec3 cool_color = vec3{0.2f, 0.2f, 0.22f};  // More greyish for smoke areas
    
    vec3 color = vec3{0.0f};           // accumulated radiance
    vec3 Tr = vec3{1.0f};              // transmittance
    vec3 p = o + d * t_near;           // current sample
    
    // Select random lights for sampling
    // int light_idx[NUM];
    // generate_random_array(light_idx, Nsample, num_lights, 556);
    
    // Edge darkening factor for billowing effect
    const float edge_contrast = 1.5f;
    
    for (int i = 0; i < Nsteps; ++i) {
        p += d * ds;
        
        // Early termination if out of bounds
        if (p.x < min.x || p.x > max.x ||
            p.y < min.y || p.y > max.y ||
            p.z < min.z || p.z > max.z)
            break;
        
        // Get density at current position
        float rho = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p);
        
        // Skip low-density regions
        if (rho <= 1e-5f) continue;  // Lower density threshold for more transparency
        
        // Calculate distance from center for edge darkening
        vec3 rel_pos = p - center;
        float dist_from_center = length(rel_pos);
        float normalized_dist = clamp(dist_from_center / length(max - min) * 2.0f, 0.0f, 1.0f);
        
        // Apply less edge darkening for more greyish billowing effect
        float density_scale = rho * mix(1.2f, edge_contrast, normalized_dist * 0.7f);
        
        // Density gradient for more interesting variation
        vec3 density_grad = getDensityGradient(d_density_grid, nx, ny, nz, min, max, center, p, ds);
        float grad_mag = length(density_grad);
        
        // Enhance edges where gradient is high (billowing effect) with increased intensity
        float edge_factor = clamp(grad_mag * 8.0f, 0.0f, 1.0f);
        
        // Light accumulation
        vec3 Lsum = vec3{0.0f};
        for (int si = 0; si < num_lights; ++si) {
            // int li = light_idx[si];
            vec3 toL = lights[si].position - p;
            float dist2 = dot(toL, toL);
            
            if (dist2 > light_radius * light_radius) continue;
            
            float dist = sqrt(dist2);
            vec3 wi = toL / dist; // Normalize direction to light
            
            // Calculate light transmittance with reduced shadowing for brighter effect
            float Tr_light = 1.0f;
            vec3 ps = p;
            float traveled = 0.0f;
            
            // March towards light to calculate occlusion with larger steps for less shadowing
            float shadow_ds = ds * 1.5f * 2.0f;
            while (traveled < dist) {
                ps += wi * shadow_ds;
                float rs = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, ps);
                
                // Apply lighter non-linear density mapping for less dramatic shadows
                rs = pow(rs, 1.2f) * 0.8f;
                
                Tr_light *= exp(-sigma_t * rs * shadow_ds);
                if (Tr_light < 1e-4f) break;
                traveled += shadow_ds;
            }
            
            // Enhanced distance attenuation model with boost factor
            float light_boost = 7.5f * 1.5f; // Boost light intensity
            float dist_factor = 1.0f / (1.0f + dist * atten_k);
            float atten = Tr_light * dist_factor * light_boost;
            
            // Improved phase function (Henyey-Greenstein)
            float cosTh = dot(wi, -d); // Note: negated d for backward scattering
            float denom = 1.0f + phase_g*phase_g - 2.0f*phase_g*cosTh;
            float phase = (1.0f - phase_g*phase_g) / (4.0f * M_PI * pow(denom, 1.5f));
            
            // Add contribution from this light
            Lsum += lights[si].col * lights[si].intensity * atten * phase;
        }
        
        // Temperature-based color mixing (hot core, cool outer smoke)
        vec3 local_color = mix(hot_color, cool_color, normalized_dist);
        
        // Calculate in-scattering with temperature coloring
        vec3 inscatter = sigma_s * density_scale * Lsum * local_color;
        
        // Apply stronger edge enhancement for more dramatic billowing effect
        inscatter = mix(inscatter, inscatter * 2.2f, edge_factor);
        
        // Add emissive component for brighter core areas
        float emissive_factor = clamp(1.0f - normalized_dist * 2.0f, 0.0f, 1.0f);
        vec3 emissive = hot_color * emissive_factor * rho * 0.8f;
        
        // Accumulate color with emissive component and update transmittance
        color += Tr * (inscatter + emissive) * ds;
        
        // Less aggressive transmittance reduction for more greyish appearance
        Tr *= exp(-sigma_t * density_scale * ds * 0.6f);  // Reduced transmittance reduction
        
        // Early termination for efficiency
        if (Tr.x + Tr.y + Tr.z < 1e-3f) break;
    }
    
    return color;
}


/**
 * Helper function to calculate density gradient for billowing effects
 */


// KDTree ray tracing


__device__ __host__ void radiusSearch(
    KDNode* node,
    const vec3& query,
    float radius,
    KDNode** out_results,
    int* out_count,
    int max_results,
    int depth = 0)
{
    if (!node) return;

    // Compute distance to current node
    float dist2 = (node->position - query).lengthSquared();
    if (dist2 <= radius * radius) {
        if (*out_count < max_results) {
            out_results[*out_count] = node;
            (*out_count)++;
        }
    }

    // Select axis
    int axis = depth % 3;
    float diff = 0.0f;

    if (axis == 0) diff = query.x - node->position.x;
    else if (axis == 1) diff = query.y - node->position.y;
    else diff = query.z - node->position.z;

    // Choose side to recurse
    if (diff <= 0.0f) {
        // Left first
        radiusSearch(node->left, query, radius, out_results, out_count, max_results, depth + 1);
        if (fabsf(diff) <= radius)
            radiusSearch(node->right, query, radius, out_results, out_count, max_results, depth + 1);
    }
    else {
        // Right first
        radiusSearch(node->right, query, radius, out_results, out_count, max_results, depth + 1);
        if (fabsf(diff) <= radius)
            radiusSearch(node->left, query, radius, out_results, out_count, max_results, depth + 1);
    }
}

__device__ __host__ vec3 render_volume_kdtree(
    const vec3& o, const vec3& d,
    const vec3& min, const vec3& max,
    float t_near,
    float* d_density_grid,
    int nx, int ny, int nz,
    const vec3& center,
    KDNode** d_lighting_grids, int num_lighting_grids,
    float voxel_size)
{
    vec3 p = o + d * t_near;
    const float step_size = voxel_size * 0.5f;
    const int max_steps = 200;

    vec3 accumulated_color = vec3{0.0f};
    float transmittance = 1.0f;

    for (int step = 0; step < max_steps; ++step) 
    {
        // Sample density at current position
        float density = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p);

        if (density > 0.001f)
        {
            vec3 lighting = vec3{0.0f};

            // Gather nearby lights from all KD trees
            for (int i = 0; i < num_lighting_grids; i++) 
            {
                KDNode* root = d_lighting_grids[i];
                const float search_radius = voxel_size * 2.0f;

                // gatherLights will collect lights into temporary buffer
                const int max_results = 32;
                KDNode* results[max_results];
                int result_count = 0;

                radiusSearch(root, p, search_radius, results, &result_count, max_results);

                for (int j = 0; j < result_count; ++j) 
                {
                    KDNode* ln = results[j];
                    float dist2 = (ln->position - p).lengthSquared();
                    float atten = 1.0f / (1.0f + dist2 * 0.1f);
                    lighting += ln->color * ln->intensity * atten;
                }
            }

            vec3 scatter = lighting * density;
            accumulated_color += transmittance * scatter;

            float absorption = density * 0.05f;
            transmittance *= expf(-absorption);

            if (transmittance < 0.01f) break;
        }

        p += d * step_size;

        // Break if outside volume
        if (p.x < min.x || p.y < min.y || p.z < min.z ||
            p.x > max.x || p.y > max.y || p.z > max.z)
            break;
    }

    return accumulated_color;
}



#endif // VOLUME_H