#include "vec.h"
#include "light.h"
#include "vdb_reader.h"
#include <curand_kernel.h>

#define NO_INTERSECTION 99999.0f
#define EPSILON           1e-6f



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




// intersect the five faces of a Cornell box (roof, floor, left, right, back)
// returns the smallest positive t, and writes the hit normal
__device__ __host__ float intersect_cornell_box(
    const vec3& o, const vec3& d,
    light* lights, int num_lights,
    vec3& normal)
{
    float t_min = NO_INTERSECTION;

    struct Face { vec3 p0, n; float min1, max1, min2, max2; };
    Face faces[] = {
        // p0           normal        extent1-min extent1-max extent2-min extent2-max
        { {  0,  50,  0 }, {  0,-1, 0 }, -50, 50, -50, 50 },  // roof (x,z)
        { {  0, -50,  0 }, {  0, 1, 0 }, -50, 50, -50, 50 },  // floor (x,z)
        { { -50,  0,  0 }, {  1, 0, 0 }, -50, 50, -50, 50 },  // left (y,z)
        { {  50,  0,  0 }, { -1, 0, 0 }, -50, 50, -50, 50 },  // right (y,z)
        { {  0,   0, -50 }, {  0, 0, 1 }, -50, 50, -50, 50 }   // back  (x,y)
    };

    for (int i = 0; i < 5; ++i) {
        vec3  p0    = faces[i].p0;
        vec3& n     = faces[i].n;
        float denom = dot(d, n);

        if (fabsf(denom) > EPSILON) {
            float t = dot(p0 - o, n) / denom;
            if (t > EPSILON && t < t_min) {
                vec3 p = o + d * t;

                float c1, c2;
                if (i <= 1) {
                    // roof/floor: check x,z
                    c1 = p.x;  c2 = p.z;
                }
                else if (i <= 3) {
                    // left/right: check y,z
                    c1 = p.y;  c2 = p.z;
                }
                else {
                    // back: check x,y
                    c1 = p.x;  c2 = p.y;
                }

                if (c1 >= faces[i].min1 && c1 <= faces[i].max1 &&
                    c2 >= faces[i].min2 && c2 <= faces[i].max2)
                {
                    t_min = t;
                    normal = n;
                }
            }
        }
    }

    return t_min;
}

__device__ __host__ vec3 light_interaction(light l, vec3 normal, vec3 intersection_point, vec3 color)
{
    vec3 light_direction = normalize(l.position - intersection_point);
    float light_distance = length(l.position - intersection_point);
    
    // Add a minimum distance to prevent extreme attenuation
    float min_distance = 0.1f;
    light_distance = fmaxf(light_distance, min_distance);
    
    // Smoother attenuation with a minimum threshold
    float attenuation = 1.0f / (0.2f + light_distance * light_distance * light_distance * 5.0f);
    
    float diffuse = max(dot(normal, light_direction), 0.0f);
    vec3 diffuse_color = diffuse * l.col * attenuation * color * l.intensity;

    return diffuse_color;
}

// shade the hit in Cornell box by face normal
__device__ __host__ vec3 cornellBox(
    const vec3&    ray_origin,
    const vec3&    ray_direction,
    float           t,
    light*         lights,
    int            num_lights,
    const vec3&    normal,
    const vec3&    min,
    const vec3&    max)
{
    vec3 intersection_point = ray_origin + ray_direction * t;
    vec3 wall_color = vec3{0,0,0};
    
    // uniform cream for roof/floor
    if (normal == vec3{0,-1,0} || normal == vec3{0,1,0})
        wall_color = vec3{0.9f, 0.9f, 0.7f};
    // dark red left wall
    else if (normal == vec3{1,0,0})
        wall_color = vec3{0.8f, 0.1f, 0.1f};
    // dark green right wall
    else if (normal == vec3{-1,0,0})
        wall_color = vec3{0.1f, 0.8f, 0.1f};
    // off-white back wall
    else if (normal == vec3{0,0,1})
        wall_color = vec3{0.9f, 0.9f, 0.9f};

    // Accumulate lighting from all lights
    vec3 final_color = vec3{0,0,0};
    for (int i = 0; i < num_lights; i++) {
        final_color = final_color + light_interaction(lights[i], normal, intersection_point, wall_color);
    }


    return final_color;
}

__device__ __host__ float intersect_sphere(const vec3& o, const vec3& d, const vec3& center, float radius)
{
    float r = radius;
    vec3 oc = o - center;
    float a = dot(d, d);
    float b = 2 * dot(oc, d);
    float c = dot(oc, oc) - r * r;
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return NO_INTERSECTION;
    }
    float t = (-b - sqrt(discriminant)) / (2 * a);
    return t;
}   

__device__ __host__ float hitLight(const vec3& ray_origin, const vec3& ray_direction, light* lights, int num_lights)
{
    float t_min = NO_INTERSECTION;
    for (int i = 0; i < num_lights; i++) {
        float radius = 0.1f * lights[i].intensity;   
        float t = intersect_sphere(ray_origin, ray_direction, lights[i].position, radius);
        if (t < t_min) {
            t_min = t;
        }
    }
    return t_min;
}

__device__ __host__ vec3 render_light(const vec3& ray_origin, const vec3& ray_direction, light* lights, int num_lights, float t)
{
    vec3 color = vec3{0,0,0};
    for (int i = 0; i < num_lights; i++) {
        float radius = lights[i].intensity;    
        float t_check = intersect_sphere(ray_origin, ray_direction, lights[i].position, radius);
        if (t_check != NO_INTERSECTION) 
        {
            color = color + lights[i].col * lights[i].intensity;
        }
    }
    return color;
}

__device__ __host__ float intersect_box(const vec3& o, const vec3& d, const vec3& min, const vec3& max)
{
    // Calculate inverse direction to avoid division

    vec3 inv_d = vec3(1.0f) / d;
    
    // Calculate t values for each slab
    float t1 = (min.x - o.x) * inv_d.x;
    float t2 = (max.x - o.x) * inv_d.x;
    float t3 = (min.y - o.y) * inv_d.y;
    float t4 = (max.y - o.y) * inv_d.y;
    float t5 = (min.z - o.z) * inv_d.z;
    float t6 = (max.z - o.z) * inv_d.z;
    
    // Find min and max t values for each axis
    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));
    
    // If tmax < 0, ray is intersecting but in the opposite direction
    if (tmax < 0) return -1.0f;
    
    // If tmin > tmax, ray doesn't intersect the box
    if (tmin > tmax) return -1.0f;
    
    // Return the first intersection point (tmin)
    return tmin;
}

// Helper function for random light sampling (device only)
__device__ int getRandomLightIndex(curandState_t* state, int num_lights) {
    return (int)(curand_uniform(state) * num_lights) % num_lights;
}

__device__ __forceinline__ vec3 render_volume(
    const vec3& o, const vec3& d,
    const vec3& min, const vec3& max,
    float t_near,
    light* lights, int num_lights,
    float* d_density_grid,
    int nx,
    int ny,
    int nz,
    const vec3& center)
{
    const float step_size = 0.1f;
    const float max_distance = 50.0f;
    const float sigma_s = 0.5f;  // scattering
    const float sigma_a = 0.1f;  // absorption
    const float sigma_t = sigma_s + sigma_a;
    const float light_radius = 2.0f;

    vec3 color = vec3{0, 0, 0};
    vec3 transmittance = vec3{1.0f};

    vec3 ray_origin = o;
    vec3 ray_dir = d;
    float t_total = 0.0f;

    for (int bounce = 0; bounce < 20; ++bounce)
    {
        while (t_total < max_distance) 
        {
            vec3 p = ray_origin + ray_dir * t_total;

            // If out of volume bounds → hit Cornell box
            if (p.x < min.x || p.x > max.x ||
                p.y < min.y || p.y > max.y ||
                p.z < min.z || p.z > max.z)
            {
                vec3 normal{0, 0, 0};
                float t_box = intersect_cornell_box(ray_origin, ray_dir, lights, num_lights, normal);
                return color + transmittance * cornellBox(ray_origin, ray_dir, t_box, lights, num_lights, normal, min, max);
            }

            float density = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p);

            // Self emission (glow in dense areas) → ALWAYS
            if (density > 0.1f)
            {
                vec3 self_emission = vec3(density * 0.2f);
                color = color + transmittance * self_emission;
            }

            // If sufficiently dense → check nearby lights
            if (density >= 0.2f)
            {
                vec3 light_contrib = vec3{0, 0, 0};
                bool has_nearby_light = false;

                for (int i = 0; i < num_lights; ++i)
                {
                    light l = lights[i];
                    float dist = length(l.position - p);

                    if (dist < light_radius * 5.0f)  // sampling radius
                    {
                        has_nearby_light = true;
                        float attenuation = 1.0f / (1.0f + dist * dist * 0.05f); // softer attenuation
                        light_contrib = light_contrib + l.col * l.intensity * attenuation;
                    }
                }

                if (has_nearby_light)
                {
                    color = color + transmittance * light_contrib * density;

                    // Update transmittance
                    float extinction = density * sigma_t * step_size;
                    transmittance = transmittance * expf(-extinction);

                    // Prevent transmittance from becoming zero
                    transmittance.x = fmaxf(transmittance.x, 0.01f);
                    transmittance.y = fmaxf(transmittance.y, 0.01f);
                    transmittance.z = fmaxf(transmittance.z, 0.01f);

                    // Slightly randomize direction (optional -> here, continue same)
                    ray_origin = p;
                    t_total = 0.0f;

                    // Continue to next bounce
                    break;
                }
            }

            // Update transmittance even when no nearby light
            float extinction = density * sigma_t * step_size;
            transmittance = transmittance * expf(-extinction);

            // Clamp transmittance
            transmittance.x = fmaxf(transmittance.x, 0.01f);
            transmittance.y = fmaxf(transmittance.y, 0.01f);
            transmittance.z = fmaxf(transmittance.z, 0.01f);

            t_total += step_size;
        }

        // Ray out of volume after stepping
        if (t_total >= max_distance)
            break;
    }

    return color;
}


// Ray trace logic (fixed t_box check)
__device__ __forceinline__ vec3 trace_ray(
    const vec3& ray_origin,
    const vec3& ray_direction,
    light* lights,
    int num_lights,
    vec3& min,
    vec3& max,
    vec3& center,
    float* d_density_grid,
    int nx,
    int ny,
    int nz)
{
    vec3 normal{0, 0, 0};
    float t = intersect_cornell_box(ray_origin, ray_direction, lights, num_lights, normal);
    float t_box = intersect_box(ray_origin, ray_direction, min, max);

    if (t >= NO_INTERSECTION)
        return vec3{0, 0, 0}; // miss

    if (t_box != -1.0f)  // safer float comparison instead of t_box != -1
        return render_volume(ray_origin, ray_direction, min, max, t_box, lights, num_lights, d_density_grid, nx, ny, nz, center);

    return cornellBox(ray_origin, ray_direction, t, lights, num_lights, normal, min, max);
}