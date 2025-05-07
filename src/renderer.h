#include "vec.h"
#include "light.h"
#include "vdb_reader.h"
#include "volume.h"
#include <curand_kernel.h>
#include "kdtree.h"

#define NO_INTERSECTION 99999.0f
#define EPSILON           1e-6f

// intersect the five faces of a Cornell box (roof, floor, left, right, back)
// returns the smallest positive t, and writes the hit normal


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
    float          t,
    light*         lights,
    int            num_lights,
    const vec3&    normal,
    const vec3&    min,
    const vec3&    max_l,
    float*         d_density_grid,
    int            nx,
    int            ny,
    int            nz,
    const vec3&    center)
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
    
    vec3 direction = center - intersection_point;
    normalize(direction);

    float box_intersection = intersect_box(ray_origin, direction, min, max_l);
    float attenuation = 1.0f / (box_intersection * box_intersection);
    float light_intensity = max(dot(normal, direction), 0.0f);
    final_color =  light_intensity * wall_color * render_volume_self(ray_origin, direction, min, max_l, box_intersection, lights, num_lights, d_density_grid, nx, ny, nz, center) * 2.0f ;

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





__device__ __host__ vec3 render_volume(
    const vec3& o, const vec3& d,
    const vec3& min, const vec3& max,
    float t_near,
    light* lights, int num_lights,
    float* d_density_grid,
    int nx, int ny, int nz,
    const vec3& center)
{
    // Parameters
    const float step_size      = 1.0f;
    const float max_distance   = 50.0f;
    const float sigma_s        = 0.5f;
    const float sigma_a        = 0.1f;
    const float sigma_t_step   = (sigma_s + sigma_a) * step_size;
    const float light_radius   = 0.25f;          // effective radius
    const float inv_radius2    = 1.0f / (light_radius * light_radius);
    const float atten_k        = 0.1f;           // attenuation coefficient

    vec3 color           = vec3{0.0f};
    vec3 transmittance   = vec3{1.0f};

    float t_total = 0.0f;
    float t_offset = t_near;

    // March through volume once (no bounce)
    while (t_total < max_distance && fmaxf(transmittance.x, fmaxf(transmittance.y, transmittance.z)) > 0.01f)
    {
        vec3 p = o + d * (t_offset + t_total);

        // exit volume
        if (p.x < min.x || p.x > max.x ||
            p.y < min.y || p.y > max.y ||
            p.z < min.z || p.z > max.z)
        {
            vec3 normal{0,0,0};
            float t_box = intersect_cornell_box(o, d, lights, num_lights, normal);
            return color + transmittance * cornellBox(o, d, t_box, lights, num_lights, normal, min, max, d_density_grid, nx, ny, nz, center);
        }

        // sample density
        float density = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, p);
        if (density > 0.3f)
        {
            // self‐emission
            color = color + transmittance * (density * 0.2f);

            // accumulate light contributions efficiently
            vec3 L = vec3{0.0f};
            for (int i = 0; i < num_lights; ++i)
            {
                vec3 lp = lights[i].position - p;
                float dist2 = dot(lp, lp);
                if (dist2 < light_radius * light_radius)
                {
                    float dist = sqrtf(dist2);
                    float att = expf(-dist * atten_k) * inv_radius2;
                    L = L + lights[i].col * lights[i].intensity * att;
                }
            }
            if (L.x > 0 || L.y > 0 || L.z > 0)
            {
                color = color + transmittance * (L * density);
            }

            // update transmittance
            vec3 e = expf(-sigma_t_step * density);
            transmittance = transmittance * e;
        }

        t_total = t_total + step_size;
    }

    return color;
}


// Ray trace logic (fixed t_box check)
__device__ __host__ __forceinline__ vec3 trace_ray(
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

    bool use_kdtree = false;
    
    vec3 normal{0, 0, 0};
    float t = intersect_cornell_box(ray_origin, ray_direction, lights, num_lights, normal);
    float t_box = intersect_box(ray_origin, ray_direction, min, max);

    if (t >= NO_INTERSECTION)
        return vec3{0, 0, 0}; // miss

    vec3 illumination = vec3{0.0f};
    if (t_box != -1.0f && !use_kdtree)  // safer float comparison instead of t_box != -1
        illumination = render_volume_self(ray_origin, ray_direction, min, max, t_box, lights, num_lights, d_density_grid, nx, ny, nz, center);
    else if (use_kdtree && t_box != -1.0f)
        illumination = render_volume_kdtree(ray_origin, ray_direction, min, max, t_box, lights, num_lights, d_density_grid, nx, ny, nz, center);

    if(illumination != vec3{0.0f})
        return illumination;


    return cornellBox(ray_origin, ray_direction, t, lights, num_lights, normal, min, max, d_density_grid, nx, ny, nz, center);


}