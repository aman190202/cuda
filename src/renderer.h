#include "vec.h"
#include "light.h"

#define NO_INTERSECTION 99999.0f
#define EPSILON           1e-6f

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
        { {  0,  10,  0 }, {  0,-1, 0 }, -10, 10, -10, 10 },  // roof (x,z)
        { {  0, -10,  0 }, {  0, 1, 0 }, -10, 10, -10, 10 },  // floor (x,z)
        { { -10,  0,  0 }, {  1, 0, 0 }, -10, 10, -10, 10 },  // left (y,z)
        { {  10,  0,  0 }, { -1, 0, 0 }, -10, 10, -10, 10 },  // right (y,z)
        { {  0,   0, -10 }, {  0, 0, 1 }, -10, 10, -10, 10 }   // back  (x,y)
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
    float attenuation = 1.0f / (1.0f + light_distance * light_distance);
    
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
    const vec3&    normal)
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

// top‑level ray trace: only Cornell box
__device__ __host__ vec3 trace_ray(
    const vec3& ray_origin,
    const vec3& ray_direction,
    light*      lights,
    int         num_lights)
{
    vec3 normal{0,0,0};
    float t = intersect_cornell_box(ray_origin, ray_direction, lights, num_lights, normal);

    if (t >= NO_INTERSECTION)
        return vec3{0,0,0};  // miss

    return cornellBox(ray_origin, ray_direction, t, lights, num_lights, normal);
}