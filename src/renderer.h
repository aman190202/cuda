#include "vec.h"

__device__ __host__ vec3 trace_ray(const vec3& ray_origin, const vec3& ray_direction)
{
    vec3 sphere_center = vec3{0, 0, -10};
    float sphere_radius = 1;


    // Calculate ray-sphere intersection using quadratic equation
    vec3 oc = ray_origin - sphere_center;
    float a = dot(ray_direction, ray_direction);
    float b = 2.0f * dot(oc, ray_direction);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) 
        return vec3{1, 0, 0};
    else
        return vec3{0.5f, 0.5f, 0.5f};
    
}