#ifndef LIGHT_H
#define LIGHT_H

#include "vec.h"    


struct color{
    float r, g, b;
};

struct light
{
    vec3 position;
    vec3 col;
    float intensity;

    // Default constructor
    __device__ __host__ light() : position(vec3(0)), col(vec3(1)), intensity(0.0f) {}

    // Constructor with position and intensity
    __device__ __host__ light(const vec3& pos, float intens) 
        : position(pos), col(vec3(1)), intensity(intens) {}

    // Full constructor
    __device__ __host__ light(const vec3& pos, const vec3& color, float intens)
        : position(pos), col(color), intensity(intens) {}
};

struct density_sample {
    vec3 position;
    float value;
};

#endif