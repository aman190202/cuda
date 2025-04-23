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
};

#endif