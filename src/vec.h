#ifndef VEC_H
#define VEC_H

struct vec3{
    float x, y, z;
    
    // Default constructor
    __device__ __host__ vec3() : x(0), y(0), z(0) {}
    
    // Constructor for all components
    __device__ __host__ vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    // Constructor for same value in all components
    __device__ __host__ vec3(float v) : x(v), y(v), z(v) {}
};

__device__ __host__ vec3 operator+(const vec3& a, const vec3& b){
    return vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ __host__ vec3 operator-(const vec3& a, const vec3& b){
    return vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ __host__ vec3 operator*(const vec3& a, const vec3& b){
    return vec3{a.x * b.x, a.y * b.y, a.z * b.z};
}

__device__ __host__ vec3 operator*(const vec3& a, float b){
    return vec3{a.x * b, a.y * b, a.z * b};
}   

__device__ __host__ vec3 operator*(float a, const vec3& b){
    return vec3{a * b.x, a * b.y, a * b.z};
}   

__device__ __host__ vec3 operator/(const vec3& a, float b){
    return vec3{a.x / b, a.y / b, a.z / b};
}

__device__ __host__ vec3 operator/(const vec3& a, const vec3& b){
    return vec3{a.x / b.x, a.y / b.y, a.z / b.z};
}

__device__ __host__ vec3 cross(const vec3& a, const vec3& b){
    return vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

__device__ __host__ float dot(const vec3& a, const vec3& b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ float length(const vec3& a){
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}   

__device__ __host__ vec3 normalize(const vec3& a){
    return a / length(a);
}   

__device__ __host__ vec3 reflect(const vec3& a, const vec3& n){
    return a - n * dot(a, n) * 2;
}   

__device__ __host__ float distance(const vec3& a, const vec3& b){
    return length(a - b);
}   

__device__ __host__ void normalize(vec3& a){
    a = a / length(a);
}

// for == operator
__device__ __host__ bool operator==(const vec3& a, const vec3& b){
    return a.x == b.x && a.y == b.y && a.z == b.z;
}      

#endif // VEC_H
















