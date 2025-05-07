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

    // Array access operator
    __device__ __host__ float& operator[](int i) {
        return (&x)[i];
    }

    // Const array access operator
    __device__ __host__ const float& operator[](int i) const {
        return (&x)[i];
    }
};

inline __device__ __host__ vec3 operator+(const vec3& a, const vec3& b){
    return vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __device__ __host__ vec3 operator-(const vec3& a, const vec3& b){
    return vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

inline __device__ __host__ vec3 operator*(const vec3& a, const vec3& b){
    return vec3{a.x * b.x, a.y * b.y, a.z * b.z};
}

inline __device__ __host__ vec3 operator*(const vec3& a, float b){
    return vec3{a.x * b, a.y * b, a.z * b};
}   

inline __device__ __host__ vec3 operator*(float a, const vec3& b){
    return vec3{a * b.x, a * b.y, a * b.z};
}   

inline __device__ __host__ vec3 operator/(const vec3& a, float b){
    return vec3{a.x / b, a.y / b, a.z / b};
}

inline __device__ __host__ vec3 operator/(const vec3& a, const vec3& b){
    return vec3{a.x / b.x, a.y / b.y, a.z / b.z};
}

inline __device__ __host__ vec3 cross(const vec3& a, const vec3& b){
    return vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline __device__ __host__ float dot(const vec3& a, const vec3& b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ __host__ float length(const vec3& a){
    return sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}   

inline __device__ __host__ vec3 normalize(const vec3& a){
    return a / length(a);
}   

inline __device__ __host__ vec3 reflect(const vec3& a, const vec3& n){
    return a - n * dot(a, n) * 2;
}   

inline __device__ __host__ float distance(const vec3& a, const vec3& b){
    return length(a - b);
}   

inline __device__ __host__ void normalize(vec3& a){
    a = a / length(a);
}

// for == operator
inline __device__ __host__ bool operator==(const vec3& a, const vec3& b){
    return a.x == b.x && a.y == b.y && a.z == b.z;
}      

// for != operator
inline __device__ __host__ bool operator!=(const vec3& a, const vec3& b){
    return !(a == b);
}

// Stream output operator for vec3
inline std::ostream& operator<<(std::ostream& os, const vec3& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

// Compound assignment operators
inline __device__ __host__ vec3& operator+=(vec3& a, const vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline __device__ __host__ vec3& operator*=(vec3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
}

// Utility functions
inline __device__ __host__ float clamp(float x, float min_val, float max_val) {
    return fmaxf(min_val, fminf(x, max_val));
}

inline __device__ __host__ float mix(float a, float b, float t) {
    return a + (b - a) * t;
}

inline __device__ __host__ vec3 mix(const vec3& a, const vec3& b, float t) {
    return a + (b - a) * t;
}

// Unary minus operator
inline __device__ __host__ vec3 operator-(const vec3& a) {
    return vec3{-a.x, -a.y, -a.z};
}

#endif // VEC_H
















