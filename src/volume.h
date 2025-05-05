#include "light.h"

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


__device__ __host__ vec3 render_volume_self(
    const vec3& o, const vec3& d,
    const vec3& min, const vec3& max,
    float t_near,
    light* lights, int num_lights,
    float* d_density_grid,
    int nx, int ny, int nz,
    const vec3& center)
{
    const float step_size = 0.1f;
    const int step_count = 200;
    const float absorption_coef = 0.02f;
    const float scattering_coef = 0.08f;
    const float extinction_coef = absorption_coef + scattering_coef;

    vec3 illumination = vec3{0.0f};
    float transmittance = 1.0f;

    vec3 sample_position = o + d * t_near;

    for (int i = 0; i < step_count; i++) 
    {
        sample_position = sample_position + (d * step_size);

        // Sample density
        float density = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, sample_position);

        // Compute transmittance
        transmittance *= expf(-density * extinction_coef * step_size);

        // --- Self-illumination calculation ---
        vec3 self_illumination = vec3{0.0f};

        for (int l = 0; l < num_lights; ++l) 
        {
            vec3 light_dir = normalize(lights[l].position - sample_position);
            float light_dist = length(lights[l].position - sample_position);

            float light_contrib = 1.0f;
            float t_light = 0.0f;

            // Raymarch towards light to calculate shadowing
            while (t_light < light_dist) 
            {
                vec3 pos = sample_position + light_dir * t_light;

                // Check bounds
                if (pos.x < min.x || pos.x > max.x ||
                    pos.y < min.y || pos.y > max.y ||
                    pos.z < min.z || pos.z > max.z)
                    break;

                float dens_inside = getDensityAtPositionDevice(d_density_grid, nx, ny, nz, min, max, center, pos);

                // Attenuate light contribution by extinction
                light_contrib *= expf(-dens_inside * extinction_coef * step_size);

                // Early termination
                if (light_contrib < 0.01f) 
                {
                    light_contrib = 0.0f;
                    break;
                }

                t_light += step_size;
            }

            self_illumination = self_illumination + lights[l].col * light_contrib;
        }

        // Scattering out
        float out_scattering = scattering_coef * density;

        vec3 current_light = self_illumination * out_scattering;

        illumination = illumination + transmittance * current_light * step_size;
    }

    return illumination;
}