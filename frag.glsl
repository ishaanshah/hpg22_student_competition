// The following built-in quality knobs are available:
//
// (1) Uncomment the preprocessor definition FULL_SCENE to get the full scene.
// (2) Set NUM_SAMPLES to increase the number of rays per pixel.
// 
// We will judge your submitted shader as follows:
// 
//  - Shader run time must not exceed 2x of our baseline shader 
//    with FULL_SCENE and NUM_SAMPLES==1
//
//  - Final image quality will be measured using SSIM against a reference 
//    with FULL_SCENE and NUM_SAMPLES==100000
// 
// Note: Changing these definitions will increase shader compilation times.
// 
// Windows users will need to configure their browser to use the native OpenGL backend.
//
// If you are using Chrome:
//  - Go to chrome://flags and search for "Angle"
//  - Change backend from "Default" to "OpenGL"
//  - Restart your browser
//
#define FULL_SCENE
#define NUM_SAMPLES 5

#define VERSION 2

#define INFINITY 9999999.0 // sorry, webgl doesn't allow to use proper float infinity :(
#define PI 3.141592653589
#define EPS 1e-6

#ifdef FULL_SCENE
#define NUM_BOUNCES 8
#else
#define NUM_BOUNCES 2
#endif

#define MAT_LEFT        0
#define MAT_RIGHT       1
#define MAT_CEILING     2
#define MAT_FLOOR       3
#define MAT_BACK        4
#define MAT_H           5
#define MAT_P           6
#define MAT_G           7
#define MAT_2           8
#define MAT_LIGHT0      9
#define MAT_LIGHT1      10
#define MAT_LIGHT2      11
#define MAT_LIGHT3      12
#define MAT_LIGHT4      13
#define MAT_LIGHT5      14
#define MAT_LIGHT6      15
#define MAT_LIGHT7      16
#define MAT_LIGHT8      17

struct Ray
{
	vec3 origin, dir;
};

struct AABB
{
	vec3 min_, max_;
};

struct MaterialSample
{
	vec3 color;
	float roughness;
	bool is_light;
    int light_idx;
};

int seed;
vec2 frag_coord;

const float cube_light_size = 0.08;
const vec4 cube_light_pos[4] = vec4[4](
	vec4(  -0.9, -1.0 + cube_light_size * 0.495,  0.6, 0.4),
	vec4(  0.3, -1.0 + cube_light_size * 0.495,  0.2, 0.8),
	vec4(  1.0 - 2.0 * cube_light_size, -1.0 + 2.0 * cube_light_size, -1.0 + 5.0 * cube_light_size, 0.0),
	vec4(  -1.0 + 2.0 * cube_light_size, -1.0 + 2.0 * cube_light_size, -0.6, 0.0)
);

const vec3 light_color[4] = vec3[4](
	vec3(5),
	vec3(5),
	vec3(80.0, 50, 30),
	vec3(30, 30, 80.0)
);


// H
const vec4 coordinates_H[3] = vec4[3](
vec4(0.000, 0.000, 0.200, 0.750),
vec4(0.427, 0.000, 0.627, 0.750),
vec4(0.116, 0.310, 0.516, 0.450));
// P
const vec4 coordinates_P[4] = vec4[4](
vec4(0.000, 0.000, 0.200, 0.750),
vec4(0.400, 0.360, 0.540, 0.675),
vec4(0.044, 0.288, 0.471, 0.428),
vec4(0.000, 0.610, 0.471, 0.750));
// G
const vec4 coordinates_G[6] = vec4[6](
vec4(0.000, 0.060, 0.200, 0.670),
vec4(0.425, 0.060, 0.625, 0.265),
vec4(0.425, 0.520, 0.625, 0.670),
vec4(0.100, 0.000, 0.625, 0.140),
vec4(0.315, 0.265, 0.625, 0.405),
vec4(0.077, 0.610, 0.550, 0.750));
// 2
const vec4 coordinates_2[5] = vec4[5](
vec4(0.000, 0.000, 0.140, 0.365) * 0.5,
vec4(0.474, 0.365, 0.614, 0.680) * 0.5,
vec4(0.044, 0.000, 0.614, 0.140) * 0.5,
vec4(0.044, 0.288, 0.544, 0.428) * 0.5,
vec4(0.044, 0.610, 0.544, 0.750) * 0.5);

// TNB, sorry!
mat3
construct_ONB_frisvad(vec3 normal)
{
	mat3 ret;
	ret[1] = normal;
	if(normal.z < -0.999805696) {
		ret[0] = vec3(0.0, -1.0, 0.0);
		ret[2] = vec3(-1.0, 0.0, 0.0);
	}
	else {
		float a = 1.0 / (1.0 + normal.z);
		float b = -normal.x * normal.y * a;
		ret[0] = vec3(1.0 - normal.x * normal.x * a, b, -normal.x);
		ret[2] = vec3(b, 1.0 - normal.y * normal.y * a, -normal.y);
	}
	return ret;
}

vec2
get_random() {
    // Use blue noise texture
    vec2 rng = texelFetch(iChannel0, (ivec2(frag_coord) + 10*seed) % ivec2(iChannelResolution[0].xy), 0).rg;
    seed += 1;
    return rng;
}


// adapted from https://cwyman.org/code/dxrTutors/tutors/Tutor14/tutorial14.md.html
float
ggxNormalDistribution(float NdotH, float roughness)
{
	float a2 = roughness * roughness;
	float d = ((NdotH * a2 - NdotH) * NdotH + 1.0);
	return a2 / (d * d * PI);
}

float
schlickMaskingTerm(float NdotL, float NdotV, float roughness)
{
	// Karis notes they use alpha / 2 (or roughness^2 / 2)
	float k = roughness*roughness / 2.0;

	// Compute G(v) and G(l).  These equations directly from Schlick 1994
	//     (Though note, Schlick's notation is cryptic and confusing.)
	float g_v = NdotV / (NdotV*(1.0 - k) + k);
	float g_l = NdotL / (NdotL*(1.0 - k) + k);
	return g_v * g_l;
}

float schlickSingleMask(float NdotV, float roughness)
{
    float k = roughness*roughness / 2.0;
    return NdotV / (NdotV*(1.0-k) + k);
}

vec3
schlickFresnel(vec3 f0, float lDotH)
{
	return f0 + (vec3(1.0, 1.0, 1.0) - f0) * pow(1.0 - lDotH, 5.0);
}

mat4
rotate_y(float a)
{
	mat4 ret = mat4(1.0);
	ret[0][0] = ret[2][2] = cos(a);
	ret[0][2] = sin(a);
	ret[2][0] = -ret[0][2];
	return ret;
}

mat4
look_at(vec3 eye, vec3 center, vec3 up)
{
	mat4 ret;

	vec3 f = normalize(center - eye);
	vec3 s = normalize(cross(f, normalize(up)));
	vec3 u = cross(s, f);

	ret[0][0] = s[0];
	ret[1][0] = s[1];
	ret[2][0] = s[2];

	ret[0][1] = u[0];
	ret[1][1] = u[1];
	ret[2][1] = u[2];

	ret[0][2] = -f[0];
	ret[1][2] = -f[1];
	ret[2][2] = -f[2];

	ret[0][3] = ret[1][3] = ret[2][3] = 0.0;

	ret[3][0] = -dot(s, eye);
	ret[3][1] = -dot(u, eye);
	ret[3][2] =  dot(f, eye);

	ret[3][3] = 1.0;
	return ret;
}

bool
intersect_aabb(in Ray ray, in AABB aabb, inout float t_min, inout float t_max)
{
	vec3 div = 1.0 / ray.dir;
	vec3 t_1 = (aabb.min_ - ray.origin) * div;
	vec3 t_2 = (aabb.max_ - ray.origin) * div;

	vec3 t_min2 = min(t_1, t_2);
	vec3 t_max2 = max(t_1, t_2);

	t_min = max(max(t_min2.x, t_min2.y), max(t_min2.z, t_min));
	t_max = min(min(t_max2.x, t_max2.y), min(t_max2.z, t_max));

	return t_min < t_max;
}

vec3
ray_at(in Ray ray, float t)
{
	return ray.origin + t * ray.dir;
}

float
intersect_plane(
	Ray ray,
    vec3 center,
    vec3 normal)
{
    float denom = dot(ray.dir, normal);
    float t = dot(center - ray.origin, normal) / denom;
	return t > 0.0 ? t : INFINITY;
}

float
intersect_box(Ray ray, out vec3 normal, vec3 position_min, vec3 position_max)
{
	float t_min = 0.0;
	float t_max = 999999999.0;
	if(intersect_aabb(ray, AABB(position_min, position_max), t_min, t_max)) {
		vec3 p = ray_at(ray, t_min);

		vec3 center = (position_min + position_max) * 0.5;

		normal = p - center;

		vec3 an = abs(normal) / (position_max - position_min);

		if(an.x > an.y && an.x > an.z) {
			normal = vec3(normal.x > 0.0 ? 1.0 : -1.0, 0, 0);
		}
		if(an.y > an.x && an.y > an.z) {
			normal = vec3(0, normal.y > 0.0 ? 1.0 : -1.0, 0);
		}
		if(an.z > an.x && an.z > an.y) {
			normal = vec3(0, 0, normal.z > 0.0 ? 1.0 : -1.0);
		}

		return t_min;
	}

	return INFINITY;
}

float
intersect(Ray ray, bool shadow, inout vec3 p, inout vec3 normal, out MaterialSample ms)
{
	float t_min = INFINITY;

	int material = -1;
{
    vec3 normal_tmp;
    Ray ray_tmp = ray;
    mat4 r = rotate_y(-0.35);
    ray_tmp.origin -= vec3(-0.9, -1, 0.0);
    ray_tmp.dir = vec3(r * vec4(ray_tmp.dir, 0));
    ray_tmp.origin = vec3(r * vec4(ray_tmp.origin, 1.0));
    float H_intersect = intersect_box(ray_tmp, normal_tmp, vec3(0.0, 0.0, 0.0), vec3(0.627, 0.750, 0.150));
    
    if (H_intersect != INFINITY)
	for(int i = 0; i < coordinates_H.length(); i++) {
		vec3 box_origin = vec3(coordinates_H[i].xy, 0.0);
		vec3 box_size = vec3(coordinates_H[i].zw - coordinates_H[i].xy, 0.15);
		float t = intersect_box(ray_tmp, normal_tmp, box_origin, box_origin + box_size);
		if(t < t_min) {
			t_min = t;
			p = ray_at(ray, t);
			material = MAT_H;
			normal = vec3(transpose(r) * vec4(normal_tmp, 0.0));
		}
	}
}

{
    vec3 normal_tmp;
    Ray ray_tmp = ray;
    mat4 r = rotate_y(0.75);
    ray_tmp.origin -= vec3(-0.28, -1, 0.2);
    ray_tmp.dir = vec3(r * vec4(ray_tmp.dir, 0));
    ray_tmp.origin = vec3(r * vec4(ray_tmp.origin, 1.0));
    float P_intersect = intersect_box(ray_tmp, normal_tmp, vec3(0.0, 0.0, 0.0), vec3(0.540, 0.750, 0.150));

    if (P_intersect != INFINITY)
	for(int i = 0; i < coordinates_P.length(); i++) {
		vec3 box_origin = vec3(coordinates_P[i].xy, 0.0);
		vec3 box_size = vec3(coordinates_P[i].zw - coordinates_P[i].xy, 0.15);
		float t = intersect_box(ray_tmp, normal_tmp, box_origin, box_origin + box_size);
		if(t < t_min) {
			t_min = t;
			p = ray_at(ray, t);
			material = MAT_P;
			normal = vec3(transpose(r) * vec4(normal_tmp, 0.0));
		}
	}
}

{
    vec3 normal_tmp;
    Ray ray_tmp = ray;
    mat4 r = rotate_y(-0.4);
    ray_tmp.origin -= vec3(0.35, -1, -0.20);
    ray_tmp.dir = vec3(r * vec4(ray_tmp.dir, 0));
    ray_tmp.origin = vec3(r * vec4(ray_tmp.origin, 1.0));
    float G_intersect = intersect_box(ray_tmp, normal_tmp, vec3(0.0, 0.0, 0.0), vec3(0.625, 0.750, 0.150));

    if (G_intersect != INFINITY)
	for(int i = 0; i < coordinates_G.length(); i++) {
		vec3 box_origin = vec3(coordinates_G[i].xy, 0.0);
		vec3 box_size = vec3(coordinates_G[i].zw - coordinates_G[i].xy, 0.15);
		float t = intersect_box(ray_tmp, normal_tmp, box_origin, box_origin + box_size);
		if(t < t_min) {
			t_min = t;
			p = ray_at(ray, t);
			material = MAT_G;
			normal = vec3(transpose(r) * vec4(normal_tmp, 0.0));
		}
	}
}

#ifdef FULL_SCENE
{
    vec3 normal_tmp;
    Ray ray_tmp = ray;
    mat4 r = rotate_y(0.0);
    ray_tmp.origin -= vec3(0.1, -0.2, -1.0);
    ray_tmp.dir = vec3(r * vec4(ray_tmp.dir, 0));
    ray_tmp.origin = vec3(r * vec4(ray_tmp.origin, 1.0));
    float TWO_intersect = intersect_box(ray_tmp, normal_tmp, vec3(0.0, 0.0, 0.0) * 0.5, vec3(0.614, 0.750, 0.125) * 0.5);

    if (TWO_intersect != INFINITY);
	for(int i = 0; i < coordinates_2.length(); i++) {
		vec3 box_origin = vec3(coordinates_2[i].xy, 0.0);
		vec3 box_size = vec3(coordinates_2[i].zw - coordinates_2[i].xy, 0.125);
		float t = intersect_box(ray_tmp, normal_tmp, box_origin, box_origin + box_size);
		if(t < t_min) {
			t_min = t;
			p = ray_at(ray, t);
			material = MAT_2;
			normal = vec3(transpose(r) * vec4(normal_tmp, 0.0));
		}
	}
}

{
    vec3 normal_tmp;
    Ray ray_tmp = ray;
    mat4 r = rotate_y(0.0);
    ray_tmp.origin -= vec3(0.45, -0.2, -1.0);
    ray_tmp.dir = vec3(r * vec4(ray_tmp.dir, 0));
    ray_tmp.origin = vec3(r * vec4(ray_tmp.origin, 1.0));
    float TWO_intersect = intersect_box(ray_tmp, normal_tmp, vec3(0.0, 0.0, 0.0) * 0.5, vec3(0.614, 0.750, 0.125) * 0.5);

    if (TWO_intersect != INFINITY);
	for(int i = 0; i < coordinates_2.length(); i++) {
		vec3 box_origin = vec3(coordinates_2[i].xy, 0.0);
		vec3 box_size = vec3(coordinates_2[i].zw - coordinates_2[i].xy, 0.125);
		float t = intersect_box(ray_tmp, normal_tmp, box_origin, box_origin + box_size);
		if(t < t_min) {
			t_min = t;
			p = ray_at(ray, t);
			material = MAT_2;
			normal = vec3(transpose(r) * vec4(normal_tmp, 0.0));
		}
	}
}
#endif


	// cube light sources
	for(int i = 0; i < cube_light_pos.length(); i++) {
		vec3 normal_tmp;
		Ray ray_tmp = ray;
		//mat4 r = rotate_y(scene_time);
		mat4 r = rotate_y(-cube_light_pos[i].w);
		ray_tmp.origin -= cube_light_pos[i].xyz;
		ray_tmp.dir = vec3(r * vec4(ray_tmp.dir, 0));
		ray_tmp.origin = vec3(r * vec4(ray_tmp.origin, 1.0));
		float t = intersect_box(ray_tmp, normal_tmp,
				vec3(-cube_light_size * 0.5),
				vec3(cube_light_size * 0.5));
		if(t < t_min) {
			t_min = t;
			p = ray_at(ray, t);
			material = MAT_LIGHT0 + i;
			normal = vec3(transpose(r) * vec4(normal_tmp, 0.0));
		}
	}
    
    if (!shadow) {
        // left
        {
            vec3 n = vec3(1, 0, 0);
            float t = intersect_plane(ray, vec3(-1, 0, 0), n);
            if(t < t_min) {
                vec3 p_tmp = ray_at(ray, t);
                if(all(lessThanEqual(p_tmp.yz, vec2(1))) && all(greaterThanEqual(p_tmp.yz,
                                vec2(-1))))
                {
                    normal = n;
                    p = p_tmp;

                    t_min = t;

                    material = MAT_LEFT;
                }
            }
        }
        // right
        {
            vec3 n = vec3(-1, 0, 0);
            float t = intersect_plane(ray, vec3(1, 0, 0), n);
            if(t < t_min) {
                vec3 p_tmp = ray_at(ray, t);
                if(all(lessThanEqual(p_tmp.yz, vec2(1))) && all(greaterThanEqual(p_tmp.yz,
                                vec2(-1))))
                {
                    normal = n;
                    p = p_tmp;

                    t_min = t;

                    material = MAT_RIGHT;
                }
            }
        }
        // floor
        {
            vec3 n = vec3(0, 1, 0);
            float t = intersect_plane(ray, vec3(0, -1, 0), n);
            if(t < t_min) {
                vec3 p_tmp = ray_at(ray, t);
                if(all(lessThan(p_tmp.xz, vec2(1))) && all(greaterThan(p_tmp.xz,
                                vec2(-1))))
                {
                    normal = n;
                    p = p_tmp;

                    t_min = t;
                    material = MAT_FLOOR;
                }
            }
        }
        // ceiling
        {
            vec3 n = vec3(0, -1, 0);
            float t = intersect_plane(ray, vec3(0, 1, 0), n);
            if(t < t_min) {
                vec3 p_tmp = ray_at(ray, t);
                if(all(lessThan(p_tmp.xz, vec2(1))) && all(greaterThan(p_tmp.xz,
                                vec2(-1))))
                {
                    normal = n;
                    p = p_tmp;
                    material = MAT_CEILING;

                    t_min = t;
                }
            }
        }
        // back wall
        {
            vec3 n = vec3(0, 0, 1);
            float t = intersect_plane(ray, vec3(0, 0, -1), n);
            if(t < t_min) {
                vec3 p_tmp = ray_at(ray, t);
                if(all(lessThan(p_tmp.xy, vec2(1))) && all(greaterThan(p_tmp.xy,
                                vec2(-1))))
                {
                    normal = n;
                    p = p_tmp;
                    material = MAT_BACK;

                    t_min = t;
                }
            }
        }
	}

    switch(material) {
	case MAT_LEFT   : ms = MaterialSample(vec3(0.9, 0.1, 0.1), 0.5,  false, -1); break;
	case MAT_RIGHT  : ms = MaterialSample(vec3(0.1, 0.9, 0.1), 0.5,  false, -1); break;
	case MAT_CEILING: ms = MaterialSample(vec3(0.7, 0.7, 0.7), 0.25, false, -1); break;
	case MAT_FLOOR  : ms = MaterialSample(vec3(0.7, 0.7, 0.7), 0.12, false, -1); break;
	case MAT_BACK   : ms = MaterialSample(vec3(0.7, 0.7, 0.7), 0.25, false, -1); break;
	case MAT_H      : ms = MaterialSample(vec3(1.0, 0.0, 0.0), 0.10, false, -1); break;
	case MAT_P      : ms = MaterialSample(vec3(0.0, 0.7, 0.7), 0.10, false, -1); break;
	case MAT_G      : ms = MaterialSample(vec3(0.1, 0.1, 0.7), 0.10, false, -1); break;
	case MAT_2      : ms = MaterialSample(vec3(0.8, 0.8, 0.8), 0.55, false, -1); break;
	default         : ms = MaterialSample(light_color[material - MAT_LIGHT0], 0.0, true, material - MAT_LIGHT0); break;
	}

	normal = normalize(normal);

	return t_min;
}


bool
test_visibility(vec3 p1, vec3 p2)
{
	const float eps = 1e-5;

	Ray r = Ray(p1, normalize(p2 - p1));
	r.origin += eps * r.dir;

	vec3 n, p;
	MaterialSample ms;
	float t_shadow = intersect(r, true, p, n, ms);

	return t_shadow > distance(p1, p2) - 2.0 * eps;
}

ivec3
valid_surfaces(vec3 point)
{
    if (point.x >= 0.0 && point.y >= 0.0 && point.z >= 0.0)
        return ivec3(1, 3, 5);
    if (point.x >= 0.0 && point.y >= 0.0 && point.z <= 0.0)
        return ivec3(1, 2, 5);
    if (point.x >= 0.0 && point.y <= 0.0 && point.z >= 0.0)
        return ivec3(0, 3, 5);
    if (point.x >= 0.0 && point.y <= 0.0 && point.z <= 0.0)
        return ivec3(0, 2, 5);
    if (point.x <= 0.0 && point.y >= 0.0 && point.z >= 0.0)
        return ivec3(1, 3, 4);
    if (point.x <= 0.0 && point.y >= 0.0 && point.z <= 0.0)
        return ivec3(1, 2, 4);
    if (point.x <= 0.0 && point.y <= 0.0 && point.z >= 0.0)
        return ivec3(0, 3, 4);
    if (point.x <= 0.0 && point.y <= 0.0 && point.z <= 0.0)
        return ivec3(0, 2, 4);
}

vec3
sample_light(vec4 rng, vec3 shade_normal, vec3 shade_point, out vec3 normal, out float pdf, out vec3 Le)
{   
    // Choose light based on distance i.e. closer lights are sampled with more probability
    const int num_lights = cube_light_pos.length();
	float distances_pdf[num_lights];
    float distances_cdf[num_lights];
    float distances_sum = 0.f;
    
    // Calculate distance based PDF of selecting lights
    int valid_lights[4] = int[4](-1, -1, -1, -1);
    int valid_light_cnt = 0;
    for(int i=0; i<num_lights; i++) {
        vec3 dir = cube_light_pos[i].xyz - shade_point;
        if (dot(dir, shade_normal) < 0.0) {
            continue;
        }
        valid_lights[valid_light_cnt] = i;
        float dist = length(dir);

        distances_pdf[i] = length(light_color[i]) / (dist*dist);
        distances_sum += distances_pdf[i];
        
        if(i == 0) {
            distances_cdf[i] = distances_pdf[i];
        } else {
            distances_cdf[i] = distances_cdf[i-1] + distances_pdf[i];
        }
        valid_light_cnt += 1;
    }
    
    // No lights on side of normal
    if (valid_light_cnt == 0) {
        pdf = 0.0;
        Le = vec3(0.0);
        return vec3(0.0, 0.0, 0.0);
    }
    
    int cube_idx = -1;
    for(int i=0; i<valid_light_cnt;i++) {
        // Normalize PDF
        distances_pdf[i] /= distances_sum;
        distances_cdf[i] /= distances_sum;
        if(i == 0) {
            if(rng.z <= distances_cdf[i]) {
                cube_idx = i;
                break;
            }
        }
        else {
            if(rng.z > distances_cdf[i-1] && 
                    rng.z <= distances_cdf[i]) {
                cube_idx = i;
                break;
            }
        }
    }
    
    cube_idx = valid_lights[cube_idx];
    Le = light_color[cube_idx];
    
    // Randomly choose a face on which to sample a point
    int face_idx = int(rng.w * 3.0);
    if (face_idx == 3) {
        face_idx -= 1;
    }
    
    // Choose point according to octant that shading point lies in
    mat4 rotation_mat = rotate_y(cube_light_pos[cube_idx].w);
    vec3 new_p = (rotate_y(-cube_light_pos[cube_idx].w) * vec4(shade_point - cube_light_pos[cube_idx].xyz, 1.0)).xyz;
    ivec3 val_faces = valid_surfaces(new_p);
    face_idx = val_faces[face_idx];
    
    vec3 n, p;
    switch(face_idx) {
        case 0:
            p = vec3(rng.x, 0, rng.y);
            n = vec3( 0, -1,  0); 
            break;
        case 1:
            p = vec3(rng.x, 1, rng.y);
            n = vec3( 0, 1,  0); 
            break;
        case 2:
            p = vec3(rng.x, rng.y, 0);
            n = vec3( 0, 0, -1); 
            break;
        case 3:
            p = vec3(rng.x, rng.y, 1);
            n = vec3( 0, 0, 1); 
            break;
        case 4:
            p = vec3(0, rng.x, rng.y);
            n = vec3( -1,  0,  0); 
            break;
        case 5:
            p = vec3(1, rng.x, rng.y);
            n = vec3( 1,  0,  0); 
            break;
    }
    p -= vec3(0.5);
	p = (rotation_mat * vec4(p, 1.0)).xyz;
	n = (rotation_mat * vec4(n, 0.0)).xyz;
	p *= cube_light_size;
    pdf = (1.0 / (3.0 * cube_light_size * cube_light_size)) * distances_pdf[cube_idx];
    normal = n;
    return p + cube_light_pos[cube_idx].xyz;
}

float
get_light_pdf(vec3 shade_point, int light_idx)
{
    const int num_lights = cube_light_pos.length();
	float distances_pdf[num_lights];
    float distances_sum = 0.f;
    
    for(int i=0; i<num_lights; i++) {
        float dist = length(cube_light_pos[i].xyz - shade_point);
        
        distances_pdf[i] = length(light_color[i]) / (dist*dist);
        distances_sum += distances_pdf[i];
    }
    
    return (distances_pdf[light_idx] / distances_sum) * (1.0 / (3.0 * cube_light_size * cube_light_size));
}


float
pdf_a_to_w(float pdf, float dist2, float cos_theta)
{
    float abs_cos_theta = abs(cos_theta);
    if(abs_cos_theta < 1e-8)
        return 0.0;

    return pdf * dist2 / abs_cos_theta;
}

// Adapted from https://jcgt.org/published/0007/04/01/
vec3 sampleGGXVNDF(vec3 V, float roughness, vec2 rng, mat3 onb)
{
    V = vec3(dot(V, onb[0]), dot(V, onb[2]), dot(V, onb[1]));
    
	// Section 3.2: transforming the view direction to the hemisphere configuration
	V = normalize(vec3(roughness * V.x, roughness * V.y, V.z));
    
	// Section 4.1: orthonormal basis (with special case if cross product is zero)
	float lensq = V.x * V.x + V.y * V.y;
	vec3 T1 = lensq > 0. ? vec3(-V.y, V.x, 0) * inversesqrt(lensq) : vec3(1,0,0);
	vec3 T2 = cross(V, T1);
    
	// Section 4.2: parameterization of the projected area
	float r = sqrt(rng.x);	
	float phi = 2.0 * PI * rng.y;
    // float phi = 2.0  * rng.y;
	float t1 = r * cos(phi);
	float t2 = r * sin(phi);
	float s = 0.5 * (1.0 + V.z);
	t2 = (1.0 - s)*sqrt(1.0 - t1*t1) + s*t2;
    
	// Section 4.3: reprojection onto hemisphere
	vec3 Nh = t1*T1 + t2*T2 + sqrt(max(0.0, 1.0 - t1*t1 - t2*t2))*V;
	// Section 3.4: transforming the normal back to the ellipsoid configuration
	vec3 Ne = normalize(vec3(roughness * Nh.x, max(0.0, Nh.z), roughness * Nh.y));	
	return normalize(onb * Ne);
}

float GGX_VNDF_PDF(float VdotH, float D) {
    return (VdotH > 0.0) ? D / (4.0 * VdotH) : 0.0;
}

vec3
pt_mis(Ray ray)
{
	vec3 contrib = vec3(0);
	vec3 tp = vec3(1.0);

	vec3 position, normal;
	MaterialSample ms;
	float t = intersect(ray, false, position, normal, ms);

	if(t == INFINITY)
		return vec3(0.0);

	if(ms.is_light) { /* hit light source */
		return ms.color;
	}

	for(int i = 0; i < NUM_BOUNCES; i++) {
		mat3 onb = construct_ONB_frisvad(normal);

		float NdotV = max(1e-4, dot(normal, -ray.dir));
        
		{ /* NEE */
            vec3 c = vec3(0.0);
            int calc = 0;
            int light_samples;
            if (i < 4) {
                light_samples = 2;
            } else {
                light_samples = 1;
            }
            for (int j = 0; j < light_samples; j += 1) {
                vec3 light_normal;
                float light_pdf;
                vec3 Le;
                vec3 pos_ls = sample_light(vec4(get_random(), get_random()), normal, position, light_normal, light_pdf, Le);
                if (light_pdf > EPS && test_visibility(position, pos_ls)) {
                    vec3 l_nee = pos_ls - position;
                    float rr_nee = dot(l_nee, l_nee);
                    l_nee /= sqrt(rr_nee);


                    vec3 H = normalize(-ray.dir + l_nee);
                    vec3 V = -ray.dir;
                    float NdotH = max(0.0, dot(normal, H));
                    float LdotH = max(0.0, dot(l_nee, H));
                    float NdotL = max(1e-6, dot(normal, l_nee));
                    float VdotH = max(1e-6, dot(V, H));

                    float D = ggxNormalDistribution(NdotH, ms.roughness);
                    float G = schlickMaskingTerm(NdotL, NdotV, ms.roughness);
                    vec3  F = schlickFresnel(ms.color, LdotH);

                    vec3 brdf = D * G * F / (4.0 * NdotV  * NdotL );
                    float brdf_pdf = GGX_VNDF_PDF(VdotH, D);
                    float light_pdf_w = pdf_a_to_w(light_pdf, rr_nee, -dot(l_nee, light_normal));
                    float w = 1.0 / (light_pdf_w + brdf_pdf);
                    c += tp * Le * brdf * w;
                    calc += 1;
                } 
            }
            if (calc > 0) {
                contrib += c / float(calc);
            }
		}
		
		{ /* brdf */
			// Randomly sample the NDF to get a microfacet in our BRDF
            vec3 V = -ray.dir;
			vec3 H = sampleGGXVNDF(V, ms.roughness, get_random(), onb);
			// Compute outgoing direction based on this (perfectly reflective) facet
			vec3 L = normalize(reflect(ray.dir, H));
			ray = Ray(position + L * 1e-5, L);

			vec3 position_next, normal_next;
			MaterialSample ms_next;
			float t = intersect(ray, false, position_next, normal_next, ms_next);

			if(t == INFINITY) {
				break;
            }

			// Compute some dot products needed for shading
			float  NdotL = max(1e-6, dot(normal, L));
			float  LdotH = max(1e-6, dot(L, H));
            
			// Evaluate our BRDF using a microfacet BRDF model
			vec3 F = schlickFresnel(ms.color, LdotH);                 
			        
            if(ms_next.is_light) {
                float  NdotH = max(1e-6, dot(normal, H));
                float  VdotH = max(1e-6, dot(V, H));
                float  NdotV = max(1e-6, dot(normal, V));
                
                float D = ggxNormalDistribution(NdotH, ms.roughness);          
                float G = schlickMaskingTerm(NdotL, NdotV, ms.roughness); 
                
                // What's the probability of sampling vector H from sampleGGXVNDF()?
                float brdf_pdf = GGX_VNDF_PDF(VdotH, D);
                vec3  brdf = D * G * F / (4.0 * NdotL * NdotV);
                
                float light_pdf_a = get_light_pdf(position, ms_next.light_idx);
                float light_pdf_w = pdf_a_to_w(light_pdf_a, t * t, -dot(ray.dir, normal_next));
                float w = 1.0 / (brdf_pdf + light_pdf_w);
                contrib += tp * (ms_next.color * w * brdf);
                break;
            };
			tp *= F * schlickSingleMask(NdotL, ms.roughness);

			position = position_next;
			normal = normal_next;
			ms = ms_next;
		}
        
        // Russian roulette
        if (i > 2) {
            float p = max(max(tp.x, tp.y), tp.z);
            if (get_random().x > p) {
                break;
            }
            tp *= 1.0 / p;
        }
    }

	return contrib;
}


void
mainImage(out vec4 fragColor, in vec2 fragCoord)
{
	seed = iFrame * NUM_SAMPLES;
    frag_coord = fragCoord;
	vec2 p = fragCoord.xy / vec2(iResolution) - vec2(0.5);
	float a = float(iResolution.x) / float(iResolution.y);
	if(a < 1.0)
		p.y /= a;
	else
		p.x *= a;

	//vec3 cam_center = vec3(0, 0.2, 6.0);
    vec3 cam_center = vec3(sin(iTime) * 0.25, sin(iTime * 0.7345) * 0.4 + 0.2, 6.0);
	vec3 cam_target = vec3(0, -0.1, 0);

	mat4 cam = transpose(look_at(cam_center, cam_target, vec3(0, 1, 0)));

	vec3 s = vec3(0);
    float exposure = 2.0;
    for(int i = 0; i < NUM_SAMPLES; i++) {
		Ray ray;
		ray.origin = cam_center;
		vec2 r = get_random();
		vec3 ray_dir = normalize(vec3(p + r.x * dFdx(p) + r.y * dFdy(p), -2.5));
		ray.dir = vec3(cam * vec4(ray_dir, 0.0));
		vec3 c = clamp(pt_mis(ray), 0.0, 1.0);
		s += c;
	}
    
	fragColor = vec4(pow(exposure * s / float(NUM_SAMPLES), vec3(1.0 / 2.2)), 1.0);
}
