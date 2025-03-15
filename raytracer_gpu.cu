
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <cmath>
#include <vector>

#include "camera.h"
#include "geometry.h"
#include "raytracer_gpu.h"

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(0); \
    } \
} while(0)

struct CudaVector3f {
    float x, y, z;
    
    __host__ __device__ CudaVector3f() : x(0), y(0), z(0) {}
    __host__ __device__ CudaVector3f(float x, float y, float z) : x(x), y(y), z(z) {}
    
    __device__ CudaVector3f operator+(const CudaVector3f& v) const {
        return CudaVector3f(x + v.x, y + v.y, z + v.z);
    }
    
    __device__ CudaVector3f operator-(const CudaVector3f& v) const {
        return CudaVector3f(x - v.x, y - v.y, z - v.z);
    }
    
    __device__ CudaVector3f operator*(float s) const {
        return CudaVector3f(x * s, y * s, z * s);
    }
    
    __device__ CudaVector3f operator/(float s) const {
        return CudaVector3f(x / s, y / s, z / s);
    }
    
    __device__ CudaVector3f operator*(const CudaVector3f& v) const {
        return CudaVector3f(x * v.x, y * v.y, z * v.z);
    }
    
    __device__ float dot(const CudaVector3f& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    __device__ CudaVector3f cross(const CudaVector3f& v) const {
        return CudaVector3f(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    
    __device__ float length() const {
        return sqrtf(x*x + y*y + z*z);
    }
    
    __device__ CudaVector3f normalize() const {
        float len = length();
        if (len > 0) {
            return CudaVector3f(x/len, y/len, z/len);
        }
        return *this;
    }
};

struct CudaRay {
    CudaVector3f origin;
    CudaVector3f direction;
    
    __device__ CudaRay() {}
    __device__ CudaRay(const CudaVector3f& o, const CudaVector3f& d) 
        : origin(o), direction(d.normalize()) {}
};

struct CudaIntersection {
    bool hit;
    float distance;
    CudaVector3f point;
    CudaVector3f normal;
    CudaVector3f color;
    
    __device__ CudaIntersection() : hit(false), distance(INFINITY) {}
};

struct CudaTriangle {
    CudaVector3f v0, v1, v2;
    CudaVector3f normal;
    
    __device__ bool intersect(const CudaRay& ray, CudaIntersection& intersection) const {
        CudaVector3f edge1 = v1 - v0;
        CudaVector3f edge2 = v2 - v0;
        CudaVector3f h = ray.direction.cross(edge2);
        float a = edge1.dot(h);
        
        if (fabsf(a) < 1e-6f)
            return false;
            
        float f = 1.0f / a;
        CudaVector3f s = ray.origin - v0;
        float u = f * s.dot(h);
        
        if (u < 0.0f || u > 1.0f)
            return false;
            
        CudaVector3f q = s.cross(edge1);
        float v = f * ray.direction.dot(q);
        
        if (v < 0.0f || u + v > 1.0f)
            return false;
            
        float t = f * edge2.dot(q);
        
        if (t > 1e-6f && t < intersection.distance) {
            intersection.hit = true;
            intersection.distance = t;
            intersection.point = ray.origin + ray.direction * t;
            intersection.normal = normal;
            return true;
        }
        
        return false;
    }
};

struct CudaPlatonicSolid {
    CudaVector3f center;
    CudaVector3f color;
    CudaTriangle* triangles;
    int triangleCount;
    
    __device__ bool intersect(const CudaRay& ray, CudaIntersection& intersection) const {
        bool hitAnything = false;
        
        for (int i = 0; i < triangleCount; i++) {
            if (triangles[i].intersect(ray, intersection)) {
                hitAnything = true;
                intersection.color = color;
            }
        }
        
        return hitAnything;
    }
};

struct CudaLight {
    CudaVector3f position;
    CudaVector3f color;
};

struct CudaFloor {
    CudaTriangle triangles[2];
    CudaVector3f color;
    
    __device__ bool intersect(const CudaRay& ray, CudaIntersection& intersection) const {
        bool hit = triangles[0].intersect(ray, intersection) || 
                   triangles[1].intersect(ray, intersection);
        
        if (hit) {
            intersection.color = color;
        }
        
        return hit;
    }
};

struct CudaScene {
    CudaPlatonicSolid* solids;
    int solidCount;
    CudaFloor floor;
    CudaLight* lights;
    int lightCount;
    
    __device__ bool intersect(const CudaRay& ray, CudaIntersection& intersection) const {
        bool hitAnything = false;
        
        for (int i = 0; i < solidCount; i++) {
            if (solids[i].intersect(ray, intersection)) {
                hitAnything = true;
            }
        }
        
        if (floor.intersect(ray, intersection)) {
            hitAnything = true;
        }
        
        return hitAnything;
    }
    
    __device__ bool isInShadow(const CudaVector3f& point, const CudaLight& light) const {
        CudaVector3f direction = (light.position - point).normalize();
        float distance = (light.position - point).length();
        
        CudaRay shadowRay(point + direction * 0.001f, direction);
        CudaIntersection shadowIntersection;
        
        if (intersect(shadowRay, shadowIntersection) && shadowIntersection.distance < distance) {
            return true;
        }
        
        return false;
    }
};


__device__ CudaRay generateRay(float x, float y, int width, int height, float aspectRatio, 
                              float fov, CudaVector3f position, CudaVector3f forward, 
                              CudaVector3f right, CudaVector3f up) {

    float ndcX = (2.0f * x / width - 1.0f) * aspectRatio;
    float ndcY = 1.0f - 2.0f * y / height;
    
    float tanHalfFov = tanf((fov * 0.5f) * (3.14159265358979323846f / 180.0f));
    ndcX *= tanHalfFov;
    ndcY *= tanHalfFov;
    
    CudaVector3f direction = forward + right * ndcX + up * ndcY;
    
    return CudaRay(position, direction.normalize());
}


__device__ CudaVector3f calculateLighting(const CudaScene& scene, const CudaIntersection& intersection, const CudaRay& ray) {
    float diffIntencity, specIntencity;
    CudaVector3f specColor(0.0f, 0.0f, 0.0f);
    CudaVector3f diffColor(0.0f, 0.0f, 0.0f);
    CudaVector3f ambColor(0.1f, 0.1f, 0.1f);
    
    for (int i = 0; i < scene.lightCount; i++) {
        const CudaLight& light = scene.lights[i];
        
        if (scene.isInShadow(intersection.point, light)) {
            continue;
        }

        CudaVector3f refDir = ray.direction + intersection.normal * -2.0f * ray.direction.dot(intersection.normal);
        
        specIntencity = powf(fmaxf(0.0f, -refDir.dot(lightDir)), 32.0f);
        specColor = specColor + light.color * specIntencity * 0.5f;
        
        CudaVector3f lightDir = (light.position - intersection.point).normalize();
        
        diffIntencity = fmaxf(0.0f, intersection.normal.dot(lightDir));
        diffColor = diffColor + light.color * intersection.color * diffIntencity;
    }
    
    CudaVector3f color = ambColor * intersection.color + diffColor + specColor;
    
    color.x = fminf(color.x, 1.0f);
    color.y = fminf(color.y, 1.0f);
    color.z = fminf(color.z, 1.0f);
    
    return color;
}


__device__ CudaVector3f backgroundColor(const CudaRay& ray) {
    float t = (ray.direction.y + 1.0f) * 0.5f;
    CudaVector3f colorA(1.0f, 1.0f, 1.0f);
    CudaVector3f colorB(0.4f, 0.6f, 0.9f);
    return colorB * t + colorA * (1.0f - t);
}


__device__ CudaVector3f traceRay(const CudaScene& scene, const CudaRay& ray) {
    CudaIntersection intersection;
    if (scene.intersect(ray, intersection)) {
        return calculateLighting(scene, intersection, ray);
    } else {
        return backgroundColor(ray);
    }
}


__global__ void rayTraceKernel(CudaVector3f* buffer, int width, int height, float aspectRatio, 
                               float fov, CudaVector3f cameraPos, CudaVector3f cameraForward, 
                               CudaVector3f cameraRight, CudaVector3f cameraUp, 
                               CudaScene scene, int ssaa) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        CudaVector3f pixelColor(0.0f, 0.0f, 0.0f);
        
        for (int sy = 0; sy < ssaa; sy++) {
            for (int sx = 0; sx < ssaa; sx++) {
                float subpixelX = x + (sx + 0.5f) / ssaa;
                float subpixelY = y + (sy + 0.5f) / ssaa;

                CudaRay ray = generateRay(subpixelX, subpixelY, width, height, aspectRatio, 
                                         fov, cameraPos, cameraForward, cameraRight, cameraUp);
                
                pixelColor = pixelColor + traceRay(scene, ray);
            }
        }
    
        pixelColor = pixelColor / (ssaa * ssaa);
    
        int idx = y * width + x;
        buffer[idx] = pixelColor;
    }
}


void prepareCudaScene(const Scene& cpuScene, CudaScene& cudaScene, CudaPlatonicSolid*& dev_solids, 
                     CudaTriangle*& dev_triangles, CudaLight*& dev_lights) {
    std::vector<CudaPlatonicSolid> host_solids(cpuScene.solids.size());
    
    int totalTriangles = 0;
    for (const auto& solid : cpuScene.solids) {
        totalTriangles += solid.triangles.size();
    }

    std::vector<CudaTriangle> host_triangles(totalTriangles);
    
    int offset = 0;
    for (size_t i = 0; i < cpuScene.solids.size(); i++) {
        const PlatonicSolid& cpuSolid = cpuScene.solids[i];
        CudaPlatonicSolid& cudaSolid = host_solids[i];
        
        cudaSolid.center = CudaVector3f(cpuSolid.center.x, cpuSolid.center.y, cpuSolid.center.z);
        cudaSolid.color = CudaVector3f(cpuSolid.color.x, cpuSolid.color.y, cpuSolid.color.z);
        cudaSolid.triangleCount = cpuSolid.triangles.size();
    
        for (size_t j = 0; j < cpuSolid.triangles.size(); j++) {
            const Triangle& cpuTriangle = cpuSolid.triangles[j];
            CudaTriangle& cudaTriangle = host_triangles[offset + j];

            cudaTriangle.v0 = CudaVector3f(cpuTriangle.v0.x, cpuTriangle.v0.y, cpuTriangle.v0.z);
            cudaTriangle.v1 = CudaVector3f(cpuTriangle.v1.x, cpuTriangle.v1.y, cpuTriangle.v1.z);
            cudaTriangle.v2 = CudaVector3f(cpuTriangle.v2.x, cpuTriangle.v2.y, cpuTriangle.v2.z);
            

            cudaTriangle.normal = CudaVector3f(cpuTriangle.normal.x, cpuTriangle.normal.y, cpuTriangle.normal.z);
        }
        
        offset += cpuSolid.triangles.size();
    }

    CudaFloor host_floor;
    Triangle floorTri1(cpuScene.floor.vertices[0], cpuScene.floor.vertices[1], cpuScene.floor.vertices[2]);
    Triangle floorTri2(cpuScene.floor.vertices[0], cpuScene.floor.vertices[2], cpuScene.floor.vertices[3]);
    
    host_floor.triangles[0].v0 = CudaVector3f(floorTri1.v0.x, floorTri1.v0.y, floorTri1.v0.z);
    host_floor.triangles[0].v1 = CudaVector3f(floorTri1.v1.x, floorTri1.v1.y, floorTri1.v1.z);
    host_floor.triangles[0].v2 = CudaVector3f(floorTri1.v2.x, floorTri1.v2.y, floorTri1.v2.z);
    host_floor.triangles[0].normal = CudaVector3f(floorTri1.normal.x, floorTri1.normal.y, floorTri1.normal.z);
    
    host_floor.triangles[1].v0 = CudaVector3f(floorTri2.v0.x, floorTri2.v0.y, floorTri2.v0.z);
    host_floor.triangles[1].v1 = CudaVector3f(floorTri2.v1.x, floorTri2.v1.y, floorTri2.v1.z);
    host_floor.triangles[1].v2 = CudaVector3f(floorTri2.v2.x, floorTri2.v2.y, floorTri2.v2.z);
    host_floor.triangles[1].normal = CudaVector3f(floorTri2.normal.x, floorTri2.normal.y, floorTri2.normal.z);
    
    host_floor.color = CudaVector3f(cpuScene.floor.color.x, cpuScene.floor.color.y, cpuScene.floor.color.z);

    std::vector<gpuLight> host_lights(cpuScene.lights.size());
    
    for (size_t i = 0; i < cpuScene.lights.size(); i++) {
        const Light& cpuLight = cpuScene.lights[i];
        CudaLight& gpuLight = host_lights[i];
        
        gpuLight.position = CudaVector3f(cpuLight.position.x, cpuLight.position.y, cpuLight.position.z);
        gpuLight.color = CudaVector3f(cpuLight.color.x, cpuLight.color.y, cpuLight.color.z);
    }

    CHECK_CUDA_ERROR(cudaMalloc(&dev_triangles, totalTriangles * sizeof(CudaTriangle)));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_triangles, host_triangles.data(), totalTriangles * sizeof(CudaTriangle), cudaMemcpyHostToDevice));

    offset = 0;
    for (size_t i = 0; i < host_solids.size(); i++) {
        host_solids[i].triangles = dev_triangles + offset;
        offset += host_solids[i].triangleCount;
    }

    CHECK_CUDA_ERROR(cudaMalloc(&dev_solids, host_solids.size() * sizeof(CudaPlatonicSolid)));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_solids, host_solids.data(), host_solids.size() * sizeof(CudaPlatonicSolid), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&dev_lights, host_lights.size() * sizeof(CudaLight)));
    CHECK_CUDA_ERROR(cudaMemcpy(dev_lights, host_lights.data(), host_lights.size() * sizeof(CudaLight), cudaMemcpyHostToDevice));

    cudaScene.solids = dev_solids;
    cudaScene.solidCount = host_solids.size();
    cudaScene.lights = dev_lights;
    cudaScene.lightCount = host_lights.size();
    cudaScene.floor = host_floor;
}

unsigned long long rayTraceGPU(const Scene& scene, const Camera& camera, 
                             std::vector<Vector3f>& buffer, 
                             int maxDepth, int ssaa) {
    int width = camera.getWidth();
    int height = camera.getHeight();
    float aspectRatio = (float)width / height;

    CudaVector3f cameraPos(camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
    CudaVector3f cameraForward(camera.getForward().x, camera.getForward().y, camera.getForward().z);
    CudaVector3f cameraRight(camera.getRight().x, camera.getRight().y, camera.getRight().z);
    CudaVector3f cameraUp(camera.getUp().x, camera.getUp().y, camera.getUp().z);
    float fov = camera.getFov();

    CudaVector3f* dev_buffer;
    CHECK_CUDA_ERROR(cudaMalloc(&dev_buffer, buffer.size() * sizeof(CudaVector3f)));

    CudaScene cudaScene;
    CudaPlatonicSolid* dev_solids;
    CudaTriangle* dev_triangles;
    CudaLight* dev_lights;
    
    prepareCudaScene(scene, cudaScene, dev_solids, dev_triangles, dev_lights);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    rayTraceKernel<<<gridSize, blockSize>>>(dev_buffer, width, height, aspectRatio, 
                                            fov, cameraPos, cameraForward, cameraRight, cameraUp, 
                                            cudaScene, ssaa);

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    std::vector<CudaVector3f> cudaBuffer(buffer.size());
    CHECK_CUDA_ERROR(cudaMemcpy(cudaBuffer.data(), dev_buffer, buffer.size() * sizeof(CudaVector3f), 
                          cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = Vector3f(cudaBuffer[i].x, cudabuffer[i].y, cudaBuffer[i].z);
    }

    CHECK_CUDA_ERROR(cudaFree(dev_buffer));
    CHECK_CUDA_ERROR(cudaFree(dev_solids));
    CHECK_CUDA_ERROR(cudaFree(dev_triangles));
    CHECK_CUDA_ERROR(cudaFree(dev_lights));

    return (unsigned long long) (ssaa * ssaa * width * height);
}

