#ifndef RAYTRACER_GPU_H
#define RAYTRACER_GPU_H

#include <vector>
#include "camera.h"
#include "geometry.h"

// Основная функция трассировки лучей на GPU
unsigned long long rayTraceGPU(const Scene& scene, const Camera& camera, 
                              std::vector<Vector3f>& buffer, 
                              int maxDepth, int ssaa);

#endif // RAYTRACER_GPU_H
