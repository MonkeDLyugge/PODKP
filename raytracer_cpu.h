#ifndef RAYTRACER_CPU_H
#define RAYTRACER_CPU_H

#include <vector>
#include "camera.h"
#include "geometry.h"

// Основная функция трассировки лучей на CPU
unsigned long long rayTraceCPU(const Scene& scene, const Camera& camera, 
                              std::vector<Vector3f>& buffer, 
                              int maxDepth, int ssaa);

#endif // RAYTRACER_H