#include <cmath>
#include <algorithm>
#include "raytracer_cpu.h"

Vector3f backgroundColor(const Ray& ray) {
    float t = 0.5f * (ray.direction.y + 1.0f);
    Vector3f startColor(1.0f, 1.0f, 1.0f);
    Vector3f endColor(0.5f, 0.7f, 1.0f);
    return startColor * (1.0f - t) + endColor * t;
}

Vector3f calculateLighting(const Scene& scene, const Intersection& intersection, const Ray& ray) {
    Vector3f specColor(0.0f, 0.0f, 0.0f);
    Vector3f diffColor(0.0f, 0.0f, 0.0f); 
    Vector3f ambColor(0.1f, 0.1f, 0.1f); 

    float distance, diff, spec;
    
    for (const auto& light : scene.lights) {
        Vector3f lightDir = (light.position - intersection.point).normalize();
        
        Ray shadowRay(intersection.point + lightDir * 0.001f, lightDir);
        Intersection shadowIntersection;
        distance = (light.position - intersection.point).length();
        
        bool inShadow = false;
        if (scene.intersect(shadowRay, shadowIntersection) && 
            shadowIntersection.distance < distance) {
            inShadow = true;
        }
        
        if (!inShadow) {
            diff = std::max(0.0f, intersection.normal.dot(lightDir));
            diffColor = diffColor + light.color * intersection.color * diff;
            Vector3f refDir = intersection.normal * -2.0f * ray.direction.dot(intersection.normal) + ray.direction;
            spec = powf(std::max(0.0f, -refDir.dot(lightDir)), 32.0f);
            specColor = specColor + light.color * spec * 0.5f;
        }
    }
    Vector3f color = ambColor * intersection.color + diffColor + specColor;
    color.x = std::min(color.x, 1.0f);
    color.y = std::min(color.y, 1.0f);
    color.z = std::min(color.z, 1.0f);
    return color;
}


Vector3f traceRay(const Scene& scene, const Ray& ray, int depth) {
    if (depth <= 0) {
        return backgroundColor(ray);
    }
    Intersection intersection;
    if (scene.intersect(ray, intersection)) {
        return calculateLighting(scene, intersection, ray);
    } else {
        return backgroundColor(ray);
    }
}

Vector3f calculateColor(const Scene& scene, const Camera& camera, 
                                int x, int y, int ssaa, int maxDepth) {
    Vector3f color(0.0f, 0.0f, 0.0f);

    for (int sy = 0; sy < ssaa; sy++) {
        for (int sx = 0; sx < ssaa; sx++) {
            float sub_X = x + (sx + 0.5f) / ssaa;
            float sub_Y = y + (sy + 0.5f) / ssaa;
        
            Ray ray = camera.generateRay(sub_X, sub_Y);
        
            color = color + traceRay(scene, ray, maxDepth);
        }
    }

    return color / (ssaa * ssaa);
}

unsigned long long rayTraceCPU(const Scene& scene, const Camera& camera, std::vector<Vector3f>& buffer, int maxDepth, int ssaa) {
    int w = camera.getWidth();
    int h = camera.getHeight();
    int x, y, id;
    unsigned long long total = 0;

    for (auto& solid : const_cast<Scene&>(scene).solids) {
        solid.buildGeometry();
    }

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            id = y * w + x;
            buffer[id] = calculateColor(scene, camera, x, y, ssaa, maxDepth);
            total += ssaa * ssaa;
        }
    }
    
    return total;
}

