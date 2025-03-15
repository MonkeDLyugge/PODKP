#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <cmath>
#include <vector>

struct Vector3f {
    float x, y, z;
    
    Vector3f() : x(0), y(0), z(0) {}
    Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
    
    float length() const {
        return sqrtf(x*x + y*y + z*z);
    }
    
    Vector3f normalize() const {
        float len = length();
        if (len > 0) {
            return Vector3f(x/len, y/len, z/len);
        }
        return *this;
    }
    
    Vector3f operator+(const Vector3f& v) const {
        return Vector3f(x + v.x, y + v.y, z + v.z);
    }
    
    Vector3f operator-(const Vector3f& v) const {
        return Vector3f(x - v.x, y - v.y, z - v.z);
    }
    
    Vector3f operator*(float s) const {
        return Vector3f(x * s, y * s, z * s);
    }
    
    Vector3f operator/(float s) const {
        return Vector3f(x / s, y / s, z / s);
    }
    
    float dot(const Vector3f& v) const {
        return x * v.x + y * v.y + z * v.z;
    }
    
    Vector3f cross(const Vector3f& v) const {
        return Vector3f(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }
    
    Vector3f operator*(const Vector3f& v) const {
        return Vector3f(x * v.x, y * v.y, z * v.z);
    }
};

struct Ray {
    Vector3f origin;
    Vector3f direction;
    
    Ray() {}
    Ray(const Vector3f& o, const Vector3f& d) : origin(o), direction(d.normalize()) {}
};

struct Intersection {
    bool hit;
    float distance;
    Vector3f point;
    Vector3f normal;
    Vector3f color;
    float reflectivity;
    float transparency;
    
    Intersection() : hit(false), distance(INFINITY), reflectivity(0), transparency(0) {}
};

enum PlatonicSolidType {
    HEXAHEDRON, 
    OCTAHEDRON,
    DODECAHEDRON 
};

struct Triangle {
    Vector3f v0, v1, v2;
    Vector3f normal;
    
    Triangle(const Vector3f& v0, const Vector3f& v1, const Vector3f& v2): v0(v0), v1(v1), v2(v2) {
        Vector3f edge1 = v1 - v0;
        Vector3f edge2 = v2 - v0;
        normal = edge1.cross(edge2).normalize();
    }
    
    bool intersect(const Ray& ray, Intersection& intersection) const {
        Vector3f edge1 = v1 - v0;
        Vector3f edge2 = v2 - v0;
        Vector3f h = ray.direction.cross(edge2);
        float a = edge1.dot(h);
        
        if (fabs(a) < 1e-6)
            return false;
            
        float f = 1.0f / a;
        Vector3f s = ray.origin - v0;
        float u = f * s.dot(h);
        
        if (u < 0.0f || u > 1.0f)
            return false;
            
        Vector3f q = s.cross(edge1);
        float v = f * ray.direction.dot(q);
        
        if (v < 0.0f || u + v > 1.0f)
            return false;
            
        float t = f * edge2.dot(q);
        
        if (t > 1e-6 && t < intersection.distance) {
            intersection.hit = true;
            intersection.distance = t;
            intersection.point = ray.origin + ray.direction * t;
            intersection.normal = normal;
            return true;
        }
        
        return false;
    }
};

struct PlatonicSolid {
    PlatonicSolidType type;
    Vector3f center;
    float radius;
    Vector3f color;
    float reflectivity;
    float transparency;
    int lightsPerEdge;
    std::vector<Triangle> triangles;
    
    PlatonicSolid() : 
        type(HEXAHEDRON), 
        radius(1.0f), 
        reflectivity(0), 
        transparency(0), 
        lightsPerEdge(0) {}
    
    void buildGeometry() {
        triangles.clear();
        
        switch (type) {
            case HEXAHEDRON:
                buildHexahedron();
                break;
            case OCTAHEDRON: 
                buildOctahedron();
                break;
            case DODECAHEDRON:
                buildDodecahedron();
                break;
        }
    }
    
    void buildHexahedron() {
        float r = radius / sqrtf(3.0f);

        Vector3f vertices[8] = {
            Vector3f(center.x - r, center.y - r, center.z - r),
            Vector3f(center.x + r, center.y - r, center.z - r),
            Vector3f(center.x + r, center.y + r, center.z - r),
            Vector3f(center.x - r, center.y + r, center.z - r),
            Vector3f(center.x - r, center.y - r, center.z + r),
            Vector3f(center.x + r, center.y - r, center.z + r),
            Vector3f(center.x + r, center.y + r, center.z + r),
            Vector3f(center.x - r, center.y + r, center.z + r)
        };
        
        triangles.push_back(Triangle(vertices[0], vertices[1], vertices[2]));
        triangles.push_back(Triangle(vertices[0], vertices[2], vertices[3]));

        triangles.push_back(Triangle(vertices[4], vertices[7], vertices[6]));
        triangles.push_back(Triangle(vertices[4], vertices[6], vertices[5]));

        triangles.push_back(Triangle(vertices[0], vertices[4], vertices[5]));
        triangles.push_back(Triangle(vertices[0], vertices[5], vertices[1]));
        
        triangles.push_back(Triangle(vertices[1], vertices[5], vertices[6]));
        triangles.push_back(Triangle(vertices[1], vertices[6], vertices[2]));
        
        triangles.push_back(Triangle(vertices[2], vertices[6], vertices[7]));
        triangles.push_back(Triangle(vertices[2], vertices[7], vertices[3]));
        
        triangles.push_back(Triangle(vertices[3], vertices[7], vertices[4]));
        triangles.push_back(Triangle(vertices[3], vertices[4], vertices[0]));
    }
    
    void buildOctahedron() {
        float r = radius;
    
        Vector3f vertices[6] = {
            Vector3f(center.x + r, center.y, center.z),
            Vector3f(center.x - r, center.y, center.z),
            Vector3f(center.x, center.y + r, center.z),
            Vector3f(center.x, center.y - r, center.z),
            Vector3f(center.x, center.y, center.z + r),
            Vector3f(center.x, center.y, center.z - r)
        };
        
        triangles.push_back(Triangle(vertices[0], vertices[2], vertices[4]));
        triangles.push_back(Triangle(vertices[0], vertices[4], vertices[3]));
        triangles.push_back(Triangle(vertices[0], vertices[3], vertices[5]));
        triangles.push_back(Triangle(vertices[0], vertices[5], vertices[2]));
        
        triangles.push_back(Triangle(vertices[1], vertices[4], vertices[2]));
        triangles.push_back(Triangle(vertices[1], vertices[3], vertices[4]));
        triangles.push_back(Triangle(vertices[1], vertices[5], vertices[3]));
        triangles.push_back(Triangle(vertices[1], vertices[2], vertices[5]));
    }
    
    void buildDodecahedron() {
        float goldenRatio = (1.0f + sqrtf(5.0f)) / 2.0f;
        float reverseRatio = 1.0f / goldenRatio;
        
        std::vector<Vector3f> vertices = {
            Vector3f(-reverseRatio, 0.0f, goldenRatio),
            Vector3f(reverseRatio, 0.0f, goldenRatio),
            Vector3f(-1.0f, 0.0f, 1.0f),
            Vector3f(1.0f, 1.0f, 1.0f),
            Vector3f(1.0f, -1.0f, 1.0f),
            Vector3f(-1.0f, -1.0f, 1.0f),
            Vector3f(0.0f, -goldenRatio, reverseRatio),
            Vector3f(0.0f, goldenRatio, reverseRatio),
            Vector3f(-goldenRatio, -reverseRatio, 0.0f),
            Vector3f(-goldenRatio, reverseRatio, 0.0f),
            Vector3f(goldenRatio, reverseRatio, 0.0f),
            Vector3f(goldenRatio, -reverseRatio, 0.0f),
            Vector3f(0.0f, -goldenRatio, -reverseRatio),
            Vector3f(0.0f, goldenRatio, -reverseRatio),
            Vector3f(1.0f, 1.0f, -1.0f),
            Vector3f(1.0f, -1.0f, -1.0f),
            Vector3f(-1.0f, -1.0f, -1.0f),
            Vector3f(-1.0f, 1.0f, -1.0f),
            Vector3f(reverseRatio, 0.0f, -goldenRatio),
            Vector3f(-reverseRatio, 0.0f, -goldenRatio),
        };

        for (auto& vertex: vertices) {
            vertex.x = vertex.x * radius / sqrt(3.0) + center.x;
            vertex.y = vertex.y * radius / sqrt(3.0) + center.y;
            vertex.z = vertex.z * radius / sqrt(3.0) + center.z;
        }
        
        triangles.push_back(Triangle(vertices[4], vertices[0], vertices[6]));
        triangles.push_back(Triangle(vertices[0], vertices[5], vertices[6]));
        triangles.push_back(Triangle(vertices[0], vertices[4], vertices[1]));
        triangles.push_back(Triangle(vertices[0], vertices[3], vertices[7]));
        triangles.push_back(Triangle(vertices[2], vertices[0], vertices[7]));
        triangles.push_back(Triangle(vertices[0], vertices[1], vertices[3]));
        
        triangles.push_back(Triangle(vertices[10], vertices[1], vertices[11]));
        triangles.push_back(Triangle(vertices[3], vertices[1], vertices[10]));
        triangles.push_back(Triangle(vertices[1], vertices[4], vertices[11]));
        triangles.push_back(Triangle(vertices[5], vertices[0], vertices[8]));
        triangles.push_back(Triangle(vertices[0], vertices[2], vertices[9]));
        triangles.push_back(Triangle(vertices[8], vertices[0], vertices[9]));
    
        triangles.push_back(Triangle(vertices[5], vertices[8], vertices[16]));
        triangles.push_back(Triangle(vertices[6], vertices[5], vertices[12]));
        triangles.push_back(Triangle(vertices[12], vertices[5], vertices[16]));
        triangles.push_back(Triangle(vertices[4], vertices[12], vertices[15]));
        triangles.push_back(Triangle(vertices[4], vertices[6], vertices[12]));
        triangles.push_back(Triangle(vertices[11], vertices[4], vertices[5]));
    
        triangles.push_back(Triangle(vertices[2], vertices[13], vertices[17]));
        triangles.push_back(Triangle(vertices[2], vertices[7], vertices[13]));
        triangles.push_back(Triangle(vertices[9], vertices[2], vertices[17]));
        triangles.push_back(Triangle(vertices[13], vertices[3], vertices[14]));
        triangles.push_back(Triangle(vertices[7], vertices[3], vertices[13]));
        triangles.push_back(Triangle(vertices[3], vertices[10], vertices[14]));

        triangles.push_back(Triangle(vertices[8], vertices[17], vertices[19]));
        triangles.push_back(Triangle(vertices[16], vertices[8], vertices[19]));
        triangles.push_back(Triangle(vertices[8], vertices[9], vertices[17]));
        triangles.push_back(Triangle(vertices[14], vertices[11], vertices[18]));
        triangles.push_back(Triangle(vertices[11], vertices[15], vertices[18]));
        triangles.push_back(Triangle(vertices[10], vertices[11], vertices[14]));

        triangles.push_back(Triangle(vertices[12], vertices[19], vertices[18]));
        triangles.push_back(Triangle(vertices[15], vertices[12], vertices[18]));
        triangles.push_back(Triangle(vertices[12], vertices[16], vertices[19]));
        triangles.push_back(Triangle(vertices[19], vertices[13], vertices[18]));
        triangles.push_back(Triangle(vertices[17], vertices[13], vertices[19]));
        triangles.push_back(Triangle(vertices[13], vertices[14], vertices[18]));
    }
    
    bool intersect(const Ray& ray, Intersection& intersection) const {
        bool hitAnything = false;
        
        for (const auto& triangle : triangles) {
            if (triangle.intersect(ray, intersection)) {
                hitAnything = true;
                intersection.color = color;
                intersection.reflectivity = reflectivity;
                intersection.transparency = transparency;
            }
        }
        
        return hitAnything;
    }
};

struct Light {
    Vector3f position;
    Vector3f color;
    
    Light(const Vector3f& pos, const Vector3f& col) : position(pos), color(col) {}
};

struct Floor {
    Vector3f vertices[4];
    Vector3f color;
    float reflectivity;
    
    Floor() : reflectivity(0) {}
    
    Floor(const Vector3f points[4], const Vector3f& col, float refl) : 
        color(col), reflectivity(refl) {
        for (int i = 0; i < 4; i++) {
            vertices[i] = points[i];
        }
    }
    
    bool intersect(const Ray& ray, Intersection& intersection) const {
        Triangle tri1(vertices[0], vertices[1], vertices[2]);
        Triangle tri2(vertices[0], vertices[2], vertices[3]);
        
        bool hit = tri1.intersect(ray, intersection) || tri2.intersect(ray, intersection);
        
        if (hit) {
            intersection.color = color;
            intersection.reflectivity = reflectivity;
        }
        
        return hit;
    }
};

struct Scene {
    std::vector<PlatonicSolid> solids;
    Floor floor;
    std::vector<Light> lights;
    
    Scene() {}

    bool intersect(const Ray& ray, Intersection& intersection) const {
        bool hitAnything = false;

        for (const auto& solid : solids) {
            if (solid.intersect(ray, intersection)) {
                hitAnything = true;
            }
        }
    
        if (floor.intersect(ray, intersection)) {
            hitAnything = true;
        }
        
        return hitAnything;
    }

    bool isInShadow(const Vector3f& point, const Light& light) const {
        Vector3f direction = (light.position - point).normalize();
        float distance = (light.position - point).length();
        
        Ray shadowRay(point + direction * 0.001f, direction);
        Intersection shadowIntersection;
        
        if (intersect(shadowRay, shadowIntersection) && shadowIntersection.distance < distance) {
            return true;
        }
        
        return false;
    }
};

#endif GEOMETRY_H