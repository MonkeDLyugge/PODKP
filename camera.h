#ifndef CAMERA_H
#define CAMERA_H

#include "geometry.h"

class Camera {
private:
    Vector3f pos;
    Vector3f Z;
    Vector3f Y;    
    Vector3f X; 
    float fov;  
    int width, height;
    float aspectRatio;

public:
    Camera() : pos(0, 0, 0), Y(0, 0, 1), Z(0, 1, 0), X(1, 0, 0), 
               fov(90.0f), width(800), height(600), aspectRatio(4.0f/3.0f) {
    }

    Camera(const Vector3f& pos, const Vector3f& lookAt, const Vector3f& ZVec, 
           float fovDegrees, int w, int h) : 
           pos(pos), width(w), height(h), fov(fovDegrees) {
        
        Y = (lookAt - pos).normalize();
        X = Y.cross(ZVec).normalize();
        Z = X.cross(Y).normalize();
        aspectRatio = (float)width / (float)height;
    }

    Ray generateRay(float x, float y) const {
        float ndcX = (2.0f * x / width - 1.0f) * aspectRatio;
        float ndcY = 1.0f - 2.0f * y / height;
        
        float tanHalfFov = tanf((fov * 0.5f) * (M_PI / 180.0f));
        ndcX *= tanHalfFov;
        ndcY *= tanHalfFov;
        Vector3f direction = Y + X * ndcX + Z * ndcY;
        
        return Ray(pos, direction.normalize());
    }

    Vector3f getPos() const { return pos; }
    Vector3f getY() const { return Y; }
    Vector3f getZ() const { return Z; }
    Vector3f getX() const { return X; }
    float getFov() const { return fov; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
};

#endif // CAMERA_H
