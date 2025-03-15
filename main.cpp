
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>  

#include "raytracer_cpu.h"
#include "geometry.h"
#include "camera.h"

#ifdef IF_CUDA
#include "raytracer_gpu.h"
#endif

using namespace std;

void printDefaultConfig() {
    std::cout << "100\n";
    std::cout << "output_%d.data\n";
    std::cout << "800 600 90\n";
    std::cout << "7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0\n";
    std::cout << "2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n";
    std::cout << "2.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0\n";
    std::cout << "0.0 2.0 0.0 0.0 1.0 0.0 0.75 0.0 0.0 0\n";
    std::cout << "0.0 0.0 2.0 0.0 0.7 0.7 0.5 0.0 0.0 0\n";
    std::cout << "-5.0 -5.0 -1.0 -5.0 5.0 -1.0 5.0 5.0 -1.0 5.0 -5.0 -1.0 none 0.0 1.0 0.0 0.0\n";
    std::cout << "1\n";
    std::cout << "-10.0 10.0 10.0 1.0 1.0 1.0\n";
    std::cout << "1 1\n";
}

int main(int argc, char* argv[]) {
    bool useGPU = false;
#ifdef IF_CUDA
    useGPU = true;
#endif
    
    if (argc > 1) {
        if (strcmp(argv[1], "--cpu") == 0) {
            useGPU = true;
        } else if (strcmp(argv[1], "--default") == 0) {
            printDefaultConfig();
            return 0;
        } else if (strcmp(argv[1], "--gpu") != 0 && strcmp(argv[1], "") != 0) {
            std::cerr << "Unknown argument: " << argv[1] << std::endl;
            return 1;
        }
    }
    
    int frameCount;
    string outputPath;
    int width, height;
    float fov;
    
    float rc0, zc0, phic0, Acr, Acz, wcr, wcz, wcphi, pcr, pcz;
    float rn0, zn0, phin0, Anr, Anz, wnr, wnz, wnphi, pnr, pnz;
    
    vector<PlatonicSolid> solids;
    
    Vector3f floorPoints[4];
    string texturePath;
    Vector3f floorColor;
    float floorReflectivity;
    
    vector<Light> lights;
    
    int maxDepth, ssaaSqrt;
    
    cin >> frameCount;
    cin >> outputPath;
    cin >> width >> height >> fov;
    cin >> rc0 >> zc0 >> phic0 >> Acr >> Acz >> wcr >> wcz >> wcphi >> pcr >> pcz;
    cin >> rn0 >> zn0 >> phin0 >> Anr >> Anz >> wnr >> wnz >> wnphi >> pnr >> pnz;

    for (int i = 0; i < 3; i++) {
        float x, y, z;
        float r, g, b;
        float radius, reflectivity, transparency;
        int lightCount;
        
        cin >> x >> y >> z >> r >> g >> b >> radius >> reflectivity >> transparency >> lightCount;
        
        PlatonicSolid solid;
        solid.center = Vector3f(x, y, z);
        solid.color = Vector3f(r, g, b);
        solid.radius = radius;
        solid.reflectivity = reflectivity;
        solid.transparency = transparency;
        solid.lightsPerEdge = lightCount;
    
        if (i == 0) solid.type = HEXAHEDRON;
        else if (i == 1) solid.type = OCTAHEDRON;
        else solid.type = DODECAHEDRON;
        
        solids.emplace_back(solid);
    }
    
    for (int i = 0; i < 4; i++) {
        cin >> floorPoints[i].x >> floorPoints[i].y >> floorPoints[i].z;
    }
    cin >> texturePath;
    cin >> floorColor.x >> floorColor.y >> floorColor.z >> floorReflectivity;
    int lightCount;
    cin >> lightCount;
    
    for (int i = 0; i < lightCount; i++) {
        float x, y, z, r, g, b;
        cin >> x >> y >> z >> r >> g >> b;
        lights.emplace_back(Light(Vector3f(x, y, z), Vector3f(r, g, b)));
        if (i == 0) break;
    }

    cin >> maxDepth >> ssaaSqrt;

    maxDepth = 1;
    Scene scene;
    scene.solids = solids;
    scene.floor = Floor(floorPoints, floorColor, floorReflectivity);
    scene.lights = lights;

    for (int frame = 0; frame < frameCount; frame++) {
        float t = 2.0f * M_PI * frame / frameCount;
        
        float rc = rc0 + Acr * sin(wcr * t + pcr);
        float zc = zc0 + Acz * sin(wcz * t + pcz);
        float phic = phic0 + wcphi * t;

        float rn = rn0 + Anr * sin(wnr * t + pnr);
        float zn = zn0 + Anz * sin(wnz * t + pnz);
        float phin = phin0 + wnphi * t;
    
        Vector3f cameraPos(rc * cos(phic), rc * sin(phic), zc);
        Vector3f lookAt(rn * cos(phin), rn * sin(phin), zn);
        
        Camera camera(cameraPos, lookAt, Vector3f(0, 0, 1), fov, width, height);
        
        vector<Vector3f> frameBuffer(width * height);

        auto start = chrono::high_resolution_clock::now();
        unsigned long long rayCount = 0;
        
        if (!useGPU) {
            rayCount = rayTraceCPU(scene, camera, frameBuffer, maxDepth, ssaaSqrt);
        } else {
#ifdef IF_CUDA
            rayCount = rayTraceGPU(scene, camera, frameBuffer, maxDepth, ssaaSqrt);
#else
            rayCount = rayTraceCPU(scene, camera, frameBuffer, maxDepth, ssaaSqrt);
#endif
        }
        
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

        cout << frame << "\t" << duration << "\t" << rayCount << "\n";
    
        char filename[1024];
        snprintf(filename, sizeof(filename), outputPath.c_str(), frame);
    
        ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            cerr << "Error opening output file: " << filename << "\n";
            continue;
        }
        
        outFile.write(reinterpret_cast<char*>(&width), sizeof(width));
        outFile.write(reinterpret_cast<char*>(&height), sizeof(height));
        
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                unsigned char r = std::min(255, static_cast<int>(frameBuffer[idx].x * 255));
                unsigned char g = std::min(255, static_cast<int>(frameBuffer[idx].y * 255));
                unsigned char b = std::min(255, static_cast<int>(frameBuffer[idx].z * 255));
                
                outFile.write(reinterpret_cast<char*>(&r), sizeof(r));
                outFile.write(reinterpret_cast<char*>(&g), sizeof(g));
                outFile.write(reinterpret_cast<char*>(&b), sizeof(b));
            }
        }
        outFile.close();
    }
    return 0;
}
