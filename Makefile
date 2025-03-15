CXX = /usr/bin/g++
NVCC = /usr/bin/nvcc
CXXFLAGS = -std=c++11 -O3 -DUSE_CUDA -I/usr/local/cuda/include
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart
SOURCES = main.cpp raytracer_cpu.cpp
CU_SOURCE = raytracer_gpu.cu
OBJECTS = main.o raytracer_cpu.o cuda_obj.o
BIN = KURS_PROJECT

$(BIN): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

main.o: main.cpp camera.h geometry.h raytracer_cpu.h raytracer_gpu.h
	$(CXX) $(CXXFLAGS) -c $<

raytracer.o: raytracer_cpu.cpp raytracer_cpu.h geometry.h camera.h
	$(CXX) $(CXXFLAGS) -c $<

cuda_obj.o: raytracer_gpu.cu raytracer_gpu.h geometry.h camera.h
	$(NVCC) -std=c++11 -O3 -c $< -o $@

clean:
	rm -f $(BIN) $(OBJECTS)

.PHONY: all clean
