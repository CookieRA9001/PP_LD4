#define CL_HPP_TARGET_OPENCL_VERSION 300
// g++ -std=c++11 -I./include -L./lib -lOpenCL -o LD2 PP_LD2.cpp
// qsub -l nodes=1:ppn=1:gpus=1 -I

#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#ifdef __linux__ 
#include <CL/cl.hpp>
#else
#include <CL/opencl.hpp>
#endif
#pragma pack(1)

#include "PNG.h"

using namespace cl;

CommandQueue queue;
Kernel kernel;
Context context;
Program program;
cl_float4* cpu_output;
Buffer cl_output;

// CONFIGURATION
#define IMG_WIDTH 1920
#define IMG_HEIGHT 1080
#define LOCAL_WORK_SIZE 64

int width, height;
string image_name = "image.ppm";

void setup_opencl() {
	std::cout << "Setting up OpenCL...\n";
	std::vector<cl::Platform> platforms;
	Platform::get(&platforms);
	auto platform = platforms[0];
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
	auto device = devices[0];
	std::ifstream renderer("renderer.cl");
	std::string src(std::istreambuf_iterator<char>(renderer), (std::istreambuf_iterator<char>()));
	Program::Sources sources(1, src.c_str());
	context = Context(devices);
	program = Program(context, sources);
	queue = CommandQueue(context, device);
	auto err = program.build();
	kernel = Kernel(program, "render_kernel");

	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char* log = (char*)malloc(log_size);
		clGetProgramBuildInfo(program(), devices[0](), CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
	}
}

void render_image(
	float p, float q
) {
	std::cout << "Editing...\n";
	cpu_output = new cl_float3[IMG_WIDTH * IMG_HEIGHT];
	kernel.setArg(0, cl_output);
	kernel.setArg(1, p);
	kernel.setArg(2, q);
	queue.enqueueNDRangeKernel(kernel, NullRange, IMG_WIDTH * IMG_HEIGHT, LOCAL_WORK_SIZE);
	queue.enqueueReadBuffer(cl_output, CL_TRUE, 0, IMG_WIDTH * IMG_HEIGHT * sizeof(cl_float3), cpu_output);
	queue.finish();
}

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline int toInt(float x) { return int(clamp(x) * 255 + .5); }
void saveImage(string name) { // save image to file .PPM
	std::cout << "Saving image...\n";
	name = name + ".ppm";
	FILE* f = fopen(name.c_str(), "w");
	if (f == 0) {
		perror("Failed to open file");
		return;
	}
	fprintf(f, "P3\n%d %d\n%d\n", IMG_WIDTH, IMG_HEIGHT, 255);

	for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++) {
		fprintf(f, "%d %d %d ", toInt(cpu_output[i].s[0]), toInt(cpu_output[i].s[1]), toInt(cpu_output[i].s[2]));
	}
}

int main() {
#if CL_DEVICE_IMAGE_SUPPORT == CL_FALSE 
	cout << "Images not supported on this OpenCL version.\n";
	return 0;
#endif

	std::cout << "Starting...\n";

	long long elapsed = 0;
	std::chrono::system_clock::time_point start, end;
	setup_opencl();

	std::cout << "Reading image data...\n";


	float p = 0.0f, q = 0.0f;
	bool loop = true;

	while(loop) {
		try {
			std::cout << "Input p and q: ";
			std::cin >> p >> q;
			loop = false;
		}
		catch (const std::exception&) {
			std::cout << "Somethings wrong with the inputs. Please try again.\n";
			loop = true;
		}
	}

	start = std::chrono::system_clock::now();
	render_image(p, q);
	std::cout << "Image edited!\n";
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time = " << elapsed << "ms\n";
	saveImage("res_image");
}

