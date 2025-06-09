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
#define IMG_NAME "cat.png"

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
	string image_name, float p, float q
) {
	std::cout << "Reading image data...\n";
	PNG img(image_name);
	unsigned long long width = img.w;
	unsigned long long height = img.h;
	const cl::ImageFormat format(CL_RGBA, CL_UNSIGNED_INT8);
	cl::Image2D in(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, width, height, 0, &img.data[0]);
	img.Free();
	cl::Image2D out(context, CL_MEM_WRITE_ONLY, format, width, height, 0, NULL);

	std::cout << "Editing...\n";
	kernel.setArg(0, in);
	kernel.setArg(1, out);
	kernel.setArg(2, p);
	kernel.setArg(3, q);
	queue.enqueueNDRangeKernel(kernel, NullRange, cl::NDRange(width, height), cl::NullRange);
	queue.finish();

	cl::detail::size_t_array origin {0,0,0};
	cl::detail::size_t_array size {width, height, 1};
	PNG outPng;
	outPng.Create(width, height);
	auto tmp = new unsigned char[width * height * 4];

	queue.enqueueReadImage(out, CL_TRUE, origin, size, 0, 0, tmp, NULL, NULL);

	std::copy(&tmp[0], &tmp[width * height * 4], std::back_inserter(outPng.data));
	outPng.Save("res.png");
	outPng.Free();
	delete[] tmp;
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
	render_image(IMG_NAME, p, q);
	std::cout << "Image edited!\n";
	end = std::chrono::system_clock::now();
	elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Elapsed time = " << elapsed << "ms\n";
}

