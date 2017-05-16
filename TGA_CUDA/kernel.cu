
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
using namespace cv;

#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define pb(bte){printf("%d\n",(bte));}
#define CUDA_ERROR_CHECK
inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// More careful checking. However, this will affect performance.
	// Comment away if needed.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}
__global__ void sobel(unsigned char* imgray, unsigned char* out, int SIZE)
{

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	unsigned char pixel00 = imgray[(x - 1) * SIZE + y - 1];
	unsigned char pixel01 = imgray[(x - 1) * SIZE + y];
	unsigned char pixel02 = imgray[(x - 1) * SIZE + y + 1];
	unsigned char pixel10 = imgray[(x) *     SIZE + y - 1];
	unsigned char pixel12 = imgray[(x) *     SIZE + y + 1];
	unsigned char pixel20 = imgray[(x + 1) * SIZE + y - 1];
	unsigned char pixel21 = imgray[(x + 1) * SIZE + y];
	unsigned char pixel22 = imgray[(x + 1) * SIZE + y + 1];
	int vert = (pixel00 + 2 * pixel01 + pixel02) - (pixel20 + 2 * pixel21 + pixel22);
	int hori = (pixel00 + 2 * pixel10 + pixel20) - (pixel02 + 2 * pixel12 + pixel22);
	int tot = vert + hori;
	tot = (tot>60) ? 255 : 0;
	out[x * SIZE + y] = tot;
	
}


void CPUSobel(unsigned char* imgray, unsigned char* out, int SIZE)
{
	for (int x = 1; x<512; ++x)
		for (int y = 0; y < 511; ++y)
		{
			unsigned char pixel00 = imgray[(x - 1) * SIZE + y - 1];
			unsigned char pixel01 = imgray[(x - 1) * SIZE + y	];
			unsigned char pixel02 = imgray[(x - 1) * SIZE + y + 1];
			unsigned char pixel10 = imgray[(x)	   * SIZE + y - 1];
			unsigned char pixel12 = imgray[(x)	   * SIZE + y + 1];
			unsigned char pixel20 = imgray[(x + 1) * SIZE + y - 1];
			unsigned char pixel21 = imgray[(x + 1) * SIZE + y	];
			unsigned char pixel22 = imgray[(x + 1) * SIZE + y + 1];
			int vert = (pixel00 + 2*pixel01 + pixel02) - (pixel20 + 2 * pixel21 + pixel22);
			int hori = (pixel00 + 2*pixel10 + pixel20) - (pixel02 + 2 * pixel12 + pixel22);
			int tot = vert + hori;
			tot = (tot>60) ? 255 : 0;
			out[x * SIZE + y] = tot;
		}

}
void serial()
{
	IplImage* image;
	image = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;

	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;

	CPUSobel(input, output, cvGetSize(image).height);
	cvShowImage("Image", h_image2);
  cvWaitKey();
}


void mycuda()
{
	IplImage* image;

	image = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	
	//pinned memory host
	//cudaMallocHost((unsigned char**)&output, imgsize);
	//cudaMallocHost((unsigned char**)&input, imgsize);


	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	dim3 dimBlock(32, 32);
	dim3 dimGrid(16, 16);

	sobel<<<dimGrid, dimBlock >>> (d_input, d_output, cvGetSize(image).height);
	CudaCheckError();


	//obtener los datos de la gr�fica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	//mostrar imagen
	cvShowImage("Image", h_image2);

	cvWaitKey();

}
int main()
{
	
	//serial();
	mycuda();
	return 0;
}
