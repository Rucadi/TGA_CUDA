
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <device_functions.h>
#include <iostream>

#ifndef __CUDACC__  
#define __CUDACC__
#endif

const unsigned int bSize = 32;

using namespace cv;

cudaEvent_t cStart, cEnd;
#define CUDA_TIME_START() cudaEventCreate(&cStart); cudaEventCreate(&cEnd); cudaEventRecord(cStart);
#define CUDA_TIME_GET(_ms) cudaEventRecord(cEnd); cudaEventSynchronize(cEnd); cudaEventElapsedTime(&_ms,cStart, cEnd);
clock_t tBegin;
#define TIME_START() { tBegin = clock();}
#define TIME_GET() {(double)(clock() - tBegin)/CLOCKS_PER_SEC;}

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


/*La shared memory es a nivel de bloque*/
/*Kernel  = Conjunto de bloques
Block =  Conjunto de threads
 
 
16*16 grid	256 grids
32*32 bloques	1024 bloques
(en total… 512*512…)
*/
__global__ void sobelBlocks(unsigned char* imgray, unsigned char* out, int SIZE)
{

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	__shared__ unsigned char sA[(bSize)*(bSize)];//creamos array que contiene toda la imagen, todos los threads participan al crearlo

	sA[tx*bSize + ty] = imgray[x*SIZE + y];
	
	__syncthreads();//ahora esperamos para que todos tengan una versión de la matriz en shared
	
	int tot;

	//boundary check? evitar que esté fuera del bloque?
	
	if (tx > 0 && ty > 0 && tx < bSize-1 && ty < bSize-1) {
		unsigned char pixel00 = sA[(tx - 1) * bSize + ty - 1];
		unsigned char pixel01 = sA[(tx - 1) * bSize + ty];
		unsigned char pixel02 = sA[(tx - 1) * bSize + ty + 1];
		unsigned char pixel10 = sA[(tx)*      bSize + ty - 1];
		unsigned char pixel12 = sA[(tx)*      bSize + ty + 1];
		unsigned char pixel20 = sA[(tx + 1) * bSize + ty - 1];
		unsigned char pixel21 = sA[(tx + 1) * bSize + ty];
		unsigned char pixel22 = sA[(tx + 1) * bSize + ty + 1];

		int vert = (pixel00 + 2 * pixel01 + pixel02) - (pixel20 + 2 * pixel21 + pixel22);
		int hori = (pixel00 + 2 * pixel10 + pixel20) - (pixel02 + 2 * pixel12 + pixel22);
		tot = vert + hori;
		tot = (tot > 60) ? 255 : 0;

	}

	__syncthreads();
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

unsigned char convertTable(unsigned char value)
{
	unsigned char asciival;

	if (value >= 230)
	{
		asciival = '@';
	}
	else if (value >= 200)
	{
		asciival = '#';
	}
	else if (value >= 180)
	{
		asciival = '8';
	}
	else if (value >= 160)
	{
		asciival = '&';
	}
	else if (value >= 130)
	{
		asciival = 'o';
	}
	else if (value >= 100)
	{
		asciival = ':';
	}
	else if (value >= 70)
	{
		asciival = '*';
	}
	else if (value >= 50)
	{
		asciival = '.';
	}
	else
	{
		asciival = ' ';
	}

	return asciival;
}

void CPUAscii(unsigned char* imgray, unsigned char* out, int SIZE, int cols, int rows)
{
	int  pixels_y = SIZE / cols;
	int pixels_x = SIZE / rows;
	unsigned char* ascii = (unsigned char*) malloc(rows*cols);
	volatile int eol = 0;

	for (int x = 0; x < rows; x++ )
	{
		for (int y = 0; y < cols; y++)
		{

			int sumt = 0;
			int dval = 0;
			for (int i = x*pixels_x; i < x + pixels_x; ++i)
			{
				for (int j = y*pixels_y; j < y + pixels_y; ++j)
				{
					++dval;
					sumt += imgray[i*SIZE + j];
				}
			}

			int media = sumt / dval;
			ascii[x*rows+y] = convertTable(media);
		}
	}

	printf((char*)ascii);
	printf("\n");


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

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32 , 32);
	dim3 dimGrid(16 ,16);


	float milis;
	CUDA_TIME_START();

	sobelBlocks<<<dimGrid, dimBlock >>> (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(milis);

	std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
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

