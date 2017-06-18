
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc_c.h"
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <device_functions.h>
#include <iostream>
#include <string.h>
#ifndef __CUDACC__  
#define __CUDACC__
#endif

const unsigned int bSize = 128;

using namespace cv;

cudaEvent_t cStart, cEnd, cStart2, cEnd2;
#define CUDA_TIME_START() cudaEventCreate(&cStart); cudaEventCreate(&cEnd); cudaEventRecord(cStart,0);   cudaEventSynchronize(cStart);
#define CUDA_TIME_GET(_ms) cudaEventRecord(cEnd,0); cudaEventSynchronize(cEnd); cudaEventElapsedTime(&_ms,cStart, cEnd); cudaEventDestroy(cEnd); cudaEventDestroy(cStart);

#define CUDA_TIME_START2() cudaEventCreate(&cStart2); cudaEventCreate(&cEnd2); cudaEventRecord(cStart2);  cudaEventSynchronize(cStart2);
#define CUDA_TIME_GET2(_ms) cudaEventRecord(cEnd2); cudaEventSynchronize(cEnd2); cudaEventElapsedTime(&_ms,cStart2, cEnd2); cudaEventDestroy(cEnd2); cudaEventDestroy(cStart2);
clock_t tBegin;
#define TIME_START() { tBegin = clock();}
#define TIME_GET() ((float)(clock() - tBegin)/(CLOCKS_PER_SEC/1000));

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
	unsigned char pixel10 = imgray[(x)*     SIZE + y - 1];
	unsigned char pixel12 = imgray[(x)*     SIZE + y + 1];
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

	int sizeShared = bSize + 2;
	//__shared__ unsigned char sA[(bSize + 2)*(bSize + 2)];//creamos array que contiene toda la imagen, todos los threads participan al crearlo
	__shared__ unsigned char sA[(bSize + 2)][(bSize + 2)];
	sA[(tx + 1)][(ty + 1)] = imgray[x*SIZE + y];
	if (ty == 0) // primera columna
	{
		//calcular todos los pixels x-1
		sA[tx + 1][0] = imgray[(x - 1)*SIZE + y];
		if (tx == 0) sA[0][0] = imgray[(x - 1)*SIZE + y - 1];//cargar arriba izquierda
	}
	if (ty == blockDim.y - 1) //ultima columna
	{
		sA[tx + 1][blockDim.y + 1] = imgray[(x + 1)*SIZE + y];
		if (tx == blockDim.x - 1) sA[blockDim.x + 1][blockDim.y + 1] = imgray[(x + 1)*SIZE + y + 1];//cargar abajo derecha
	}
	if (tx == 0)// primera fila
	{
		sA[0][ty + 1] = imgray[(x)*SIZE + y - 1];
		if (ty == blockDim.y - 1) sA[0][blockDim.y + 1] = imgray[(x + 1)*SIZE + y - 1];//cargar arriba derecha
	}
	if (tx == blockDim.x - 1) //ultima fila
	{
		sA[blockDim.x + 1][ty + 1] = imgray[(x)*SIZE + y + 1];
		if (ty == 0) sA[blockDim.y + 1][0] = imgray[(x - 1)*SIZE + y + 1];//cargar abajo izquierda
	}
	__syncthreads();//ahora esperamos para que todos tengan una versión de la matriz en shared

	int tot;

	//boundary check? evitar que esté fuera del bloque?

	int ntx = tx + 1;
	int nty = ty + 1;

	unsigned char pixel00 = sA[(ntx - 1)][nty - 1];
	unsigned char pixel01 = sA[(ntx - 1)][nty];
	unsigned char pixel02 = sA[(ntx - 1)][nty + 1];
	unsigned char pixel10 = sA[(ntx)][nty - 1];
	unsigned char pixel12 = sA[(ntx)][nty + 1];
	unsigned char pixel20 = sA[(ntx + 1)][nty - 1];
	unsigned char pixel21 = sA[(ntx + 1)][+nty];
	unsigned char pixel22 = sA[(ntx + 1)][nty + 1];

	int vert = (pixel00 + 2 * pixel01 + pixel02) - (pixel20 + 2 * pixel21 + pixel22);
	int hori = (pixel00 + 2 * pixel10 + pixel20) - (pixel02 + 2 * pixel12 + pixel22);
	tot = vert + hori;
	tot = (tot > 60) ? 255 : 0;


	__syncthreads();
	out[x * SIZE + y] = tot;

}

__global__ void sobelBlocks_4(unsigned char* imgray, unsigned char* out, int SIZE)
{

	int x = blockDim.x*blockIdx.x * 4 + threadIdx.x * 4;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int tx = threadIdx.x * 4;
	int ty = threadIdx.y;

	__shared__ unsigned char sA[(bSize + 2)][(bSize + 2)];

	sA[(ty + 1)][(tx + 1)] = imgray[y*SIZE + x];
	sA[(ty + 1)][(tx + 2)] = imgray[y*SIZE + x + 1];
	sA[(ty + 1)][(tx + 3)] = imgray[y*SIZE + x + 2];
	sA[(ty + 1)][(tx + 4)] = imgray[y*SIZE + x + 3];
	//cada thread carga 4
	if (ty == 0) // primera fila
	{
		sA[0][tx + 1] = imgray[(y - 1)*SIZE + x];
		sA[0][tx + 2] = imgray[(y - 1)*SIZE + x + 1];
		sA[0][tx + 3] = imgray[(y - 1)*SIZE + x + 2];
		sA[0][tx + 4] = imgray[(y - 1)*SIZE + x + 3];
	}
	if (ty == blockDim.y - 1) //ultima fila
	{
		sA[blockDim.y + 1][tx + 1] = imgray[(y + 1)*SIZE + x];
		sA[blockDim.y + 1][tx + 2] = imgray[(y + 1)*SIZE + x + 1];
		sA[blockDim.y + 1][tx + 3] = imgray[(y + 1)*SIZE + x + 2];
		sA[blockDim.y + 1][tx + 4] = imgray[(y + 1)*SIZE + x + 3];

	}
	if (tx == 0)// primera columna
	{
		sA[ty + 1][0] = imgray[(y - 1)*SIZE + x];

	}
	if (threadIdx.x == blockDim.x - 1) //ultima columna
	{
		sA[ty + 1][blockDim.x * 4 + 1] = imgray[(y + 1)*SIZE + x + 4];
	}
	sA[0][0] = sA[0][1];
	sA[blockDim.y + 1][blockDim.y + 1] = sA[blockDim.y][blockDim.y];
	sA[0][blockDim.y + 1] = sA[0][blockDim.y];
	sA[blockDim.y + 1][0] = sA[blockDim.y][0];
	__syncthreads();

	int ntx = tx + 1;
	int nty = ty + 1;
	uchar4 rest;
	int tot;

	unsigned char pixel00 = sA[(nty - 1)][ntx - 1];
	unsigned char pixel01 = sA[(nty - 1)][ntx];
	unsigned char pixel02 = sA[(nty - 1)][ntx + 1];
	unsigned char pixel10 = sA[nty][ntx - 1];
	unsigned char pixel11 = sA[nty][ntx];
	unsigned char pixel12 = sA[nty][ntx + 1];
	unsigned char pixel20 = sA[(nty + 1)][ntx - 1];
	unsigned char pixel21 = sA[(nty + 1)][ntx];
	unsigned char pixel22 = sA[(nty + 1)][ntx + 1];


	int vert = (pixel00 + 2 * pixel01 + pixel02) - (pixel20 + 2 * pixel21 + pixel22);
	int hori = (pixel00 + 2 * pixel10 + pixel20) - (pixel02 + 2 * pixel12 + pixel22);
	rest.x = ((vert + hori) > 60) ? 255 : 0;

	pixel00 = sA[(nty - 1)][ntx + 2];
	pixel10 = sA[nty][ntx + 2];
	pixel20 = sA[(nty + 1)][ntx + 2];

	vert = (pixel01 + 2 * pixel02 + pixel00) - (pixel21 + 2 * pixel22 + pixel20);
	hori = (pixel01 + 2 * pixel11 + pixel21) - (pixel00 + 2 * pixel10 + pixel20);
	rest.y = ((vert + hori) > 60) ? 255 : 0;

	pixel01 = sA[(nty - 1)][ntx + 3];
	pixel11 = sA[nty][ntx + 3];
	pixel21 = sA[(nty + 1)][ntx + 3];

	vert = (pixel02 + 2 * pixel00 + pixel01) - (pixel22 + 2 * pixel20 + pixel21);
	hori = (pixel02 + 2 * pixel12 + pixel22) - (pixel01 + 2 * pixel11 + pixel21);

	rest.z = ((vert + hori) > 60) ? 255 : 0;

	pixel02 = sA[(nty - 1)][ntx + 4];
	pixel12 = sA[nty][ntx + 4];
	pixel22 = sA[(nty + 1)][ntx + 4];
	vert = (pixel00 + 2 * pixel01 + pixel02) - (pixel20 + 2 * pixel21 + pixel22);
	hori = (pixel00 + 2 * pixel10 + pixel20) - (pixel02 + 2 * pixel12 + pixel22);

	rest.w = ((vert + hori) > 60) ? 255 : 0;
	__syncthreads();

	out[y * SIZE + x] = rest.x;
	out[y * SIZE + x + 1] = rest.y;
	out[y * SIZE + x + 2] = rest.z;
	out[y * SIZE + x + 3] = rest.w;
}
__global__ void asciiBlocks(unsigned char* imgray, unsigned char* out, int SIZEX, int bSizex, int bSizey) {

	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int sum = 0;
	/*
	for (int i = x*bSizex; i < x*bSizex+bSizex; i++)
	for (int j = y*bSizey; j < y*bSizey+bSizey; j++)
	sum = sum + imgray[j*SIZEX+i];

	*/
	__shared__ unsigned char sdata[8192];
	int tid = ty*SIZEX + tx;

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = imgray[y*SIZEX + x];

	__syncthreads();

	// contiguous range pattern
	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
		if (threadIdx.x < offset) {
			// add a partial sum upstream to our own
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];
		}
		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}

	// thread 0 writes the final result
	if (threadIdx.x == 0) {
		//	per_block_results[blockIdx.x] = sdata[0];
	}



	//sdata[tid] = g_imgray[i];
	__syncthreads();





	__syncthreads();

	sum = sum / (bSizex*bSizey);
	unsigned char asciival = 'a';
	unsigned char value = sum;

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

	int outSizex = SIZEX / bSizex;
	out[y*outSizex + x] = asciival;

}

__global__ void asciiMean(unsigned char* imgray, unsigned char* out, int SIZE, int bSizex, int bSizey) {

	int resx = blockIdx.x; //columna
	int resy = blockIdx.y; //fila
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int sum = 0;
	extern __shared__ int sdata[];//int para evitar overflow....

	sdata[tx] = imgray[x*SIZE + y];
	sdata[tx] = imgray[x*SIZE + y];
}

struct kernel_global
{
	float kernel_time;
	float global_time;
};
struct kernel_global k_global;

void CPUSobel(unsigned char* imgray, unsigned char* out, int SIZE)
{
	for (int x = 1; x<SIZE - 1; ++x)
		for (int y = 0; y < SIZE - 1; ++y)
		{
			unsigned char pixel00 = imgray[(x - 1) * SIZE + y - 1];
			unsigned char pixel01 = imgray[(x - 1) * SIZE + y];
			unsigned char pixel02 = imgray[(x - 1) * SIZE + y + 1];
			unsigned char pixel10 = imgray[(x)* SIZE + y - 1];
			unsigned char pixel12 = imgray[(x)* SIZE + y + 1];
			unsigned char pixel20 = imgray[(x + 1) * SIZE + y - 1];
			unsigned char pixel21 = imgray[(x + 1) * SIZE + y];
			unsigned char pixel22 = imgray[(x + 1) * SIZE + y + 1];
			int vert = (pixel00 + 2 * pixel01 + pixel02) - (pixel20 + 2 * pixel21 + pixel22);
			int hori = (pixel00 + 2 * pixel10 + pixel20) - (pixel02 + 2 * pixel12 + pixel22);
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
void CPUAscii(unsigned char* imgray, int SIZE, int cols, int rows)
{


	cols = 128;
	rows = 32;

	int  pixels_y = SIZE / cols;
	int pixels_x = SIZE / rows;
	//printf("pixelx=%d, pixely=%d SIZE=%d ", pixels_x, pixels_y, SIZE);
	//printf("Cols:%d Rows:%d", cols, rows);
	unsigned char* ascii = (unsigned char*)malloc(rows*cols + 1);
	volatile int eol = 0;

	for (int x = 0; x < rows; x++)
	{
		for (int y = 0; y < cols; y++)
		{

			int sumt = 0;
			int dval = 1;
			for (int i = x*pixels_x; i < x*pixels_x + pixels_x; ++i)
			{
				for (int j = y*pixels_y; j < y*pixels_y + pixels_y; ++j)
				{
					++dval;
					sumt += imgray[i*SIZE + j];
				}
			}

			if (dval == 0) dval = 1;
			int media = sumt / dval;
			ascii[x*cols + y] = convertTable(media);
		}
	}

	ascii[rows*cols] = 0;
	printf((char*)ascii);
	printf("\n\n");
	free(ascii);

}
float serial()
{
	IplImage* image;
	image = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;

	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	TIME_START();
	CPUSobel(input, output, cvGetSize(image).height);
	return TIME_GET();
}
void ASCII()
{
	IplImage* image;
	image = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;

	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;

	CPUSobel(input, output, cvGetSize(image).height);
	CPUAscii((unsigned char*)h_image2->imageData, cvGetSize(image).height, 207, 61);

	cvShowImage("Image", h_image2);
	cvWaitKey();
}
void cudaASCII() {

	IplImage* image;
	image = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;

	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;

	CPUSobel(input, output, cvGetSize(image).height);


	//CONSOLE_SCREEN_BUFFER_INFO csbi;
	int a; std::cin >> a;

	int cols = 128;
	int rows = 32;

	std::cout << cols << " g " << rows;
	unsigned char *ascii = (unsigned char*)malloc(rows*cols);

	int SIZE = cvGetSize(image).height;

	int pixels_x = SIZE / rows; // character sizeX
	int pixels_y = SIZE / cols; //character sizeY
	int asciisize = rows*cols;
	unsigned char *d_input;
	unsigned char *d_output;

	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, asciisize);

	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//thread x block GRID
	int x, y;
	x = y = 2;
	while (x < rows)
		x *= 2;
	while (y < cols)
		y *= 2;

	//thread x block GRID
	dim3 dimBlock(pixels_x, pixels_y);
	dim3 dimGrid(rows, cols);

	//thread x pixel GRID
	//dim3 dimBlock(32, 32);

	float milis;
	CUDA_TIME_START();

	asciiBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height, pixels_x, pixels_y);
	CUDA_TIME_GET(k_global.kernel_time);

	std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica

	cudaMemcpy(ascii, d_output, asciisize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);

	//for (int i = 0; i < asciisize; i++)
	//std::cout << std::hex << (int)ascii[i];
	printf((char*)ascii);
	printf("\n");
	printf("hello");
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
	dim3 dimBlock(32, 32);
	dim3 dimGrid(16, 16);


	float milis;
	CUDA_TIME_START();

	sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	//mostrar imagen

	cvShowImage("Image", h_image2);
	cvWaitKey();

}

/*	namedWindow("Image", WINDOW_NORMAL);
cvShowImage("Image", h_image2);
cvWaitKey();*/

float serial128()
{

	IplImage* src;
	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *image = cvCreateImage(cvSize(128, 128), src->depth, src->nChannels);
	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;

	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	TIME_START();
	CPUSobel(input, output, cvGetSize(image).height);
	return TIME_GET();
}
float serial512()
{

	IplImage* src;
	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *image = cvCreateImage(cvSize(512, 512), src->depth, src->nChannels);
	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;

	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	TIME_START();
	CPUSobel(input, output, cvGetSize(image).height);
	return TIME_GET();
}
float serial3072()
{

	IplImage* src;
	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *image = cvCreateImage(cvSize(3072, 3072), src->depth, src->nChannels);
	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;

	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	TIME_START();
	CPUSobel(input, output, cvGetSize(image).height);
	return TIME_GET();
}
float serial4096()
{

	IplImage* src;
	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);
	IplImage *image = cvCreateImage(cvSize(4096, 4096), src->depth, src->nChannels);
	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;

	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	TIME_START();
	CPUSobel(input, output, cvGetSize(image).height);
	return TIME_GET();
}
struct kernel_global cuda128()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(128, 128), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32, 32);
	dim3 dimGrid(4, 4);

	float milis;
	CUDA_TIME_START();
	sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	return k_global;
}
struct kernel_global cuda128_4()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(128, 128), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(16, 64);
	dim3 dimGrid(2, 2);

	float milis;
	CUDA_TIME_START();

	sobelBlocks_4 << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen

	return k_global;

}
struct kernel_global cuda128_s()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(128, 128), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32, 32);
	dim3 dimGrid(4, 4);

	float milis;
	CUDA_TIME_START();

	sobel << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen

	return k_global;
}
struct kernel_global cuda512()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(512, 512), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32, 32);
	dim3 dimGrid(16, 16);

	float milis;
	CUDA_TIME_START();

	sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen

	return k_global;


}
struct kernel_global cuda512_4()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(512, 512), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(16, 64);//x = 8
	dim3 dimGrid(8, 8);

	float milis;
	CUDA_TIME_START();

	sobelBlocks_4 << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen

	//cvShowImage("Image", h_image2);
	//CPUAscii((unsigned char*)h_image2->imageData, cvGetSize(image).height, 207, 61);
	//cvWaitKey();
	return k_global;

}
struct kernel_global cuda512_s()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(512, 512), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32, 32);
	dim3 dimGrid(16, 16);

	float milis;
	CUDA_TIME_START();

	sobel << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen

	return k_global;


}
struct kernel_global cuda3072()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(3072, 3072), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32, 32);
	dim3 dimGrid(96, 96);

	float milis;
	CUDA_TIME_START();
	sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen

	return k_global;

}
struct kernel_global cuda3072_4()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(3072, 3072), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(16, 64);
	dim3 dimGrid(48, 48);

	float milis;
	CUDA_TIME_START();

	sobelBlocks_4 << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);

	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen

	return k_global;

}
struct kernel_global cuda3072_s()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(3072, 3072), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32, 32);
	dim3 dimGrid(96, 96);

	float milis;
	CUDA_TIME_START();

	sobel << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);

	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen

	return k_global;

}
struct kernel_global cuda4096()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(4096, 4096), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32, 32);
	dim3 dimGrid(128, 128);

	float milis;
	CUDA_TIME_START();

	sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen
	return k_global;

}
struct kernel_global cuda4096_4()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(4096, 4096), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(16, 64);
	dim3 dimGrid(64, 64);

	float milis;
	CUDA_TIME_START();

	sobelBlocks_4 << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	return k_global;

}
struct kernel_global cuda4096_s()
{
	IplImage* src;

	src = cvLoadImage("cameraman.png", CV_LOAD_IMAGE_GRAYSCALE);


	IplImage *image = cvCreateImage(cvSize(4096, 4096), src->depth, src->nChannels);


	cvResize(src, image);

	IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	int imgsize = cvGetSize(image).height* cvGetSize(image).width;
	unsigned char *output = (unsigned char*)h_image2->imageData;
	unsigned char *input = (unsigned char*)image->imageData;
	unsigned char *d_input;
	unsigned char *d_output;

	CUDA_TIME_START2();
	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32, 32);
	dim3 dimGrid(128, 128);

	float milis;
	CUDA_TIME_START();

	sobel << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(k_global.kernel_time);

	//std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica
	cudaMemcpy(output, d_output, imgsize, cudaMemcpyDeviceToHost);
	cudaFree(d_output);
	cudaFree(d_input);
	CUDA_TIME_GET2(k_global.global_time);
	//mostrar imagen
	return k_global;

}
void createVideoAscii(char* arg)
{

	CvCapture* capture = cvCaptureFromAVI(arg);
	if (capture == NULL) printf("Capture null");

	unsigned char *d_input;
	unsigned char *d_output;

	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, 512 * 512);
	cudaMalloc((unsigned char**)&d_output, 512 * 512);
	while (1)
	{
		IplImage* frame = NULL;

		frame = cvQueryFrame(capture);
		if (frame == NULL)
		{
			int fafafa;
			printf("frame null\n");
			//scanf("%d", fafafa);
			break;
		}

		//transformamos la imagen
		IplImage* gray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
		cvCvtColor(frame, gray, CV_RGB2GRAY);
		IplImage *image = cvCreateImage(cvSize(512, 512), IPL_DEPTH_8U, 1);
		cvResize(gray, image);

		IplImage* h_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
		IplImage* d_image2 = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

		unsigned char *output = (unsigned char*)h_image2->imageData;
		unsigned char *input = (unsigned char*)image->imageData;


		//copiamos el input al device
		cudaMemcpy(d_input, input, 512 * 512, cudaMemcpyHostToDevice);
		//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
		dim3 dimBlock(32, 32);//x = 8
		dim3 dimGrid(16, 16);

		float milis;
		CUDA_TIME_START();
		sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
		CUDA_TIME_GET(k_global.kernel_time);

		CudaCheckError();
		cudaMemcpy(output, d_output, 512 * 512, cudaMemcpyDeviceToHost);
		CPUAscii((unsigned char*)h_image2->imageData, cvGetSize(image).height, 128, 32);
	}
	cudaFree(d_output);
	cudaFree(d_input);

}


void printStatistics(int size)
{
	printf("Tiempo Global: %4.6f milseg\n", k_global.global_time);
	printf("Tiempo Kernel: %4.6f milseg\n", k_global.kernel_time);
	printf("Rendimiento Global: %4.2f GFLOPS\n", (2.0 * size*size) / (1000000.0 *  k_global.global_time));
	printf("Rendimiento Kernel: %4.2f GFLOPS\n", (2.0 *  size*size) / (1000000.0 * k_global.kernel_time));
}

void k128()
{

	printf("Image of 128x128\n");
	serial128();

	printf("CUDA_o1\n");
	cuda128();
	printStatistics(128);

	printf("CUDA_o2\n");
	cuda128_s();
	printStatistics(128);


	printf("CUDA_o3\n");
	cuda128_4();
	printStatistics(128);


}

void k512()
{
	printf("Image of 512*512\n");

	printf("CUDA_o1\n");
	cuda512();
	printStatistics(512);

	printf("CUDA_o2\n");
	cuda512_s();
	printStatistics(512);


	printf("CUDA_o3\n");
	cuda512_4();
	printStatistics(512);
}

void k3072()
{
	printf("Image of 3072*3072\n");
	printf("CUDA_o1\n");
	cuda3072();
	printStatistics(3072);

	printf("CUDA_o2\n");
	cuda3072_s();
	printStatistics(3072);


	printf("CUDA_o3\n");
	cuda3072_4();
	printStatistics(3072);
}

void k4096()
{

	printf("Image of 4096*4096\n");
	printf("CUDA_o1\n");
	cuda4096();
	printStatistics(4096);

	printf("CUDA_o2\n");
	cuda4096_s();
	printStatistics(4096);


	printf("CUDA_o3\n");
	cuda4096_4();
	printStatistics(4096);
}

int main(int argc, char **argv)
{

	if (argc == 2)
	{
		createVideoAscii(argv[1]);
	}
	else
	{
		k128();
		k512();
		k3072();
		k4096();
	}

	return 0;
}

