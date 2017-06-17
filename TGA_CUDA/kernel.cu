
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <iostream>
#include <stdio.h>
#include <device_functions.h>
#include <iostream>
#include <Windows.h>
#ifndef __CUDACC__  
#define __CUDACC__
#endif

const unsigned int bSize = 128;

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

	int sizeShared = bSize + 2;
	//__shared__ unsigned char sA[(bSize + 2)*(bSize + 2)];//creamos array que contiene toda la imagen, todos los threads participan al crearlo
	__shared__ unsigned char sA[(bSize + 2)][(bSize + 2)];
	sA[(tx + 1)][(ty + 1)] = imgray[x*SIZE + y];
	if (ty == 0) // primera columna
	{
		//calcular todos los pixels x-1
		sA[tx+1][0] = imgray[(x - 1)*SIZE+y]; 
		if(tx==0) sA[0][0] = imgray[(x-1)*SIZE+y-1];//cargar arriba izquierda
	}
	if (ty == blockDim.y-1) //ultima columna
	{
		sA[tx + 1][blockDim.y + 1] = imgray[(x + 1)*SIZE+y];
		if(tx==31) sA[blockDim.x+1][blockDim.y+1] = imgray[(x + 1)*SIZE+ y+1];//cargar abajo derecha
	}
	if (tx == 0)// primera fila
	{
		sA[0][ty+1] = imgray[(x)*SIZE + y - 1];
		if(ty==31) sA[0][blockDim.y+1] = imgray[(x+1)*SIZE+y -1];//cargar arriba derecha
	}
	if (tx == blockDim.x-1) //ultima fila
	{
		sA[blockDim.x + 1][ty+1] = imgray[(x)*SIZE + y + 1];
		if(ty==0) sA[blockDim.y+1][0] = imgray[(x - 1)*SIZE+y + 1];//cargar abajo izquierda
	}
	__syncthreads();//ahora esperamos para que todos tengan una versión de la matriz en shared

	int tot;

	//boundary check? evitar que esté fuera del bloque?

	int ntx = tx + 1;
	int nty = ty + 1;

	unsigned char pixel00 = sA[(ntx - 1)][nty - 1];
	unsigned char pixel01 = sA[(ntx - 1)][ nty];
	unsigned char pixel02 = sA[(ntx - 1)][nty + 1];
	unsigned char pixel10 = sA[(ntx)][nty - 1];
	unsigned char pixel12 = sA[(ntx)]	[nty + 1];
	unsigned char pixel20 = sA[(ntx + 1)][nty - 1];
	unsigned char pixel21 = sA[(ntx + 1)][+ nty];
	unsigned char pixel22 = sA[(ntx + 1)][nty + 1];

	int vert = (pixel00 + 2 * pixel01 + pixel02) - (pixel20 + 2 * pixel21 + pixel22);
	int hori = (pixel00 + 2 * pixel10 + pixel20) - (pixel02 + 2 * pixel12 + pixel22);
	tot = vert + hori;
	tot = (tot > 60) ? 255 : 0;


	__syncthreads();
	out[x * SIZE + y] = tot;

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
		sdata[tid] = imgray[y*SIZEX+x];

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
void CPUAscii(unsigned char* imgray, int SIZE, int cols, int rows)
{
	CONSOLE_SCREEN_BUFFER_INFO csbi;

	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
	rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;

	int  pixels_y = SIZE / cols;
	int pixels_x = SIZE / rows;

	//printf("Cols:%d Rows:%d", cols, rows);
	unsigned char* ascii = (unsigned char*) malloc(rows*cols);
	volatile int eol = 0;

	for (int x = 0; x < rows; x++ )
	{
		for (int y = 0; y < cols; y++)
		{

			int sumt = 0;
			int dval = 1;
			for (int i = x*pixels_x; i < x*pixels_x + pixels_x; ++i)
			{
				for (int j = y*pixels_y; j < y*pixels_y +  pixels_y; ++j)
				{
					++dval;
					sumt += imgray[i*SIZE + j];

					// printf("i:%d j:%d\n", i, j);
				}
			}
		//	printf("Val:%f\n", sumt / dval);
			if (dval == 0) dval = 1;
			int media = sumt / dval;

			//printf("i:%d j:%d\n", x, y);
			ascii[x*cols+y] = convertTable(media);
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


	CONSOLE_SCREEN_BUFFER_INFO csbi;
	int a; std::cin >> a;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	int cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
	int rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;

	std::cout << cols << " g "  << rows;
	unsigned char *ascii = (unsigned char*) malloc(rows*cols);

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
	dim3 dimBlock(pixels_x,pixels_y);
	dim3 dimGrid(rows, cols);

	//thread x pixel GRID
	//dim3 dimBlock(32, 32);

	float milis;
	CUDA_TIME_START();

	asciiBlocks<<<dimGrid, dimBlock >>> (d_input, d_output, cvGetSize(image).height, pixels_x, pixels_y);
	CUDA_TIME_GET(milis);

	std::cout << "Milisegundos ejecución CPU:" << milis << std::endl;
	CudaCheckError();


	//obtener los datos de la gráfica

	cudaMemcpy(ascii, d_output,  asciisize, cudaMemcpyDeviceToHost);
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
	dim3 dimBlock(32 , 32);
	dim3 dimGrid(16,16);


	float milis;
	CUDA_TIME_START();

	sobelBlocks<<<dimGrid, dimBlock >>> (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(milis);

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


void cuda128()
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

	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(16, 16);
	dim3 dimGrid(8, 8);

	float milis;
	CUDA_TIME_START();

	sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(milis);

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


void cuda512()
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

	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(8, 8);
	dim3 dimGrid(16, 16);

	float milis;
	CUDA_TIME_START();

	sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);
	CUDA_TIME_GET(milis);

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


void cuda3072()
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

	//reservamos espacio en la tg para nuestras imagenes
	cudaMalloc((unsigned char**)&d_input, imgsize);
	cudaMalloc((unsigned char**)&d_output, imgsize);


	//copiamos el input al device
	cudaMemcpy(d_input, input, imgsize, cudaMemcpyHostToDevice);

	//32*16 = 512 deberíamos soportar hasta 128x128,512x512,3072x3072,4096x4096
	dim3 dimBlock(32,32);
	dim3 dimGrid(96, 96);

	float milis;
	CUDA_TIME_START();

	sobelBlocks << <dimGrid, dimBlock >> > (d_input, d_output, cvGetSize(image).height);

	CUDA_TIME_GET(milis);

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


void cuda4096()
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
	CUDA_TIME_GET(milis);

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


int main()
{
	
	//serial();
	//ASCII();
	//cudaASCII();
	//mycuda();
	//cuda128();
	cuda4096();
	return 0;
}

