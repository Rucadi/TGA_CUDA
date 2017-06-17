#include <iostream>
#include <stdio.h>      /* printf */
#include <time.h> 
#include <unistd.h> 
#include <stdlib.h>
using namespace std;
clock_t tBegin;
#define TIME_START() { tBegin = clock();}
#define TIME_GET() (double)(clock() - tBegin)/(CLOCKS_PER_SEC/1000)


int col_x = 128;
int col_y = 32;
int fps = 24;

int main()
{
  
  //128x32
  
  char* frame = (char*)malloc(col_x*col_y+2);
  TIME_START();
  long long int frametime =  1000/fps;	
  while(1)
  {
	  
	    if(fread(frame, sizeof(char), col_x*col_y+2, stdin)<0) return 0;
		while(TIME_GET()<frametime);//Mientras no haya pasado el tiempo suficiente esperamos)
		for(int i=0;i<col_y;++i)
		{
			write(1,frame,col_x); printf("\n");
		}
		TIME_START();
  }

  
}