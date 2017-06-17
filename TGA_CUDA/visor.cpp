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
  
  char* frame = (char*)malloc(4098);
  TIME_START();
  long long int frametime =  1000/fps;	
  while(1)
  {
	  int r;
	    //if( (r = fread(frame, sizeof(char), 4098, stdin))==0) return 0;
		fread(frame, sizeof(char), 4098, stdin);
		while(TIME_GET()<frametime);//Mientras no haya pasado el tiempo suficiente esperamos)
			system("cls");
		for(int i=0;i<col_y;++i)
		{
		fwrite(frame+i*128, sizeof(char), 128, stdout); printf("\n");
			//write(1,frame,col_x*col_y); //printf("\n");
		}
		TIME_START();
  }

  
}