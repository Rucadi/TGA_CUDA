#include <iostream>
#include <stdio.h>      /* printf */
#include <time.h> 
#include <unistd.h> 
#include <stdlib.h>
using namespace std;
clock_t tBegin;
#define TIME_START() { tBegin = clock();}
#define TIME_GET() (double)(clock() - tBegin)/(CLOCKS_PER_SEC/1000)


int col_x;
int col_y;
int frames;
int fps;
int width, height;

int main()
{
  
 cin>>col_x>>col_y>>frames>>fps>>width>>height;
  char* frame = (char*)malloc(col_x*col_y);
  char  		buff[] = "\e[8;509;100t";
  sprintf(buff,"\e[8;%d;%dt",width,height);
  cout << buff;
  
  TIME_START();
  long long int frametime =  1000/fps;	
  while(frames--)
  {
	  
	    fread(frame, sizeof(char), col_x*col_y, stdin);
		while(TIME_GET()<frametime);//Mientras no haya pasado el tiempo suficiente esperamos)
		write(1,frame,col_x*col_y);
	    //for(int i=0;i<10;++i) cout<<frame[i]<<endl; 	
		fflush(stdout);
		TIME_START();
  }

  
}