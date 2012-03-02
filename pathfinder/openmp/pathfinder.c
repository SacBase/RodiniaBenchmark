#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>


#ifndef SIZE
#define SIZE 131072 
#endif

#ifndef ITER 
#define ITER 200 
#endif

#define MIN(a, b) ((a)<=(b) ? (a) : (b))


int main(int argc, char** argv)
{
  int rows, cols;
  int* wall;
  int* result;
  int *src, *dst, *temp;
  int min, i, j, t, n;
  double runtime;

  rows = ITER;
  cols = SIZE;

  wall = (int*)malloc(sizeof(int)*rows*cols);
  result = (int*)malloc(sizeof(int)*cols); 
	
  srand( 7);

  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      wall[i*SIZE+j] = rand() % 10;
    }
  }

  for (j = 0; j < cols; j++) {
    result[j] = wall[j];
  }

  dst = result;
  src = (int*)malloc(sizeof(int)*cols);

  struct timeval tv1, tv2;
  gettimeofday( &tv1, NULL);

  for (t = 0; t < rows-1; t++) {
    temp = src;
    src = dst;
    dst = temp;

    #pragma omp parallel for private(min)
    for(n = 0; n < cols; n++){
      min = src[n];
      if (n > 0) {
        min = MIN(min, src[n-1]);
      }
      if (n < cols-1) {
        min = MIN(min, src[n+1]);
      }
      dst[n] = wall[(t+1)*SIZE+n]+min;
    }
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  //printf("%f\n", runtime);

#ifdef OUTPUT 
  for (i = 0; i < cols; i++) {
    printf("%d ",dst[i]);
  }
  printf("\n");
#else
  printf("%d ",dst[0]);
#endif

  free(wall);
  free(dst);
  free(src);
 
  return( 0);
}

