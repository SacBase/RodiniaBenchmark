#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef N
#define N 2048
#endif
#define MAXLINE 128

int create_matrix_from_file(float **mp)
{
  int i, j, size;
  float *m;
  FILE *fp = NULL;
  char filename[MAXLINE];

  sprintf(filename, "../input/%d.dat", N); 

  fp = fopen(filename, "rb");
  if ( fp == NULL) {
    return( 0);
  }

  fscanf(fp, "%d\n", &size);

  if( size != N) {
    printf("Wrong file read. Expecting %d.dat but read %d.dat\n", N, size);
    return( 0);
  }

  m = (float*) malloc(sizeof(float)*N*N);
  if ( m == NULL) {
      fclose(fp);
      return( 0);
  }

  for (i=0; i < size; i++) {
    for (j=0; j < size; j++) {
      fscanf(fp, "%f ", m+i*size+j);
    }
  }

  fclose(fp);

  *mp = m;

  return( 1);
}



int main()
{
  int k, i, j, ret;
  float *mat;
  double runtime;
  struct timeval tv1, tv2;
  float kk;


  ret = create_matrix_from_file( &mat);
 
  if( !ret) {
    printf("Error has occured during reading file. Abort.\n");
    return( 1);
  } 

  gettimeofday( &tv1, NULL);

  for( k = 0; k < N-1; k++) {
    for( i = k+1; i < N; i++) {
      mat[i*N+k] = mat[i*N+k]/mat[k*N+k];
    }
    for( i = k+1; i < N; i++) {
      for( j = k+1; j < N; j++) {
        mat[i*N+j] = mat[i*N+j]-mat[i*N+k]*mat[k*N+j];
      }
    } 
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0 + tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0 + tv1.tv_usec/1000.0));
  printf("Runtime(milli-seconds): %f\n", runtime);

#ifdef OUTPUT
  for( i = 0; i < N; i++) {
    for( j = 0; j < N; j++) {
      printf("%f ", mat[i*N+j]);
    }
    printf("\n");
  }
#else
  printf("%f\n", mat[0]);
#endif

  return( 0);
}



