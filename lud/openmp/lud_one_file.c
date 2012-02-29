#include <stdio.h>
#include <stdlib.h>

#ifndef SIZE
#define SIZE 256
#endif

static int read_matrix_from_file(double **mp, const char* filename)
{
  int i, j, size;
  double *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if( fp == NULL) {
    return 0; 
  }

  fscanf(fp, "%d\n", &size);

  if( size != SIZE) {
    fprintf(stderr, "Size mismatch!\n");
    exit(1);
  }

  m = (double*) malloc(sizeof(double)*SIZE*SIZE);
  if ( m == NULL) {
    fclose(fp);
    return 0;
  }

  for (i=0; i < SIZE; i++) {
    for (j=0; j < SIZE; j++) {
      fscanf(fp, "%lf ", &m[i*SIZE+j]);
    }
  }

  fclose(fp);

  *mp = m;

  return 1;
}


int main(int argc, char **argv)
{
  double *mat, runtime;
  int ret, i, j, m, n, res;
  char *input_file;
  struct timeval tv1, tv2;

  if( argc != 2) {
    fprintf(stderr, "Usage: ./a.out input_file\n");
    exit(1);
  }

  input_file = argv[1];

  ret = read_matrix_from_file(&mat, input_file);

  if( !ret) {
    fprintf(stderr, "error read matrix from file %s\n", input_file);
    exit(1);
  }

  gettimeofday( &tv1, NULL);

  for( i = 0; i < SIZE-1; i++) {
    for( j=i+1; j < SIZE; j++) {
      for( n=i; n < i+1; n++) {
        mat[j*SIZE+n] /= mat[i*SIZE+i]; 
      }
    } 

    for( m=i+1; m < SIZE; m++) {
      for( j=i+1; j < SIZE; j++) {
        mat[m*SIZE+j] -= mat[m*SIZE+i]*mat[i*SIZE+j]; 
      }
    } 
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

  res = (int)mat[0];

  free(mat);

  return( res);
}



