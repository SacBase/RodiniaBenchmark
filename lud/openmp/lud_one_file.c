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

#ifdef SIM /*  */

  
  gettimeofday( &tv1, NULL);
 
  double *mat2;

  for( i = 0; i < SIZE-1; i++) {

    /*************************************************/
    for( j=i+1; j < SIZE; j++) {
      for( n=i; n < i+1; n++) {
        mat[j*SIZE+n] /= mat[i*SIZE+i]; 
      }
    } 

    mat2 = (double*)malloc(sizeof(double)*SIZE*SIZE); 
    /*************************************************/

    for( j=0; j < i+1; j++) {
      for( n=0; n < SIZE; n++) {
        mat2[j*SIZE+n] = mat[j*SIZE+n]; 
      }
    }  

    for( j=i+1; j < SIZE; j++) {
      for( n=0; n < i+1; n++) {
        mat2[j*SIZE+n] = mat[j*SIZE+n]; 
      }
    }  

    /*************************************************/

    for( m=i+1; m < SIZE; m++) {
      for( j=i+1; j < SIZE; j++) {
        mat2[m*SIZE+j] = mat[m*SIZE+j] - mat[m*SIZE+i]*mat[i*SIZE+j]; 
      }
    } 

    /*************************************************/

    free(mat);
    mat = mat2;
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("Total time %f\n", runtime);

#endif



#ifdef NIPU /* No In-Place Update */

  struct timeval copy_start, copy_stop, compute_start, compute_stop;
  double copy_time=0.0, compute_time=0.0;
  
  gettimeofday( &tv1, NULL);
 
  double *mat2;

  for( i = 0; i < SIZE-1; i++) {

    /*************************************************/
    gettimeofday( &compute_start, NULL);
    for( j=i+1; j < SIZE; j++) {
      for( n=i; n < i+1; n++) {
        mat[j*SIZE+n] /= mat[i*SIZE+i]; 
      }
    } 
    gettimeofday( &compute_stop, NULL);
    compute_time += ((compute_stop.tv_sec*1000.0+ compute_stop.tv_usec/1000.0)-(compute_start.tv_sec*1000.0+ compute_start.tv_usec/1000.0));


    mat2 = (double*)malloc(sizeof(double)*SIZE*SIZE); 
    /*************************************************/

    gettimeofday( &copy_start, NULL);
    for( j=0; j < i+1; j++) {
      for( n=0; n < SIZE; n++) {
        mat2[j*SIZE+n] = mat[j*SIZE+n]; 
      }
    }  

    for( j=i+1; j < SIZE; j++) {
      for( n=0; n < i+1; n++) {
        mat2[j*SIZE+n] = mat[j*SIZE+n]; 
      }
    }  
    gettimeofday( &copy_stop, NULL);
    copy_time += ((copy_stop.tv_sec*1000.0+ copy_stop.tv_usec/1000.0)-(copy_start.tv_sec*1000.0+ copy_start.tv_usec/1000.0));


    /*************************************************/


    gettimeofday( &compute_start, NULL);
    for( m=i+1; m < SIZE; m++) {
      for( j=i+1; j < SIZE; j++) {
        mat2[m*SIZE+j] = mat[m*SIZE+j] - mat[m*SIZE+i]*mat[i*SIZE+j]; 
      }
    } 
    gettimeofday( &compute_stop, NULL);
    compute_time += ((compute_stop.tv_sec*1000.0+ compute_stop.tv_usec/1000.0)-(compute_start.tv_sec*1000.0+ compute_start.tv_usec/1000.0));

    /*************************************************/

    free(mat);
    mat = mat2;
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("Total time %f, Compute time %f, Copy time %f\n", runtime, compute_time, copy_time);

#endif


#ifdef IPU  /* In-Place Update */
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
  printf("Reuse time: %f\n", runtime);
#endif

  res = (int)mat[0];

  free(mat);

  return( res);
}



