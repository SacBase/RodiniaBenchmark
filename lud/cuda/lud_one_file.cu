#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#ifndef SIZE
#define SIZE 256
#endif

#define BLOCK_X 16
#define BLOCK_Y 16

static int read_matrix_from_file(double **mp, const char* filename)
{
  int i, j, size;
  double *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if( fp == NULL) {
    return 0; 
  }

  int res = fscanf(fp, "%d\n", &size);

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
      res = fscanf(fp, "%lf ", &m[i*SIZE+j]);
    }
  }

  fclose(fp);

  *mp = m;

  return 1;
}

__global__ void column( double *mat, int i)
{
  int tx = blockIdx.x*blockDim.x+threadIdx.x+i;
  int ty = blockIdx.y*blockDim.y+threadIdx.y+i+1;

  if( tx >= i+1 || ty >= SIZE) return;

  mat[ty*SIZE+tx] /= mat[i*SIZE+i];  
}

__global__ void column2( double *mat, double mat_i_i, int i)
{
  int tx = blockIdx.x*blockDim.x+threadIdx.x+i;
  int ty = blockIdx.y*blockDim.y+threadIdx.y+i+1;

  if( tx >= i+1 || ty >= SIZE) return;

  mat[ty*SIZE+tx] /= mat_i_i;  
}

__global__ void submatrix( double *mat_out, double *mat_in, int i)
{
  int tx = blockIdx.x*blockDim.x+threadIdx.x+i+1;
  int ty = blockIdx.y*blockDim.y+threadIdx.y+i+1;

  if( tx >= SIZE || ty >= SIZE) return;

  mat_out[ty*SIZE+tx] -= mat_in[ty*SIZE+i]*mat_in[i*SIZE+tx];  
}

__global__ void copy( double *mat_out, double *mat_in, int lb0, int lb1, int ub0, int ub1)
{
  int tx = blockIdx.x*blockDim.x+threadIdx.x+lb1;
  int ty = blockIdx.y*blockDim.y+threadIdx.y+lb0;

  if( tx >= ub1 || ty >= ub0) return;

  mat_out[ty*SIZE+tx] = mat_in[ty*SIZE+tx];  
}


int main(int argc, char **argv)
{
  double *mat, runtime, *mat_in_d, *mat_out_d;
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


/*
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
*/
  
  dim3 block(BLOCK_X, BLOCK_Y);
  dim3 grid( 0,0);


#ifdef UNOPT 

#if 0
  gettimeofday( &tv1, NULL);
  for( i = 0; i < 20; i++) {
    double mat_i_i = mat[i*SIZE+i];
  
    double *tmp1_d;
    cudaMalloc((void**)&tmp1_d, sizeof(double)*SIZE*SIZE); 
    cudaMemcpy(tmp1_d, mat, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);
    free(mat);

    grid.x = 1/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    column2<<<grid, block>>>( tmp1_d, mat_i_i, i); 

    double *tmp1 = (double*)malloc(sizeof(double)*SIZE*SIZE);
    cudaMemcpy(tmp1, tmp1_d, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);
    cudaFree(tmp1_d);

    double *tmp2_d;
    cudaMalloc((void**)&tmp2_d, sizeof(double)*SIZE*SIZE); 
    cudaMemcpy(tmp2_d, tmp1, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);
    free(tmp1);

    cudaMalloc((void**)&mat_out_d, sizeof(double)*SIZE*SIZE); 
   
    /********** Copy Kernels ************/
    grid.x = SIZE/BLOCK_X+1; 
    grid.y = (i+1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, tmp2_d, 0, 0, i+1, SIZE); 

    grid.x = (i+1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, tmp2_d, i+1, 0, SIZE, i+1); 
    /************************************/
 
    grid.x = (SIZE-i-1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1;
    submatrix<<<grid, block>>>( mat_out_d, tmp2_d, i); 

    cudaFree(tmp2_d);

    double *tmp2 = (double*)malloc(sizeof(double)*SIZE*SIZE);
    cudaMemcpy(tmp2, mat_out_d, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);
    cudaFree(mat_out_d);
   
    mat = tmp2; 
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);
#endif


  gettimeofday( &tv1, NULL);
  for( i = 0; i < SIZE-1; i++) {
    double mat_i_i = mat[i*SIZE+i];
  
    double *tmp1_d;
    cudaMalloc((void**)&tmp1_d, sizeof(double)*SIZE*SIZE); 
    cudaMemcpy(tmp1_d, mat, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);

    grid.x = 1/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    column2<<<grid, block>>>( tmp1_d, mat_i_i, i); 

    
    cudaMemcpy(mat, tmp1_d, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);
    cudaFree(tmp1_d);

    double *tmp2_d;
    cudaMalloc((void**)&tmp2_d, sizeof(double)*SIZE*SIZE); 
    cudaMemcpy(tmp2_d, mat, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&mat_out_d, sizeof(double)*SIZE*SIZE); 
   
    /********** Copy Kernels ************/
    grid.x = SIZE/BLOCK_X+1; 
    grid.y = (i+1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, tmp2_d, 0, 0, i+1, SIZE); 

    grid.x = (i+1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, tmp2_d, i+1, 0, SIZE, i+1); 
    /************************************/
 
    grid.x = (SIZE-i-1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1;
    submatrix<<<grid, block>>>( mat_out_d, tmp2_d, i); 

    cudaFree(tmp2_d);

    cudaMemcpy(mat, mat_out_d, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);
    cudaFree(mat_out_d);
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

#endif
   


#ifdef MEMOPT 
#if 0
  gettimeofday( &tv1, NULL);
  for( i = 0; i < SIZE-1; i++) {
    double mat_i_i = mat[i*SIZE+i];

    cudaMalloc((void**)&mat_in_d, sizeof(double)*SIZE*SIZE); 
    cudaMemcpy(mat_in_d, mat, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);
    free(mat);

    grid.x = 1/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    column2<<<grid, block>>>( mat_in_d, mat_i_i, i); 

    cudaMalloc((void**)&mat_out_d, sizeof(double)*SIZE*SIZE); 
   
    /********** Copy Kernels ************/
    grid.x = SIZE/BLOCK_X+1; 
    grid.y = (i+1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, mat_in_d, 0, 0, i+1, SIZE); 

    grid.x = (i+1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, mat_in_d, i+1, 0, SIZE, i+1); 
    /************************************/
 
    grid.x = (SIZE-i-1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    submatrix<<<grid, block>>>( mat_out_d, mat_in_d, i); 

    cudaFree(mat_in_d);

    mat = (double*)malloc(sizeof(double)*SIZE*SIZE);
    cudaMemcpy(mat, mat_out_d, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);
    cudaFree(mat_out_d); 
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);
#endif

  gettimeofday( &tv1, NULL);
  for( i = 0; i < SIZE-1; i++) {
    double mat_i_i = mat[i*SIZE+i];

    cudaMalloc((void**)&mat_in_d, sizeof(double)*SIZE*SIZE); 
    cudaMemcpy(mat_in_d, mat, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);

    grid.x = 1/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    column2<<<grid, block>>>( mat_in_d, mat_i_i, i); 

    cudaMalloc((void**)&mat_out_d, sizeof(double)*SIZE*SIZE); 
   
    /********** Copy Kernels ************/
    grid.x = SIZE/BLOCK_X+1; 
    grid.y = (i+1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, mat_in_d, 0, 0, i+1, SIZE); 

    grid.x = (i+1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, mat_in_d, i+1, 0, SIZE, i+1); 
    /************************************/
 
    grid.x = (SIZE-i-1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    submatrix<<<grid, block>>>( mat_out_d, mat_in_d, i); 

    cudaFree(mat_in_d);

    cudaMemcpy(mat, mat_out_d, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);
    cudaFree(mat_out_d); 
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

#endif

#ifdef EXPAR 

  cudaMalloc((void**)&mat_in_d, sizeof(double)*SIZE*SIZE); 
  cudaMemcpy(mat_in_d, mat, sizeof(double)*SIZE*SIZE, cudaMemcpyHostToDevice);

  gettimeofday( &tv1, NULL);
  for( i = 0; i < SIZE-1; i++) {

    grid.x = 1/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    column<<<grid, block>>>( mat_in_d, i); 

    cudaMalloc((void**)&mat_out_d, sizeof(double)*SIZE*SIZE); 

    /********** Copy Kernels ************/
    grid.x = SIZE/BLOCK_X+1; 
    grid.y = (i+1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, mat_in_d, 0, 0, i+1, SIZE); 

    grid.x = (i+1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1; 
    copy<<<grid, block>>>( mat_out_d, mat_in_d, i+1, 0, SIZE, i+1); 
    /************************************/

    grid.x = (SIZE-i-1)/BLOCK_X+1; 
    grid.y = (SIZE-i-1)/BLOCK_Y+1;
    submatrix<<<grid, block>>>( mat_out_d, mat_in_d, i); 

    cudaFree(mat_in_d);
    mat_in_d = mat_out_d;
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

  cudaMemcpy(mat, mat_in_d, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost);

#endif

  res = (int)mat[0];

  free(mat);

  return( res);
}



