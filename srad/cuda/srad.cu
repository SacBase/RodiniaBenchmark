#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef SIZE
#define SIZE 1024
#endif

#define R1     0
#define R2     127
#define C1     0
#define C2     127

#define ROWS     SIZE
#define COLS     SIZE
#define LAMBDA   0.5f
#define NITER    1

#define BLOCK_X 32 
#define BLOCK_Y 16 

void random_matrix(float *I, int rows, int cols);
void runTest( int argc, char** argv);

__global__ void
srad_cuda_1( float *E_C, 
	     float *W_C, 
	     float *N_C, 
	     float *S_C,
	     float * J_cuda, 
	     float * C_cuda, 
	     int cols, 
	     int rows, 
	     float q0sqr
) 
{

  //block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  //indices
  int index   = cols * BLOCK_Y * by + BLOCK_X * bx + cols * ty + tx;
  int index_n = cols * BLOCK_Y * by + BLOCK_X * bx + tx - cols;
  int index_s = cols * BLOCK_Y * by + BLOCK_X * bx + cols * BLOCK_Y + tx;
  int index_w = cols * BLOCK_Y * by + BLOCK_X * bx + cols * ty - 1;
  int index_e = cols * BLOCK_Y * by + BLOCK_X * bx + cols * ty + BLOCK_X;

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation
  __shared__ float temp[BLOCK_Y][BLOCK_X];
  __shared__ float temp_result[BLOCK_Y][BLOCK_X];

  __shared__ float north[BLOCK_Y][BLOCK_X];
  __shared__ float south[BLOCK_Y][BLOCK_X];
  __shared__ float  east[BLOCK_Y][BLOCK_X];
  __shared__ float  west[BLOCK_Y][BLOCK_X];

  //load data to shared memory
  north[ty][tx] = J_cuda[index_n]; 
  south[ty][tx] = J_cuda[index_s];
  if ( by == 0 ){
    north[ty][tx] = J_cuda[BLOCK_X * bx + tx]; 
  }
  else if ( by == gridDim.y - 1 ){
    south[ty][tx] = J_cuda[cols * BLOCK_Y * (gridDim.y - 1) + BLOCK_X * bx + cols * ( BLOCK_Y - 1 ) + tx];
  }
  __syncthreads();
 
  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];

  if ( bx == 0 ){
    west[ty][tx] = J_cuda[cols * BLOCK_Y * by + cols * ty]; 
  }
  else if ( bx == gridDim.x - 1 ){
    east[ty][tx] = J_cuda[cols * BLOCK_Y * by + BLOCK_X * ( gridDim.x - 1) + cols * ty + BLOCK_X-1];
  }
 
  __syncthreads();

  temp[ty][tx] = J_cuda[index];

  __syncthreads();

  jc = temp[ty][tx];

  if ( ty == 0 && tx == 0 ){ //nw
    n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
  }	    
  else if ( ty == 0 && tx == BLOCK_X-1 ){ //ne
    n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
  }
  else if ( ty == BLOCK_Y -1 && tx == BLOCK_X - 1){ //se
    n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx]  - jc;
  }
  else if ( ty == BLOCK_Y -1 && tx == 0 ){//sw
    n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
  }
  else if ( ty == 0 ){ //n
    n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
  }
  else if ( tx == BLOCK_X -1 ){ //e
    n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
  }
  else if ( ty == BLOCK_Y -1){ //s
    n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
  }
  else if ( tx == 0 ){ //w
    n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc; 
    e  = temp[ty][tx+1] - jc;
  }
  else{  //the data elements which are not on the borders 
    n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
  }

  g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

  l = ( n + s + w + e ) / jc;

  num  = (0.5f*g2) - ((1.0f/16.0f)*(l*l)) ;
  den  = 1.0f + (0.25f*l);
  qsqr = num/(den*den);

  // diffusion coefficent (equ 33)
  den = (qsqr-q0sqr) / (q0sqr * (1.0f+q0sqr)) ;
  c = 1.0f / (1.0f+den) ;

  // saturate diffusion coefficent
  if (c < 0.0f){temp_result[ty][tx] = 0.0f;}
  else if (c > 1.0f) {temp_result[ty][tx] = 1.0f;}
  else {temp_result[ty][tx] = c;}

  __syncthreads();

  C_cuda[index] = temp_result[ty][tx];
  E_C[index] = e;
  W_C[index] = w;
  S_C[index] = s;
  N_C[index] = n;
}

__global__ void
srad_cuda_2( float *E_C, 
	     float *W_C, 
	     float *N_C, 
	     float *S_C,	
	     float * J_cuda, 
	     float * C_cuda, 
	     int cols, 
	     int rows, 
	     float lambda,
	     float q0sqr
) 
{
  //block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  //indices
  int index   = cols * BLOCK_Y* by + BLOCK_X* bx + cols * ty + tx;
  int index_s = cols * BLOCK_Y* by + BLOCK_X* bx + cols * BLOCK_Y+ tx;
  int index_e = cols * BLOCK_Y* by + BLOCK_X* bx + cols * ty + BLOCK_X;
  float cc, cn, cs, ce, cw, d_sum;

  //shared memory allocation
  __shared__ float south_c[BLOCK_Y][BLOCK_X];
  __shared__ float  east_c[BLOCK_Y][BLOCK_X];

  __shared__ float c_cuda_temp[BLOCK_Y][BLOCK_X];
  __shared__ float c_cuda_result[BLOCK_Y][BLOCK_X];
  __shared__ float temp[BLOCK_Y][BLOCK_X];

  //load data to shared memory
  temp[ty][tx]      = J_cuda[index];

  __syncthreads();
	 
  south_c[ty][tx] = C_cuda[index_s];

  if ( by == gridDim.y - 1 ){
    south_c[ty][tx] = C_cuda[cols * BLOCK_Y* (gridDim.y - 1) + BLOCK_X * bx + cols * ( BLOCK_Y - 1 ) + tx];
  }
  __syncthreads();
	 
  east_c[ty][tx] = C_cuda[index_e];
	
  if ( bx == gridDim.x - 1 ){
    east_c[ty][tx] = C_cuda[cols * BLOCK_Y * by + BLOCK_X * ( gridDim.x - 1) + cols * ty + BLOCK_X-1];
  }
	 
  __syncthreads();
  
  c_cuda_temp[ty][tx]      = C_cuda[index];

  __syncthreads();

  cc = c_cuda_temp[ty][tx];

  if ( ty == BLOCK_Y -1 && tx == BLOCK_X - 1){ //se
    cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
  } 
  else if ( tx == BLOCK_X -1 ){ //e
    cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
  }
  else if ( ty == BLOCK_Y -1){ //s
    cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
  }
  else{ //the data elements which are not on the borders 
    cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
  }

  // divergence (equ 58)
  d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

  // image update (equ 61)
  c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

  __syncthreads();
              
  J_cuda[index] = c_cuda_result[ty][tx];
}

__global__ void
srad_cuda_1_noshr( float *E_C, 
	           float *W_C, 
	           float *N_C, 
	           float *S_C,
	           float * J_cuda, 
	           float * C_cuda, 
	           int cols, 
	           int rows, 
	           float q0sqr) 
{

  //block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;
 
  int idx_x = bx*blockDim.x+tx;
  int idx_y = by*blockDim.y+ty;

  int index = idx_y*cols+idx_x;
  int index_n = index-cols;  
  int index_s = index+cols;  
  int index_e = index+1;  
  int index_w = index-1;  

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;
  float res;

  jc = J_cuda[index];

  if( idx_y == 0) {
    n = jc - jc;
  } else {
    n = J_cuda[index_n]-jc;
  }

  if( idx_y == rows-1) {
    s = jc - jc;
  } else {
    s = J_cuda[index_s]-jc;
  }

  if( idx_x == 0) {
    w = jc - jc;
  } else {
    w = J_cuda[index_w]-jc;
  }

  if( idx_x == cols-1) {
    e = jc - jc;
  } else {
    e = J_cuda[index_e]-jc;
  }

  g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

  l = ( n + s + w + e ) / jc;

  num  = (0.5f*g2) - ((1.0f/16.0f)*(l*l)) ;
  den  = 1.0f + (0.25f*l);
  qsqr = num/(den*den);

  // diffusion coefficent (equ 33)
  den = (qsqr-q0sqr) / (q0sqr * (1.0f+q0sqr)) ;
  c = 1.0f / (1.0f+den) ;

  // saturate diffusion coefficent
  if (c < 0.0f){ res = 0.0f;}
  else if (c > 1.0f) { res = 1.0f;}
  else { res = c;}

  C_cuda[index] = res;
  E_C[index] = e;
  W_C[index] = w;
  S_C[index] = s;
  N_C[index] = n;
}

__global__ void
srad_cuda_2_noshr( float *E_C, 
	           float *W_C, 
	           float *N_C, 
	           float *S_C,	
	           float * J_cuda, 
	           float * C_cuda, 
	           int cols, 
	           int rows, 
	           float lambda,
	           float q0sqr) 
{
  //block id
  int bx = blockIdx.x;
  int by = blockIdx.y;

  //thread id
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int idx_x = bx*blockDim.x+tx;
  int idx_y = by*blockDim.y+ty;

  int index = idx_y*cols+idx_x;
  int index_s = index+cols;  
  int index_e = index+1;  

  float cc, cn, cs, ce, cw, d_sum;

  cn = C_cuda[index]; 
  cw = C_cuda[index];

  if( idx_y == rows-1) {
    cs = C_cuda[index];
  } else {
    cs = C_cuda[index_s];
  }

  if( idx_x == cols-1) {
    ce = C_cuda[index];
  } else {
    ce = C_cuda[index_e];
  }

  // divergence (equ 58)
  d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

  // image update (equ 61)
  J_cuda[index] = J_cuda[index] + 0.25 * lambda * d_sum;
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
    runTest( argc, argv);

    return( 0);
}


void
runTest( int argc, char** argv) 
{
  int rows, cols, size_I, size_R, niter = 10, iter;
  float *I, *J, lambda, q0sqr, sum, sum2, tmp, meanROI,varROI ;

  float *J_cuda;
  float *C_cuda;
  float *E_C, *W_C, *N_C, *S_C;

  unsigned int r1, r2, c1, c2;
  float *c;
    
  rows = ROWS;
  cols = COLS;
  if ((rows%16!=0) || (cols%16!=0)){
    fprintf(stderr, "rows and cols must be multiples of 16\n");
    exit(1);
  }
  r1 = R1;
  r2 = R2;
  c1 = C1;
  c2 = C2;	
  lambda = LAMBDA; 
  niter = NITER;

  size_I = cols * rows;
  size_R = (r2-r1+1)*(c2-c1+1);   

  I = (float *)malloc( size_I * sizeof(float) );
  J = (float *)malloc( size_I * sizeof(float) );
  c  = (float *)malloc(sizeof(float)* size_I) ;


  //Allocate device memory
  cudaMalloc((void**)& J_cuda, sizeof(float)* size_I);
  cudaMalloc((void**)& C_cuda, sizeof(float)* size_I);
  cudaMalloc((void**)& E_C, sizeof(float)* size_I);
  cudaMalloc((void**)& W_C, sizeof(float)* size_I);
  cudaMalloc((void**)& S_C, sizeof(float)* size_I);
  cudaMalloc((void**)& N_C, sizeof(float)* size_I);

  //printf("Randomizing the input matrix\n");
  //Generate a random matrix
  random_matrix(I, rows, cols);

  for (int k = 0;  k < size_I; k++ ) {
    J[k] = (float)expf(I[k]) ;
  }

  //printf("Start the SRAD main loop\n");

  for (iter=0; iter< niter; iter++){     
    sum=0; sum2=0;
    for (int i=r1; i<=r2; i++) {
      for (int j=c1; j<=c2; j++) {
        tmp   = J[i * cols + j];
        sum  += tmp ;
        sum2 += tmp*tmp;
      }
    }
    meanROI = sum / size_R;
    varROI  = (sum2 / size_R) - meanROI*meanROI;
    q0sqr   = varROI / (meanROI*meanROI);

    //Copy data from main memory to device memory
    cudaMemcpy(J_cuda, J, sizeof(float) * size_I, cudaMemcpyHostToDevice);

    //Run kernels
#ifdef NOSHR // these are kernels without utilising shared memory
    //Currently the input size must be divided by 16 - the block size
    int block_x = cols/BLOCK_X;
    int block_y = rows/BLOCK_Y;

    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    dim3 dimGrid(block_x, block_y);

    srad_cuda_1_noshr<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
    srad_cuda_2_noshr<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 
#else
    //Currently the input size must be divided by 16 - the block size
    int block_x = cols/BLOCK_X ;
    int block_y = rows/BLOCK_Y ;

    dim3 dimBlock(BLOCK_X, BLOCK_Y);
    dim3 dimGrid(block_x, block_y);

    srad_cuda_1<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, q0sqr); 
    srad_cuda_2<<<dimGrid, dimBlock>>>(E_C, W_C, N_C, S_C, J_cuda, C_cuda, cols, rows, lambda, q0sqr); 
#endif

    //Copy data from device memory to main memory
    cudaMemcpy(J, J_cuda, sizeof(float) * size_I, cudaMemcpyDeviceToHost);
  }

  cudaThreadSynchronize();

#ifdef OUTPUT
    //Printing output	
    //printf("Printing Output:\n"); 
    for( int i = 0 ; i < rows ; i++){
      for ( int j = 0 ; j < cols ; j++){
         printf("%.5f\n", J[i * cols + j]); 
      }	
    }
#endif 

  //printf("Computation Done\n");

  free(I);
  free(J);
  cudaFree(C_cuda);
  cudaFree(J_cuda);
  cudaFree(E_C);
  cudaFree(W_C);
  cudaFree(N_C);
  cudaFree(S_C);
  free(c);
}


void random_matrix(float *I, int rows, int cols)
{
  srand(7);
	
  for( int i = 0 ; i < rows ; i++){
    for ( int j = 0 ; j < cols ; j++){
      I[i * cols + j] = (float)rand()/(float)RAND_MAX ;
//      I[i * cols + j] = (float)(i+j)/(float)(2147483647);
    }
  }
}

