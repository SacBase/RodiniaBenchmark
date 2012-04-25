#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>


#define LIMIT -999

#ifndef SIZE
#define SIZE 4096
#endif

#define BLOCK 256

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
inline __device__ int maximum( int a, int b, int c)
{
  int m = a;
  
  if( m > b ) m = b;
  if( m > c ) m = c; 

  return( m);
}


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};


__global__ void upper_left(int *input_itemsets, int *reference, int *tmp, int max_rows, int max_cols, int i, int penalty)
{



   int r, c;
  
   r = i; 
   c = blockIdx.x*blockDim.x+threadIdx.x+1; 

   if( c >= i) return;

   tmp[r*max_cols+c] = maximum( tmp[(r-2)*max_cols+c-1] + reference[(r-c)*max_cols+c], 
		                tmp[(r-1)*max_cols+c-1] - penalty, 
		                tmp[(r-1)*max_cols+c]   - penalty);
}

__global__ void middle(int *input_itemsets, int *reference, int *tmp, int max_rows, int max_cols, int i, int penalty)
{

   int r, c;
  
   r = max_rows; 
   c = blockIdx.x*blockDim.x+threadIdx.x; 

   if( c >= (max_cols - 1)) return;

   tmp[r*max_cols+c] = maximum( tmp[(r-2)*max_cols+c]   + reference[(r-1-c)*max_cols+c+1], 
		                tmp[(r-1)*max_cols+c]   - penalty, 
		                tmp[(r-1)*max_cols+c+1] - penalty);
}


__global__ void lower_right(int *input_itemsets, int *reference, int *tmp, int max_rows, int max_cols, int i, int penalty)
{

   int r, c;
  
   r = i; 
   c = blockIdx.x*blockDim.x+threadIdx.x; 

   if( c >= (max_cols-(i-max_rows+1))) return;

   tmp[r*max_cols+c] = maximum( tmp[(r-2)*max_cols+c+1]+ reference[(max_rows-1-c)*max_cols+c+(i-max_rows+1)], 
		                tmp[(r-1)*max_cols+c] - penalty, 
		                tmp[(r-1)*max_cols+c+1] - penalty);
}

__global__ void marshalling1(int *input_itemsets, int *tmp, int max_rows, int max_cols)
{
  int i, j;
  
  i = blockIdx.y*blockDim.y+threadIdx.y; 
  j = blockIdx.x*blockDim.x+threadIdx.x; 

  if( i >= max_rows || j >= max_cols) return;

  if( j <= i) {
    tmp[i*max_cols+j] = input_itemsets[(i-j)*max_cols+j];
  }
  else {
    tmp[i*max_cols+j] = 0;
  } 
}

__global__ void marshalling2(int *input_itemsets, int *tmp, int max_rows, int max_cols)
{
  int i, j;
  
  i = blockIdx.y*blockDim.y+threadIdx.y+max_rows; 
  j = blockIdx.x*blockDim.x+threadIdx.x; 

  if( i >= max_rows*2-1 || j >= max_cols) return;

  if( j < max_cols-(i-max_rows+1)) {
    tmp[i*max_cols+j] = input_itemsets[(max_rows-1-j)*max_cols+j+1+(i-max_rows)];
  }
  else {
    tmp[i*max_cols+j] = 0;
  } 
}

__global__ void unmarshalling(int *input_itemsets, int *tmp, int max_rows, int max_cols)
{
  int i, j;
  
  i = blockIdx.y*blockDim.y+threadIdx.y; 
  j = blockIdx.x*blockDim.x+threadIdx.x; 

  if( i >= max_rows || j >= max_cols) return;

  if( (i+j) < max_rows) {
    input_itemsets[i*max_cols+j] = tmp[(i+j)*max_cols+j];
  }
  else {
    input_itemsets[i*max_cols+j] = tmp[(i+j)*max_cols+j-(i+j-max_rows+1)];
  } 

}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  int max_rows, max_cols, penalty;
  int *input_itemsets, *reference, *input_itemsets_d, *reference_d, *tmp, *tmp_d;
  int i,j;
  struct timeval tv1, tv2;
  double runtime;

  penalty = 10;
    
  // the lengths of the two sequences should be able to divided by 16.
  // And at current stage  max_rows needs to equal max_cols
  max_rows = SIZE + 1;
  max_cols = SIZE + 1;


  reference = (int *)malloc( max_rows * max_cols * sizeof(int) );
  input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
  tmp = (int *)malloc( (2*max_rows-1) * max_cols * sizeof(int) );
  cudaMalloc((void**)&input_itemsets_d, max_rows*max_cols*sizeof(int));
  cudaMalloc((void**)&reference_d, max_rows*max_cols*sizeof(int));
  cudaMalloc((void**)&tmp_d, (2*max_rows-1)*max_cols*sizeof(int));
	
  srand (7);

  for (i = 0 ; i < max_rows; i++) {
    for (j = 0 ; j < max_cols; j++) {
      input_itemsets[i*max_cols+j] = rand()%SIZE;
    }
  }

  srand (2012);

  for (i = 0 ; i < max_rows; i++) {
    for (j = 0 ; j < max_cols; j++) {
      reference[i*max_cols+j] = rand()%20;
    }
  }

  cudaMemcpy(input_itemsets_d, input_itemsets, sizeof(int)*max_rows*max_cols, cudaMemcpyHostToDevice); 
  cudaMemcpy(reference_d, reference, sizeof(int)*max_rows*max_cols, cudaMemcpyHostToDevice); 
  cudaMemcpy(tmp_d, tmp, sizeof(int)*(2*max_rows-1)*max_cols, cudaMemcpyHostToDevice); 

  gettimeofday( &tv1, NULL);

  /* Data marshalling */
  dim3 block1(16,16);
  dim3 grid1(max_cols/16+1, max_rows/16+1);
  marshalling1<<<grid1, block1>>>( input_itemsets_d, tmp_d, max_rows, max_cols);

  dim3 block2(16,16);
  dim3 grid2(max_cols/16+1, (max_rows-1)/16+1);
  marshalling2<<<grid2, block2>>>( input_itemsets_d, tmp_d, max_rows, max_cols);

/*
  for (i = 0 ; i < max_rows; i++) {
    for (j = 0 ; j < max_cols; j++) {
      if( j <= i) {
        tmp[i*max_cols+j] = input_itemsets[(i-j)*max_cols+j];
      }
      else {
        tmp[i*max_cols+j] = 0;
      } 
    }
  }

  for (i = max_rows ; i < max_rows*2-1; i++) {
    for (j = 0 ; j < max_cols; j++) {
      if( j < max_cols-(i-max_rows+1)) {
        tmp[i*max_cols+j] = input_itemsets[(max_rows-1-j)*max_cols+j+1+(i-max_rows)];
      }
      else {
        tmp[i*max_cols+j] = 0;
      } 
    }
  }
*/

  for( i = 2; i < max_cols; i++) {

    dim3 block(BLOCK);
    dim3 grid((i-1)/BLOCK+1);
    upper_left<<<grid, block>>>( input_itemsets_d, reference_d, tmp_d, max_rows, max_cols, i, penalty);
  }

  dim3 block3(BLOCK);
  dim3 grid3((max_cols-1)/BLOCK+1);
  middle<<<grid3, block3>>>( input_itemsets_d, reference_d, tmp_d, max_rows, max_cols, i, penalty);

  for( i = max_rows+1; i < 2*max_rows-1; i++) {

    dim3 block(BLOCK);
    dim3 grid((max_cols-(i-max_rows+1))/BLOCK+1);
    lower_right<<<grid, block>>>( input_itemsets_d, reference_d, tmp_d, max_rows, max_cols, i, penalty);
  }

  /* Unmarshalling data */
  dim3 block4(16,16);
  dim3 grid4(max_cols/16+1, max_rows/16+1);
  unmarshalling<<<grid4, block4>>>( input_itemsets_d, tmp_d, max_rows, max_cols);

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("Computation runtime: %f\n", runtime);

  cudaMemcpy(input_itemsets, input_itemsets_d, sizeof(int)*max_rows*max_cols, cudaMemcpyDeviceToHost); 


/*
  for (i = 0; i < max_rows; i++) {
    for (j = 0 ; j < max_cols; j++) {
      if( (i+j) < max_rows) {
        input_itemsets[i*max_cols+j] = tmp[(i+j)*max_cols+j];
      }
      else {
        input_itemsets[i*max_cols+j] = tmp[(i+j)*max_cols+j-(i+j-max_rows+1)];
      } 
    }
  }
*/
 

#ifdef OUTPUT 
  for( i = 0; i < max_rows; i++) {
    for( j = 0; j < max_cols; j++) {
      printf("%d ", input_itemsets[i*max_cols+j]);
    }
    printf("\n");
  }
  return( 0);
#else
  printf("%d\n", input_itemsets[0]);
  return(0);
#endif
}


