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
#define BLOCK_X 16
#define BLOCK_Y 16

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


__global__ void upper_left(int *input_itemsets, int *reference, int max_rows, int max_cols, int i, int penalty)
{
   int idx, r, c;

   idx = blockIdx.x*blockDim.x+threadIdx.x; 

   if( idx >= i) return;

   r = i - idx;  
   c = i + 1 - r;  

   int base = r*max_cols+c;
   input_itemsets[base] 
		= maximum( input_itemsets[base-max_cols-1]+ reference[base], 
			   input_itemsets[base-1] - penalty, 
			   input_itemsets[base-max_cols] - penalty);
}

__global__ void lower_right(int *input_itemsets, int *reference, int max_rows, int max_cols, int i, int penalty)
{
   int idx, r, c;
  
   idx = blockIdx.x*blockDim.x+threadIdx.x; 

   if( idx >= i) return;

   r = max_rows-1-idx; 
   c = max_cols-i+idx;  
   

   int base = r*max_cols+c;
   input_itemsets[base] 
		= maximum( input_itemsets[base-max_cols-1]+ reference[base], 
			   input_itemsets[base-1] - penalty, 
			   input_itemsets[base-max_cols] - penalty);
}


__global__ void copy(int *dst, int *input_itemsets, int max_rows, int max_cols, int lb0, int lb1, int ub0, int ub1)
{
   int r, c;
  
   r = blockIdx.y*blockDim.y+threadIdx.y+lb0; 
   c = blockIdx.x*blockDim.x+threadIdx.x+lb1; 

   if( r >= ub0 || c >= ub1) return;

   int idx = r*max_cols+c; 
   dst[idx] = input_itemsets[idx];
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  int max_rows, max_cols, penalty;
  int *input_itemsets, *reference, *input_itemsets_d, *reference_d;
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
  cudaMalloc((void**)&input_itemsets_d, max_rows*max_cols*sizeof(int));
  cudaMalloc((void**)&reference_d, max_rows*max_cols*sizeof(int));
	
  srand (7);

  for (i = 0 ; i < max_cols; i++) {
    for (j = 0 ; j < max_rows; j++) {
      input_itemsets[i*max_cols+j] = rand()%SIZE;
    }
  }

  srand (2012);

  for (i = 0 ; i < max_cols; i++) {
    for (j = 0 ; j < max_rows; j++) {
      reference[i*max_cols+j] = rand()%20;
    }
  }


#ifdef LAO
  /* memopt+lao */

  cudaMemcpy(input_itemsets_d, input_itemsets, sizeof(int)*max_rows*max_cols, cudaMemcpyHostToDevice); 
  cudaMemcpy(reference_d, reference, sizeof(int)*max_rows*max_cols, cudaMemcpyHostToDevice); 

  gettimeofday( &tv1, NULL);
  int *tmp_d, *p_d;

  cudaMalloc((void**)&tmp_d, max_rows*max_cols*sizeof(int));

  for( i = 1; i < max_cols; i++) {

    {
      dim3 block(BLOCK_X,BLOCK_Y);
      dim3 grid(max_cols/BLOCK_X+1,(max_rows-i-1)/BLOCK_Y+1);
      copy<<<grid, block>>>( tmp_d, input_itemsets_d, max_rows, max_cols, i+1, 0, max_rows, max_cols);
    }

    {
      dim3 block(BLOCK_X,BLOCK_Y);
      dim3 grid(max_cols/BLOCK_X+1,1/BLOCK_Y+1);
      copy<<<grid, block>>>( tmp_d, input_itemsets_d, max_rows, max_cols, 0, 0, 1, max_cols);
    }

    {
      dim3 block(BLOCK_X,BLOCK_Y);
      dim3 grid((max_cols-i-1)/BLOCK_X+1,i/BLOCK_Y+1);
      copy<<<grid, block>>>( tmp_d, input_itemsets_d, max_rows, max_cols, 1, i+1, i+1, max_cols);
    }

    {
      dim3 block(BLOCK_X,BLOCK_Y);
      dim3 grid(1/BLOCK_X+1,i/BLOCK_Y+1);
      copy<<<grid, block>>>( tmp_d, input_itemsets_d, max_rows, max_cols, 1, 0, i+1, 1);
    }

    {
      //dim3 block(BLOCK_X,BLOCK_Y);
      //dim3 grid(i/BLOCK_X+1,i/BLOCK_Y+1);
      //upper_left_copy<<<grid, block>>>( tmp_d, input_itemsets_d, reference_d, max_rows, max_cols, i, penalty);
      dim3 block(BLOCK);
      dim3 grid(i/BLOCK+1);
      upper_left<<<grid, block>>>( tmp_d, reference_d, max_rows, max_cols, i, penalty);
    }
   
    p_d = input_itemsets_d;  
    input_itemsets_d = tmp_d;
    tmp_d = p_d;
  }

  for( i = 1; i < max_cols-1; i++) {

    {
      dim3 block(BLOCK_X,BLOCK_Y);
      dim3 grid(max_cols/BLOCK_X+1,0/BLOCK_Y+1);
      copy<<<grid, block>>>( tmp_d, input_itemsets_d, max_rows, max_cols, max_rows, 0, max_rows, max_cols);
    }

    {
      dim3 block(BLOCK_X,BLOCK_Y);
      dim3 grid(max_cols/BLOCK_X+1,(i+1)/BLOCK_Y+1);
      copy<<<grid, block>>>( tmp_d, input_itemsets_d, max_rows, max_cols, 0, 0, i+1, max_cols);
    }

    {
      dim3 block(BLOCK_X,BLOCK_Y);
      dim3 grid(0/BLOCK_X+1,(max_rows-i-1)/BLOCK_Y+1);
      copy<<<grid, block>>>( tmp_d, input_itemsets_d, max_rows, max_cols, i+1, max_cols, max_rows, max_cols);
    }

    {
      dim3 block(BLOCK_X,BLOCK_Y);
      dim3 grid((i+1)/BLOCK_X+1,(max_rows-i-1)/BLOCK_Y+1);
      copy<<<grid, block>>>( tmp_d, input_itemsets_d, max_rows, max_cols, i+1, 0, max_rows, i+1);
    }

    {
      //dim3 block(BLOCK_X,BLOCK_Y);
      //dim3 grid((max_cols-i-1)/BLOCK_X+1,(max_rows-i-1)/BLOCK_Y+1);
      //lower_right_copy<<<grid, block>>>( tmp_d, input_itemsets_d, reference_d, max_rows, max_cols, i, penalty);
      dim3 block(BLOCK);
      dim3 grid(i/BLOCK+1);
      lower_right<<<grid, block>>>( tmp_d, reference_d, max_rows, max_cols, i, penalty);
    }

    p_d = input_itemsets_d;  
    input_itemsets_d = tmp_d;
    tmp_d = p_d;
  }

  cudaThreadSynchronize();

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

  cudaMemcpy(input_itemsets, input_itemsets_d, sizeof(int)*max_rows*max_cols, cudaMemcpyDeviceToHost); 
#endif



#ifdef PRA 

  cudaMemcpy(input_itemsets_d, input_itemsets, sizeof(int)*max_rows*max_cols, cudaMemcpyHostToDevice); 
  cudaMemcpy(reference_d, reference, sizeof(int)*max_rows*max_cols, cudaMemcpyHostToDevice); 


  gettimeofday( &tv1, NULL);

  for( i = 1; i < max_cols; i++) {
/*
    input_itemsets = with {
                       ( [1,1] <= iv=[r,c] < [1+i,1+i]) {
                         if( r == (i - c + 1)) {
                           res = maximum( input_itemsets[r-1, c-1]+ reference[r, c], 
                                          input_itemsets[r, c-1] - penalty, 
			                  input_itemsets[r-1, c] - penalty);
                         }
                         else {
                           res = input_itemsets[r,c];
                         }
                       }:res;
                     }:modarray( input_itemsets);
*/
    dim3 block(BLOCK);
    dim3 grid(i/BLOCK+1);
    upper_left<<<grid, block>>>( input_itemsets_d, reference_d, max_rows, max_cols, i, penalty);
  }


  for( i = max_cols-2; i >= 1; i--) {
/*
    input_itemsets = with {
                       ( [1+i,1+i] <= iv=[r,c] < [max_rows,max_cols]) {
                         if( r == (max_cols - c + i)) { 
                           res = maximum( input_itemsets[r-1, c-1]+ reference[r, c], 
                                          input_itemsets[r, c-1] - penalty, 
			                  input_itemsets[r-1, c] - penalty);
                         }
                         else {
                           res = input_itemsets[r,c];
                         }
                       }:res;
                     }:modarray( input_itemsets);
*/
    dim3 block(BLOCK);
    dim3 grid(i/BLOCK+1);
    lower_right<<<grid, block>>>( input_itemsets_d, reference_d, max_rows, max_cols, i, penalty);
  }

  cudaThreadSynchronize();

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("Runtime: %f\n", runtime);

  cudaMemcpy(input_itemsets, input_itemsets_d, sizeof(int)*max_rows*max_cols, cudaMemcpyDeviceToHost); 

#endif


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


