#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>


#define LIMIT -999
//#define OPEN

#ifndef NUM_THREAD
#define NUM_THREAD 4
#endif

#ifndef SIZE
#define SIZE 4096
#endif

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
inline int maximum( int a, int b, int c)
{
  int k;
  if( a <= b) {
    k = b;
  } 
  else { 
    k = a;
  }

  if( k <=c) {
    return(c);
  }
  else {
    return(k);
  }
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


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv) 
{
  int max_rows, max_cols, penalty,idx, index;
  int *input_itemsets, *referrence;
  int *matrix_cuda, *matrix_cuda_out, *referrence_cuda;
  int size;
  int omp_num_threads;
  int i,j;
  struct timeval tv1, tv2;
  double runtime;
    
  // the lengths of the two sequences should be able to divided by 16.
  // And at current stage  max_rows needs to equal max_cols
  max_rows = SIZE + 1;
  max_cols = SIZE + 1;
  omp_num_threads = NUM_THREAD; 
  penalty = 10;


  referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
  input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
	
  if (!input_itemsets) {
    fprintf(stderr, "error: can not allocate memory");
  }

  srand (7);

  for (i = 0 ; i < max_cols; i++) {
    for (j = 0 ; j < max_rows; j++) {
      input_itemsets[i*max_cols+j] = 0;
    }
  }

  //printf("Start Needleman-Wunsch\n");

  for( i=1; i< max_rows ; i++) {    //please define your own sequence. 
    input_itemsets[i*max_cols] = rand() % 10 + 1;
  }
  
  for( j=1; j< max_cols ; j++) {    //please define your own sequence.
    input_itemsets[j] = rand() % 10 + 1;
  }

  for (i = 1 ; i < max_rows; i++) {
    for (j = 1 ; j < max_cols; j++) {
      referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
    }
  }

  for( i = 1; i< max_rows ; i++) {
    input_itemsets[i*max_cols] = -i * penalty;
  }

  for( j = 1; j< max_cols ; j++) {
    input_itemsets[j] = -j * penalty;
  }


#if V == 1

  gettimeofday( &tv1, NULL);

  /* Compute top-left matrix.
   * It includes the center diagonal line with 2048 elements 
   */
  for( i = 0 ; i < max_cols-1 ; i++) {
    #ifdef OPEN
    omp_set_num_threads(omp_num_threads);
    #pragma omp parallel for shared(input_itemsets) firstprivate(i,max_cols,penalty) private(idx, index) 
    #endif 
    /* This loop goes from to top right to lower left along each diagonal line
     * Of course, if it gets parallelized, the order doesn't matter 
     */
    for( idx = 0 ; idx <= i ; idx++) {
      index = (idx + 1) * max_cols + (i + 1 - idx);
      input_itemsets[index]= maximum( input_itemsets[index-1-max_cols] + referrence[index], 
                                      input_itemsets[index-1]          - penalty, 
				      input_itemsets[index-max_cols]   - penalty);
    }
  }

  //Compute bottom-right matrix 
  for( i = max_cols - 3 ; i >= 0 ; i--) {
    #ifdef OPEN	
    omp_set_num_threads(omp_num_threads);
    #pragma omp parallel for shared(input_itemsets) firstprivate(i,max_cols,penalty) private(idx, index) 
    #endif    
    /* This loop goes from to lower left to top right along each diagonal line  
     * Of course, if it gets parallelized, the order doesn't matter 
     */ 
    for( idx = 0 ; idx <= i ; idx++) {
      index =  ( max_cols - idx - 1 ) * max_cols + idx + max_cols - i - 1 ;
      input_itemsets[index]= maximum( input_itemsets[index-1-max_cols]+ referrence[index], 
                                      input_itemsets[index-1]         - penalty, 
	                              input_itemsets[index-max_cols]  - penalty);
    } 
  }
  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

#elif V == 2

  int r, c;

  gettimeofday( &tv1, NULL);

  /* Upper left */
  for( i = 1; i < max_cols; i++) {
    for( r = 1; r < 1+i; r++) {
      for( c = 1; c < 1+i; c++) {
	if( r == (i - c + 1)) {
	  input_itemsets[r*max_cols+c] = maximum( input_itemsets[(r-1)*max_cols+(c-1)]+ referrence[r*max_cols+c], 
				       input_itemsets[r*max_cols+(c-1)] - penalty, 
				       input_itemsets[(r-1)*max_cols+c] - penalty);
	}
      }
    }
  }

  /* Lower right */
  for( i = 1; i < max_cols-1; i++) {
    for( r = 1+i; r < max_rows; r++) {
      for( c = 1+i; c < max_cols; c++) {
        if( r == (max_cols - c + i)) { 
	  input_itemsets[r*max_cols+c] = maximum( input_itemsets[(r-1)*max_cols+(c-1)]+ referrence[r*max_cols+c], 
	                               input_itemsets[r*max_cols+(c-1)] - penalty, 
	                               input_itemsets[(r-1)*max_cols+c] - penalty);
	 }
      }
    }
  }
  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

#else
  int r, c;
  int *tmp;

  gettimeofday( &tv1, NULL);

  /* Upper left */
  for( i = 1; i < max_cols; i++) {
    tmp = (int *)malloc( max_rows * max_cols * sizeof(int) );

    for( r = i+1; r < max_rows; r++) {
      for( c = 0; c < max_cols; c++) {
        tmp[r*max_cols+c] = input_itemsets[r*max_cols+c]; 
      }
    }
    for( r = 0; r < 1; r++) {
      for( c = 0; c < max_cols; c++) {
        tmp[r*max_cols+c] = input_itemsets[r*max_cols+c];
      }
    }
    for( r = 1; r < i+1; r++) {
      for( c = i+1; c < max_cols; c++) {
        tmp[r*max_cols+c] = input_itemsets[r*max_cols+c]; 
      }
    }
    for( r = 1; r < i+1; r++) {
      for( c = 0; c < 1; c++) {
        tmp[r*max_cols+c] = input_itemsets[r*max_cols+c];
      }
    }
    for( r = 1; r < 1+i; r++) {
      for( c = 1; c < 1+i; c++) {
	if( r == (i - c + 1)) {
	  tmp[r*max_cols+c] = maximum( input_itemsets[(r-1)*max_cols+(c-1)]+ referrence[r*max_cols+c], 
				       input_itemsets[r*max_cols+(c-1)] - penalty, 
				       input_itemsets[(r-1)*max_cols+c] - penalty);
	}
        else {
          tmp[r*max_cols+c] = input_itemsets[r*max_cols+c]; 
        }
      }
    }
    free(input_itemsets); 
    input_itemsets = tmp;
  }

  /* Lower right */
  for( i = 1; i < max_cols-1; i++) {
    tmp = (int *)malloc( max_rows * max_cols * sizeof(int) );

    for( r = max_rows; r < max_rows; r++) {
      for( c = 0; c < max_cols; c++) {
        tmp[r*max_cols+c] = input_itemsets[r*max_cols+c]; // Empty partition! 
      }
    }
    for( r = 0; r < i+1; r++) {
      for( c = 0; c < max_cols; c++) {
        tmp[r*max_cols+c] = input_itemsets[r*max_cols+c];
      }
    }
    for( r = i+1; r < max_rows; r++) {
      for( c = max_cols; c < max_cols; c++) {
        tmp[r*max_cols+c] = input_itemsets[r*max_cols+c]; // Empty partition! 
      }
    }
    for( r = i+1; r < max_rows; r++) {
      for( c = 0; c < i+1; c++) {
        tmp[r*max_cols+c] = input_itemsets[r*max_cols+c];
      }
    }
    for( r = 1+i; r < max_rows; r++) {
      for( c = 1+i; c < max_cols; c++) {
        if( r == (max_cols - c + i)) { 
	  tmp[r*max_cols+c] = maximum( input_itemsets[(r-1)*max_cols+(c-1)]+ referrence[r*max_cols+c], 
	                               input_itemsets[r*max_cols+(c-1)] - penalty, 
	                               input_itemsets[(r-1)*max_cols+c] - penalty);
	 }
         else {
           tmp[r*max_cols+c] = input_itemsets[r*max_cols+c]; 
         }
      }
    }

    free(input_itemsets); 
    input_itemsets = tmp;
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

#endif



#ifdef OUTPUT 
  for( i = 0; i < max_rows; i++) {
    for( j = 0; j < max_cols; j++) {
      printf("%d ", input_itemsets[i*max_cols+j]);
    }
    printf("\n");
  }
#else
  //printf("%d\n", input_itemsets[0]);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  runTest( argc, argv);

  return(0); 
}

