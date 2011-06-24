/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"

static int do_verify = 0;
static int do_shared = 0;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

extern void lud_cuda(float *d_m, int matrix_dim, int do_shared);

int main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default matrix_dim */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *d_m, *mm;

  while ((opt = getopt_long(argc, argv, "::v:ms:i:", 
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'i':
	input_file = optarg;
	break;
      case 'v':
	do_verify = 1;
	break;
      case 'm':
	do_shared = 1;
	break;
      case 's':
	matrix_dim = atoi(optarg);
	fprintf(stderr, "Currently not supported, use -i instead\n");
	fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
	exit(EXIT_FAILURE);
      case '?':
	fprintf(stderr, "invalid option\n");
	break;
      case ':':
	fprintf(stderr, "missing argument\n");
	break;
      default:
	fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
		argv[0]);
	exit(EXIT_FAILURE);
    }
  }
  
  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  } else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  }

  if (do_verify){
    printf("Before LUD\n");
    print_matrix(m, matrix_dim);
    matrix_duplicate(m, &mm, matrix_dim);
  }

  cudaMalloc((void**)&d_m, matrix_dim*matrix_dim*sizeof(float));

  cudaMemcpy(d_m, m, matrix_dim*matrix_dim*sizeof(float), cudaMemcpyHostToDevice);
  lud_cuda(d_m, matrix_dim, do_shared);
  cudaMemcpy(m, d_m, matrix_dim*matrix_dim*sizeof(float), cudaMemcpyDeviceToHost);

#ifdef OUTPUT
  int i, j;
  for( i = 0; i < matrix_dim; i++) {
    for( j = 0; j < matrix_dim; j++) {
      printf("[%d %d]:%f\n", i, j, m[i*matrix_dim+j]);
    }
    printf("\n");
  }
#else
  printf("[0 0 ]:%f\n", m[0]);
#endif

  cudaFree(d_m);

  if (do_verify){
    printf("After LUD\n");
    print_matrix(m, matrix_dim);
    printf(">>>Verify<<<<\n");
    lud_verify(mm, m, matrix_dim); 
    free(mm);
  }

  free(m);

  return EXIT_SUCCESS;
}
/* ----------  end of function main  ---------- */

