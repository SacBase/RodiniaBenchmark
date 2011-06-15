#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "backprop.h"

////////////////////////////////////////////////////////////////////////////////

extern void bpnn_layerforward(float *l1, float *l2, float **conn, int n1, int n2);

extern void bpnn_output_error(float *delta, float *target, float *output, int nj, float *err);

extern void bpnn_hidden_error(float *delta_h, int nh, float *delta_o, int no, float **who, float *hidden, float *err);

extern void bpnn_adjust_weights(float *delta, int ndelta, float *ly, int nly, float **w, float **oldw);


extern int setup(int argc, char** argv);

extern float **alloc_2d_dbl(int m, int n);

extern float squash(float x);

extern char *strcpy();
extern void exit();

int layer_size = 0;

double gettime() 
{
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

void bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
  int in, hid, out;
  float out_err, hid_err;
  int i, j;  

/* 
  in = net->input_n;   // 65536
  hid = net->hidden_n; // 16
  out = net->output_n; // 1   
*/

  in =  65536;   // 65536
  hid = 16;      // 16
  out = 1;       // 1   
   
#ifdef VERBOSE
  printf("Performing CPU computation\n");
#endif

  struct timeval tv1,tv2;

  int iter;
  for( iter = 0; iter < ITER; iter++) {
    gettimeofday( &tv1, NULL);

    bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);

  #ifdef OUTPUT
/*
    for( i = 0; i < hid+1; i++) {
      printf("%f\n", net->hidden_units[i]);
    }
    for( i = 0; i < out+1; i++) {
      printf("%f\n", net->output_units[i]);
    }
*/
  #endif

    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err); 

  #ifdef OUTPUT
/*
    printf("output error=%f, hidden error=%f\n", out_err, hid_err);
*/
  #endif
   
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

  #ifdef OUTPUT
/*
    for( i = 0; i < hid+1; i++) {
      for( j = 0; j < out+1; j++) {
	printf("%f", net->hidden_weights[i][j]);
      }
      printf("\n");
    }
    for( i = 0; i < hid+1; i++) {
      for( j = 0; j < out+1; j++) {
	printf("%f", net->hidden_prev_weights[i][j]);
      }
      printf("\n");
    }
*/
  #endif

    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);

    gettimeofday( &tv2, NULL);
    double runtime = ((tv2.tv_sec*1000.0 + tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0 + tv1.tv_usec/1000.0));
    printf("Back propagation runtime(1 iteration in milliseconds): %f\n", runtime);
  }

#ifdef OUTPUT
  for( i = 0; i < hid+1; i++) {
    for( j = 0; j < out+1; j++) {
      printf("%f\n", net->hidden_weights[i][j]);
    }
  }

  for( i = 0; i < hid+1; i++) {
    for( j = 0; j < out+1; j++) {
      printf("%f\n", net->hidden_prev_weights[i][j]);
    }
  }

  for( i = 0; i < in+1; i++) {
    for( j = 0; j < hid+1; j++) {
      printf("%f\n", net->input_weights[i][j]);
    }
  }
  for( i = 0; i < in+1; i++) {
    for( j = 0; j < hid+1; j++) {
      printf("%f\n", net->input_prev_weights[i][j]);
    }
  }
#endif
}

void load( BPNN *net)
{
  float *units;
  int nr, nc, imgsize, i, j, k;

  nr = layer_size;
  
  imgsize = nr * nc;
  units = net->input_units;

  k = 1;
  for (i = 0; i < nr; i++) {
    units[k] = (float) rand()/RAND_MAX ;
    k++;
  }
}

void backprop_face()
{
  BPNN *net;
  int i;
  float out_err, hid_err;

  // 16 is hidden and 1 and output
  net = bpnn_create(layer_size, 16, 1); // (16, 1 can not be changed)

#ifdef VERBOSE
  printf("Input layer size : %d\n", layer_size);
#endif
  load(net);

  //entering the training kernel, only one iteration

#ifdef VERBOSE
  printf("Starting training kernel\n");
#endif
  bpnn_train_kernel(net, &out_err, &hid_err);
  bpnn_free(net);

#ifdef VERBOSE
  printf("Training done\n");
#endif
}

int setup(int argc, char *argv[])
{
  if(argc!=2){
    fprintf(stderr, "usage: backprop <num of input elements>\n");
    exit(0);
  }

  //layer_size = atoi(argv[1]);
  layer_size = 65536;
  
  int seed;

  seed = 7;   
  bpnn_initialize(seed); // in file backprop.c
  backprop_face();

  exit(0);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
  setup(argc, argv); // in file facetrain.c
}


