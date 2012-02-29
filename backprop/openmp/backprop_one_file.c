#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

//#include "backprop.h"

#ifndef SIZE
#define SIZE 65536
#endif

#ifndef ITER 
#define ITER 500
#endif

#define ETA      0.3   //eta value
#define MOMENTUM 0.3   //momentum value

#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))


typedef struct {
  int input_n;                  /* number of input units */
  int hidden_n;                 /* number of hidden units */
  int output_n;                 /* number of output units */

  double *input_units;          /* the input units */
  double *hidden_units;         /* the hidden units */
  double *output_units;         /* the output units */

  double *hidden_delta;         /* storage for hidden unit error */
  double *output_delta;         /* storage for output unit error */

  double *target;               /* storage for target vector */

  double **input_weights;       /* weights from input to hidden layer */
  double **hidden_weights;      /* weights from hidden to output layer */

                                /*** The next two are for momentum ***/
  double **input_prev_weights;  /* previous change on input to hidden wgt */
  double **hidden_prev_weights; /* previous change on hidden to output wgt */
} BPNN;


/*** The squashing function.  Currently, it's a sigmoid. ***/

double squash(float x)
{
  double m;
  return (1.0 / (1.0 + exp(-x)));
}

/*** Allocate 1d array of doubles ***/

static double *alloc_1d_dbl(int n)
{
  double *new;

  new = (double *) malloc ((unsigned) (n * sizeof (double)));
  if (new == NULL) {
    printf("ALLOC_1D_DBL: Couldn't allocate array of doubles\n");
    return (NULL);
  }
  return (new);
}


/*** Allocate 2d array of doubles ***/

static double **alloc_2d_dbl(int m, int n)
{
  int i;
  double **new;

  new = (double **) malloc ((unsigned) (m * sizeof (double *)));
  if (new == NULL) {
    printf("ALLOC_2D_DBL: Couldn't allocate array of dbl ptrs\n");
    return (NULL);
  }

  for (i = 0; i < m; i++) {
    new[i] = alloc_1d_dbl(n);
  }

  return (new);
}

static void bpnn_randomize_weights(double **w, int m, int n)
{
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
     w[i][j] = (double) rand()/RAND_MAX;
    }
  }
}

static void bpnn_randomize_row(double *w, int m)
{
  int i;
  for (i = 0; i < m; i++) {
    w[i] = 0.1;
  }
}

static void bpnn_zero_weights(double **w, int m, int n)
{
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      w[i][j] = 0.0;
    }
  }
}

BPNN *bpnn_create(int n_in, int n_hidden, int n_out)
{
  BPNN *newnet;

  newnet = (BPNN *) malloc (sizeof (BPNN));

  newnet->input_n = n_in;
  newnet->hidden_n = n_hidden;
  newnet->output_n = n_out;
  newnet->input_units = alloc_1d_dbl(n_in);
  newnet->hidden_units = alloc_1d_dbl(n_hidden);
  newnet->output_units = alloc_1d_dbl(n_out);

  newnet->hidden_delta = alloc_1d_dbl(n_hidden);
  newnet->output_delta = alloc_1d_dbl(n_out);
  newnet->target = alloc_1d_dbl(n_out);

  newnet->input_weights = alloc_2d_dbl(n_in, n_hidden);
  newnet->hidden_weights = alloc_2d_dbl(n_hidden, n_out);

  newnet->input_prev_weights = alloc_2d_dbl(n_in, n_hidden);
  newnet->hidden_prev_weights = alloc_2d_dbl(n_hidden, n_out);

  bpnn_randomize_weights(newnet->input_weights, n_in, n_hidden);
  bpnn_randomize_weights(newnet->hidden_weights, n_hidden, n_out);
  bpnn_zero_weights(newnet->input_prev_weights, n_in, n_hidden);
  bpnn_zero_weights(newnet->hidden_prev_weights, n_hidden, n_out);
  bpnn_randomize_row(newnet->target, n_out);

  return (newnet);
}

static void load( BPNN *net)
{
  double *units;
  int nc, i;
  
  units = net->input_units;

  for (i = 1; i < SIZE; i++) {
    units[i] = (double) rand()/RAND_MAX ;
  }
}

static void bpnn_free(BPNN *net)
{
  int n1, n2, i;

  n1 = net->input_n;
  n2 = net->hidden_n;

  free((char *) net->input_units);
  free((char *) net->hidden_units);
  free((char *) net->output_units);

  free((char *) net->hidden_delta);
  free((char *) net->output_delta);
  free((char *) net->target);

  for (i = 0; i < n1; i++) {
    free((char *) net->input_weights[i]);
    free((char *) net->input_prev_weights[i]);
  }
  free((char *) net->input_weights);
  free((char *) net->input_prev_weights);

  for (i = 0; i < n2; i++) {
    free((char *) net->hidden_weights[i]);
    free((char *) net->hidden_prev_weights[i]);
  }
  free((char *) net->hidden_weights);
  free((char *) net->hidden_prev_weights);

  free((char *) net);
}

////////////////////////////////////////////////////////////////////////////////

//extern void bpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2);
//extern void bpnn_output_error(double *delta, double *target, double *output, int nj, double *err);
//extern void bpnn_hidden_error(double *delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err);
//extern void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly, double **w, double **oldw);

inline static void bpnn_layerforward(double *l1, double *l2, double **conn, int n1, int n2)
{
  double sum;
  int j, k;

  /*** Set up thresholding unit ***/
  //l1[0] = 1.0;

#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for shared(conn, n1, n2, l1) private(k, j) reduction(+: sum) schedule(static)
#endif 
  /*** For each unit in second layer ***/
  for (j = 0; j < n2; j++) {
    /*** Compute weighted sum of its inputs ***/
    sum = 0.0;
    for (k = 0; k < n1; k++) {	
      sum += conn[k][j] * l1[k]; 
    }
    l2[j] = squash(sum);
  }
}

inline static void bpnn_output_error(double *delta, double *target, double *output, int nj, double *err)  
{
  int j;
  double o, t, errsum;

  errsum = 0.0;
  for (j = 0; j < nj; j++) {
    o = output[j];
    t = target[j];
    delta[j] = o * (1.0 - o) * (t - o);
    //errsum += ABS(delta[j]);
  }
  //*err = errsum;
}

inline static void bpnn_hidden_error(double *delta_h, int nh, double *delta_o, int no, double **who, double *hidden, double *err)
{
  int j, k;
  double h, sum, errsum;

  errsum = 0.0;
  for (j = 0; j < nh; j++) {
    h = hidden[j];
    sum = 0.0;
    for (k = 0; k < no; k++) {
      sum += delta_o[k] * who[j][k];
    }
    delta_h[j] = h * (1.0 - h) * sum;
    //errsum += ABS(delta_h[j]);
  }
  //*err = errsum;
}

inline static void bpnn_adjust_weights(double *delta, int ndelta, double *ly, int nly, double **w, double **oldw)
{
  double new_dw;
  int k, j;

  //ly[0] = 1.0;

#ifdef OPEN
  omp_set_num_threads(NUM_THREAD);
  #pragma omp parallel for shared(oldw, w, delta) private(j, k, new_dw) firstprivate(ndelta, nly) 
#endif 
  for (k = 0; k < nly; k++) {
    for (j = 0; j < ndelta; j++) {
      new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k][j]));
      w[k][j] += new_dw;
      oldw[k][j] = new_dw;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Traning kernel
////////////////////////////////////////////////////////////////////////////////

void bpnn_train_kernel(BPNN *net, double *eo, double *eh)
{
  int in, hid, out;
  double out_err, hid_err, runtime;
  int i;  

  in =  SIZE;   // 65536
  hid = 16;      // 16
  out = 1;       // 1   
   
  struct timeval tv1,tv2;

  gettimeofday( &tv1, NULL);

  for( i = 0; i < ITER; i++) 
  {
    bpnn_layerforward(net->input_units, net->hidden_units,net->input_weights, in, hid);
    bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
    bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
    bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err); 
    bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);
    bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in, net->input_weights, net->input_prev_weights);
  }

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0 + tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0 + tv1.tv_usec/1000.0));
  printf("%f\n", runtime);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main()
{
  BPNN *net;
  int i, res;
  double out_err, hid_err;

  srand(7);

  // 16 is hidden and 1 and output
  net = bpnn_create(SIZE, 16, 1); // (16, 1 can not be changed)

  load(net);

  //entering the training kernel, only one iteration
  bpnn_train_kernel(net, &out_err, &hid_err);

  res = (int)(net->input_weights[0][0]+net->input_prev_weights[0][0]+net->hidden_weights[0][0]+net->hidden_prev_weights[0][0]);

  bpnn_free(net);

  return( res);
}



