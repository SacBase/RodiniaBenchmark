#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

//using namespace std;

#define STR_SIZE 256

#ifndef SIZE
#define SIZE 1024
#endif

#ifndef ITER 
#define ITER 200
#endif

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
//#define MAX_PD	      (3.0e6)
#define MAX_PD	      3000000.0
/* required precision in degrees	*/
#define PRECISION     0.001
//#define SPEC_HEAT_SI  1.75e6
#define SPEC_HEAT_SI  1750000.0
#define K_SI          100
/* capacitance fitting factor	*/
#define FACTOR_CHIP   0.5

//#define OPEN
#define NUM_THREAD 4

/* chip parameters	*/
double t_chip = 0.0005;
double chip_height = 0.016;
double chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
double amb_temp = 80.0;

int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
inline
void single_iteration( double *result, double *temp, double *power, int row, int col,
                       double Rx_1, double Ry_1, double Rz_1,  double step_div_Cap)
{
  double delta;
  int r, c;

#ifdef OPEN
  omp_set_num_threads(num_omp_threads);
  #pragma omp parallel for shared(power, temp,result) private(r, c, delta) firstprivate(row, col) schedule(static)
#endif

  for (r = 0; r < row; r++) {
    for (c = 0; c < col; c++) {
      /*	Corner 1	*/
      if ( (r == 0) && (c == 0) ) {
	delta = (step_div_Cap) * (power[0] +
		(temp[1] - temp[0]) * Rx_1 +
		(temp[col] - temp[0]) * Ry_1 +
		(amb_temp - temp[0]) * Rz_1);
      }	/*	Corner 2	*/
      else if ((r == 0) && (c == col-1)) {
	delta = (step_div_Cap) * (power[c] +
		(temp[c-1] - temp[c]) * Rx_1 +
		(temp[c+col] - temp[c]) * Ry_1 +
		(amb_temp - temp[c]) * Rz_1);
      }	/*      Corner 3	*/
      else if ((r == row-1) && (c == col-1)) {
	delta = (step_div_Cap) * (power[r*col+c] + 
		(temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
		(temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
		(amb_temp - temp[r*col+c]) * Rz_1);					
      }	/*	Corner 4	*/
      else if ((r == row-1) && (c == 0)) {
	delta = (step_div_Cap) * (power[r*col] + 
		(temp[r*col+1] - temp[r*col]) * Rx_1 + 
		(temp[(r-1)*col] - temp[r*col]) * Ry_1 + 
		(amb_temp - temp[r*col]) * Rz_1);
      }	/*	Edge 1	*/
      else if (r == 0) {
	delta = (step_div_Cap) * (power[c] + 
		(temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 + 
		(temp[col+c] - temp[c]) * Ry_1 + 
		(amb_temp - temp[c]) * Rz_1);
      }	/*	Edge 2	*/
      else if (c == col-1) {
	delta = (step_div_Cap) * (power[r*col+c] + 
		(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 + 
		(temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
		(amb_temp - temp[r*col+c]) * Rz_1);
      }	/*	Edge 3	*/
      else if (r == row-1) {
	delta = (step_div_Cap) * (power[r*col+c] + 
		(temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) * Rx_1 + 
		(temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
		(amb_temp - temp[r*col+c]) * Rz_1);
      }	/*	Edge 4	*/
      else if (c == 0) {
	delta = (step_div_Cap) * (power[r*col] + 
		(temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 + 
		(temp[r*col+1] - temp[r*col]) * Rx_1 + 
		(amb_temp - temp[r*col]) * Rz_1);
      }	/*	Inside the chip	*/
      else {
	delta = (step_div_Cap) * (power[r*col+c] + 
		(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 + 
		(temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) * Rx_1 + 
		(amb_temp - temp[r*col+c]) * Rz_1);
      }
      
      /*	Update Temperatures	*/
      result[r*col+c] = temp[r*col+c]+ delta;
    }
  }

#ifdef OPEN
  omp_set_num_threads(num_omp_threads);
  #pragma omp parallel for shared(result, temp) private(r, c) schedule(static)
#endif
  for (r = 0; r < row; r++) {
    for (c = 0; c < col; c++) {
      temp[r*col+c] = result[r*col+c];
    }
  }
}

/* 
 * Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(double *result, int num_iterations, double *temp, double *power, int row, int col) 
{
  int i;
  double runtime;

  double grid_height = chip_height / row;
  double grid_width = chip_width / col;

  double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  double Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  double Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  double Rz = t_chip / (K_SI * grid_height * grid_width);
  double Rx_1 = 1.0/Rx; 
  double Ry_1 = 1.0/Ry;
  double Rz_1 = 1.0/Rz;

  double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  double step = PRECISION / max_slope;
  double t;

  double step_div_Cap = step/Cap;

  struct timeval tv1, tv2;
  gettimeofday( &tv1, NULL);

  for ( i = 0; i < num_iterations ; i++) {
    single_iteration(result, temp, power, row, col, Rx_1, Ry_1, Rz_1, step_div_Cap);
  }	

  gettimeofday( &tv2, NULL);
  runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);
}

int main(int argc, char **argv)
{
  int i, j, res;
  double *temp, *power, *result;
    
  num_omp_threads = NUM_THREAD; 

  /* allocate memory for the temperature and power arrays */
  temp = (double *) malloc (SIZE*SIZE*sizeof(double));
  power = (double *) malloc (SIZE*SIZE*sizeof(double));
  result = (double *) malloc (SIZE*SIZE*sizeof(double));

  srand( 2012);

  for( i = 0 ; i < SIZE; i++){
    for ( j = 0 ; j < SIZE; j++){
      temp[i*SIZE+j] = (double)(rand()%20983591+323980566)/1000000.0; 
    }
  }

  for( i = 0 ; i < SIZE; i++){
    for ( j = 0 ; j < SIZE; j++){
      power[i*SIZE+j] = (double)(rand()%702+4)/1000000.0; 
    }
  }  

  /* Main computation */
  compute_tran_temp( result, ITER, temp, power, SIZE, SIZE);

  res = (int)temp[0]; 

  /* cleanup	*/
  free(temp);
  free(power);
  free(result);

  return( res);
}

