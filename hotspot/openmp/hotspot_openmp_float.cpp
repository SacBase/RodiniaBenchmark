#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

using namespace std;

#define STR_SIZE 256
#define ITER 5000

#ifndef SIZE
#define SIZE 1024
#endif

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
//#define MAX_PD	      (3.0e6)
#define MAX_PD	      3000000.0f
/* required precision in degrees	*/
#define PRECISION     0.001f
//#define SPEC_HEAT_SI  1.75e6
#define SPEC_HEAT_SI  1750000.0f
#define K_SI          100
/* capacitance fitting factor	*/
#define FACTOR_CHIP   0.5f

//#define OPEN
#define NUM_THREAD 4

/* chip parameters	*/
float t_chip = 0.0005f;
float chip_height = 0.016f;
float chip_width = 0.016f;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0f;

int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
void single_iteration( float *result, float *temp, float *power, int row, int col,
                       float Rx_1, float Ry_1, float Rz_1,  float step_div_Cap)
{
  float delta;
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
		(temp[c+1] + temp[c-1] - 2.0f*temp[c]) * Rx_1 + 
		(temp[col+c] - temp[c]) * Ry_1 + 
		(amb_temp - temp[c]) * Rz_1);
      }	/*	Edge 2	*/
      else if (c == col-1) {
	delta = (step_div_Cap) * (power[r*col+c] + 
		(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[r*col+c]) * Ry_1 + 
		(temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
		(amb_temp - temp[r*col+c]) * Rz_1);
      }	/*	Edge 3	*/
      else if (r == row-1) {
	delta = (step_div_Cap) * (power[r*col+c] + 
		(temp[r*col+c+1] + temp[r*col+c-1] - 2.0f*temp[r*col+c]) * Rx_1 + 
		(temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
		(amb_temp - temp[r*col+c]) * Rz_1);
      }	/*	Edge 4	*/
      else if (c == 0) {
	delta = (step_div_Cap) * (power[r*col] + 
		(temp[(r+1)*col] + temp[(r-1)*col] - 2.0f*temp[r*col]) * Ry_1 + 
		(temp[r*col+1] - temp[r*col]) * Rx_1 + 
		(amb_temp - temp[r*col]) * Rz_1);
      }	/*	Inside the chip	*/
      else {
	delta = (step_div_Cap) * (power[r*col+c] + 
		(temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0f*temp[r*col+c]) * Ry_1 + 
		(temp[r*col+c+1] + temp[r*col+c-1] - 2.0f*temp[r*col+c]) * Rx_1 + 
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

/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(float *result, int num_iterations, float *temp, float *power, int row, int col) 
{
  #ifdef VERBOSE
  int i = 0;
  #endif

  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0f * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0f * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);
  float Rx_1 = 1.0f/Rx; 
  float Ry_1 = 1.0f/Ry;
  float Rz_1 = 1.0f/Rz;

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  float t;

  float step_div_Cap = step/Cap;

  #ifdef VERBOSE
  fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
  fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
  #endif

  struct timeval tv1, tv2;
  gettimeofday( &tv1, NULL);

  for (int i = 0; i < num_iterations ; i++) {
    #ifdef VERBOSE
    fprintf(stdout, "iteration %d\n", i++);
    #endif
    single_iteration(result, temp, power, row, col, Rx_1, Ry_1, Rz_1, step_div_Cap);
  }	

  gettimeofday( &tv2, NULL);
  double runtime = ((tv2.tv_sec+ tv2.tv_usec/1000000.0)-(tv1.tv_sec+ tv1.tv_usec/1000000.0));
  printf("Runtime(seconds): %f\n", runtime);

  #ifdef VERBOSE
  fprintf(stdout, "iteration %d\n", i++);
  #endif
}

void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
  exit(1);
}

void readinput(float *vect, int grid_rows, int grid_cols, char *file)
{
  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 ) {
    printf( "The file was not opened\n" );
  }

  for (i=0; i <= grid_rows-1; i++) {
    for (j=0; j <= grid_cols-1; j++) {
      fgets(str, STR_SIZE, fp);
      if (feof(fp)) {
        fatal("not enough lines in file");
      }
      if ((sscanf(str, "%f", &val) != 1)) {
        fatal("invalid file format");
      }
      vect[i*grid_cols+j] = val;
    }
  }
  fclose(fp);	
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file)
{
/*
  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 )
     printf( "The file was not opened\n" );

  for (i=0; i < grid_rows; i++) { 
    for (j=0; j < grid_cols; j++)  {
       sprintf(str, "%g\n", vect[i*grid_cols+j]);
       fputs(str,fp);
       index++;
    }
  }		
  fclose(fp);	
*/

#ifdef OUTPUT
  int i,j;
  for (i=0; i < grid_rows; i++) { 
    for (j=0; j < grid_cols; j++) {
      printf("%f\n", vect[i*grid_cols+j]);
    }
  }
#else
  printf("%f\n", vect[0]);
#endif
}

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
  fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<no. of threads>   - number of threads\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  exit(1);
}

int main(int argc, char **argv)
{
  int grid_rows, grid_cols, sim_time, i;
  float *temp, *power, *result;
  char *tfile, *pfile, *ofile;
    
/*  
  if (argc != 7) {
    usage(argc, argv);
  }

  if ((grid_rows = atoi(argv[1])) <= 0 ||
      (grid_cols = atoi(argv[1])) <= 0 ||
      (sim_time = atoi(argv[2])) <= 0 || 
      (num_omp_threads = atoi(argv[3])) <= 0) {
    usage(argc, argv);
  }
*/

  grid_rows = SIZE;
  grid_cols = SIZE; 

  sim_time = ITER;
  num_omp_threads = NUM_THREAD; 

  /* allocate memory for the temperature and power arrays	*/
  temp = (float *) calloc (grid_rows * grid_cols, sizeof(float));
  power = (float *) calloc (grid_rows * grid_cols, sizeof(float));
  result = (float *) calloc (grid_rows * grid_cols, sizeof(float));

  if(!temp || !power) {
    fatal("unable to allocate memory");
  }

  /* read initial temperatures and input power	*/
  tfile=argv[1];
  pfile=argv[2];
  ofile=argv[3];
  readinput(temp, grid_rows, grid_cols, tfile);
  readinput(power, grid_rows, grid_cols, pfile);

  /* Main computation */
#ifdef VERBOSE
  printf("Start computing the transient temperature\n");
#endif
  compute_tran_temp( result, sim_time, temp, power, grid_rows, grid_cols);
#ifdef VERBOSE
  printf("Ending simulation\n");
#endif

  /* output results	*/
#ifdef VERBOSE
  fprintf(stdout, "Final Temperatures:\n");
#endif

  writeoutput(temp,grid_rows, grid_cols, ofile);

  /* cleanup	*/
  free(temp);
  free(power);

  return 0;
}

