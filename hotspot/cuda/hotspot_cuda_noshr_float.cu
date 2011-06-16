#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define BLOCK_SIZE 16
#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001f
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5f

/* chip parameters	*/
float t_chip = 0.0005f;
float chip_height = 0.016f;
float chip_width = 0.016f;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0f;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

void fatal(char *s)
{
  fprintf(stderr, "error: %s\n", s);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file)
{
  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 )
    printf( "The file was not opened\n" );

  for (i=0; i < grid_rows; i++) 
    for (j=0; j < grid_cols; j++) {
      sprintf(str, "%g\n", vect[i*grid_cols+j]);
      fputs(str,fp);
      index++;
  }
  fclose(fp);	
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file)
{
  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  if( (fp  = fopen(file, "r" )) ==0 )
    printf( "The file was not opened\n" );

  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++) {
      fgets(str, STR_SIZE, fp);
      if (feof(fp))
        fatal("not enough lines in file");
      //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
      if ((sscanf(str, "%f", &val) != 1))
        fatal("invalid file format");
      vect[i*grid_cols+j] = val;
  }
  fclose(fp);	
}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void calculate_temp(float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
                               float Cap,      //Capacitance
                               float Rx, 
                               float Ry, 
                               float Rz, 
                               float step){

  float amb_temp = 80.0f;
  float Rx_1,Ry_1,Rz_1;
  float delta;
        
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx=threadIdx.x;
  int ty=threadIdx.y;
	
  Rx_1=1.0f/Rx;
  Ry_1=1.0f/Ry;
  Rz_1=1.0f/Rz;

  int idxX = bx*BLOCK_SIZE+tx;
  int idxY = by*BLOCK_SIZE+ty;

  int index = idxY*grid_cols+idxX;

  int N, S, W, E;

  N = index - grid_cols;
  S = index + grid_cols;
  W = index - 1;
  E = index + 1;

  /*	Corner 1	*/
  if ( (idxY == 0) && (idxX == 0) ) {
    delta = (step / Cap) * (power[index] +
            (temp_src[E] - temp_src[index]) * Rx_1 +
	    (temp_src[S] - temp_src[index]) * Ry_1 +
	    (amb_temp - temp_src[index]) * Rz_1);
  }	/*	Corner 2	*/
  else if ((idxY == 0) && (idxX == grid_cols-1)) {
    delta = (step / Cap) * (power[index] +
            (temp_src[W] - temp_src[index]) * Rx_1 +
				(temp_src[S] - temp_src[index]) * Ry_1 +
				(amb_temp - temp_src[index]) * Rz_1);
	}	/*  Corner 3	*/
	else if ((idxY == grid_rows-1) && (idxX == grid_cols-1)) {
		delta = (step / Cap) * (power[index] + 
				(temp_src[W] - temp_src[index]) * Rx_1 + 
				(temp_src[N] - temp_src[index]) * Ry_1 + 
				(amb_temp - temp_src[index]) * Rz_1);					
	}	/*	Corner 4	*/
	else if ((idxY == grid_rows-1) && (idxX == 0)) {
		delta = (step / Cap) * (power[index] + 
				(temp_src[E] - temp_src[index]) * Rx_1 + 
				(temp_src[N] - temp_src[index]) * Ry_1 + 
				(amb_temp - temp_src[index]) * Rz_1);
	}	/*	Edge 1	*/
	else if (idxY == 0) {
		delta = (step / Cap) * (power[index] + 
				(temp_src[E] + temp_src[W] - 2.0f*temp_src[index]) * Rx_1 + 
				(temp_src[S] - temp_src[index]) * Ry_1 + 
				(amb_temp - temp_src[index]) * Rz_1);
	}	/*	Edge 2	*/
	else if (idxX == grid_cols-1) {
		delta = (step / Cap) * (power[index] + 
				(temp_src[S] + temp_src[N] - 2.0f*temp_src[index]) * Ry_1 + 
				(temp_src[W] - temp_src[index]) * Rx_1 + 
				(amb_temp - temp_src[index]) * Rz_1);
	}	/*	Edge 3	*/
	else if (idxY == grid_rows-1) {
		delta = (step / Cap) * (power[index] + 
				(temp_src[E] + temp_src[W] - 2.0f*temp_src[index]) * Rx_1 + 
				(temp_src[N] - temp_src[index]) * Ry_1 + 
				(amb_temp - temp_src[index]) * Rz_1);
	}	/*	Edge 4	*/
	else if (idxX == 0) {
		delta = (step / Cap) * (power[index] + 
				(temp_src[S] + temp_src[N] - 2.0f*temp_src[index]) * Ry_1 + 
				(temp_src[E] - temp_src[index]) * Rx_1 + 
				(amb_temp - temp_src[index]) * Rz_1);
	}	/*	Inside the chip	*/
	else {
		delta = (step / Cap) * (power[index] + 
				(temp_src[S] + temp_src[N] - 2.0f*temp_src[index]) * Ry_1 + 
				(temp_src[E] + temp_src[W] - 2.0f*temp_src[index]) * Rx_1 + 
				(amb_temp - temp_src[index]) * Rz_1);
	}
  			
	/*	Update Temperatures	*/
	temp_dst[index] = temp_src[index]+ delta;
}


/*
   compute N time steps
*/

int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
		int total_iterations, int blockCols, int blockRows) 
{
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // [16, 16]
  dim3 dimGrid(blockCols, blockRows);  
	
  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0f * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0f * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;
  float t;

  int src = 1, dst = 0;
	
  for (t = 0; t < total_iterations; t++) {
    int temp = src;
    src = dst;
    dst = temp;
    calculate_temp<<<dimGrid, dimBlock>>>( MatrixPower,MatrixTemp[src],MatrixTemp[dst],\
                                           col,row, Cap,Rx,Ry,Rz,step);
  }
  return dst;
}

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file\n");
  exit(1);
}

int main(int argc, char** argv)
{
    run(argc,argv);

    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    int size;
    int grid_rows,grid_cols;
    float *FilesavingTemp,*FilesavingPower,*MatrixOut; 
    char *tfile, *pfile, *ofile;
    
    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations

    if (argc != 7) {
      usage(argc, argv);
    }

    if((grid_rows = atoi(argv[1]))<=0||
       (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0) {
 	usage(argc, argv);
    }
		
    tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];
	
    size=grid_rows*grid_cols;

    int blockCols = grid_cols/BLOCK_SIZE;
    int blockRows = grid_rows/BLOCK_SIZE;

    FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingPower = (float *) malloc(size*sizeof(float));
    MatrixOut = (float *) calloc (size, sizeof(float));

    if( !FilesavingPower || !FilesavingTemp || !MatrixOut) {
        fatal("unable to allocate memory");
    }

    printf( "gridSize: [%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
	    grid_cols, grid_rows,  blockCols, blockRows, BLOCK_SIZE, BLOCK_SIZE);
	
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);

    float *MatrixTemp[2], *MatrixPower;
    cudaMalloc((void**)&MatrixTemp[0], sizeof(float)*size);
    cudaMalloc((void**)&MatrixTemp[1], sizeof(float)*size);
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&MatrixPower, sizeof(float)*size);
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float)*size, cudaMemcpyHostToDevice);
    printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp( MatrixPower,MatrixTemp,grid_cols,grid_rows, \
	                         total_iterations, blockCols, blockRows);
    printf("Ending simulation\n");
    cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);

    writeoutput(MatrixOut,grid_rows, grid_cols, ofile);

    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);
    free(MatrixOut);
}
