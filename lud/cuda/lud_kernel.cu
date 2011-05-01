#include <cuda.h>
#include <stdio.h>

#define BLOCK_SIZE 16


__global__ void lud_diagonal(float *m, int matrix_dim, int offset)
{
  int i,j;
  __shared__ float shadow[BLOCK_SIZE][BLOCK_SIZE];

  /* Each thread block, i.e. 1D 16 threads, loads a 
   * 2D block, i.e. 16x16, of data from the diagonal
   * of the matrix into shared memory 'shadow' */
  int array_offset = offset*matrix_dim+offset;
  for(i = 0; i < BLOCK_SIZE; i++){
    shadow[i][threadIdx.x] = m[array_offset+threadIdx.x];
    array_offset += matrix_dim;
  }
  __syncthreads();

  for(i = 0; i < BLOCK_SIZE-1; i++) {
    if ( threadIdx.x > i) { /* starts at 15 threads and then decrease one thread each time */
      for(j = 0; j < i; j++) { /* This for loop computes cols */
        shadow[threadIdx.x][i] -= shadow[threadIdx.x][j]*shadow[j][i];
      }
      shadow[threadIdx.x][i] /= shadow[i][i];
    }
    __syncthreads();

    if ( threadIdx.x > i) {
      for( j = 0; j < i+1; j++) { /* This for loop computes rows */
        shadow[i+1][threadIdx.x] -= shadow[i+1][j]*shadow[j][threadIdx.x];
      }
    }
    __syncthreads();
  }

  /* The first row is not modified, it
   * is no need to write it back to the
   * global memory */
  array_offset = (offset+1)*matrix_dim+offset;
  for(i = 1; i < BLOCK_SIZE; i++) {
    m[array_offset+threadIdx.x] = shadow[i][threadIdx.x];
    array_offset += matrix_dim;
  }
}

__global__ void lud_diagonal_noshr(float *m, int matrix_dim, int offset)
{
  int i,j;

  int array_offset = offset*matrix_dim+offset;

  for(i = 0; i < BLOCK_SIZE-1; i++) {
    if ( threadIdx.x > i) { /* starts at 15 threads and then decrease one thread each time */
      for(j = 0; j < i; j++) { /* This for loop computes cols */
        m[(array_offset+threadIdx.x*matrix_dim) + i] -= 
          m[(array_offset+threadIdx.x*matrix_dim) + j]*
          m[(array_offset+j*matrix_dim) + i];
      }
      m[(array_offset+threadIdx.x*matrix_dim) + i] /= m[(array_offset+i*matrix_dim) + i];
    }
    __syncthreads();

    if ( threadIdx.x > i) {
      for( j = 0; j < i+1; j++) { /* This for loop computes rows */
        m[(array_offset+(i+1)*matrix_dim) + threadIdx.x] -=
          m[(array_offset+(i+1)*matrix_dim) + j]*
          m[(array_offset+j*matrix_dim) + threadIdx.x];
      }
    }
    __syncthreads();
  }
}

__global__ void lud_perimeter(float *m, int matrix_dim, int offset)
{
  __shared__ float dia[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i,j, array_offset;
  int idx;

  /* For this kernel, each block contains 32 threads */
  if ( threadIdx.x < BLOCK_SIZE) { /* threads 0 ... 15 */
    idx = threadIdx.x;
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE/2; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = offset*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_row[i][idx]=m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }

  } else { /* threads 16 ... 31 */
    idx = threadIdx.x-BLOCK_SIZE;
    
    array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
    for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++){
      dia[i][idx]=m[array_offset+idx];
      array_offset += matrix_dim;
    }
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i][idx] = m[array_offset+idx];
      array_offset += matrix_dim;
    }
  }
  __syncthreads();

/* this version works ok on hardware, but not gpgpusim
 **************************************************************
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++)
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
    }

    
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++)
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      peri_col[idx][i] /= dia[i][i];
    }

    __syncthreads();
    
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }
***************************************************************/
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++) {
        peri_row[i][idx]-=dia[i][j]*peri_row[j][idx];
      }
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++) {
        peri_col[idx][i]-=peri_col[idx][j]*dia[j][i];
      }
      peri_col[idx][i] /= dia[i][i];
    }
  }

  __syncthreads();
   
  /* write data back to global memory */ 
  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    idx=threadIdx.x;
    array_offset = (offset+1)*matrix_dim+offset;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+(blockIdx.x+1)*BLOCK_SIZE+idx] = peri_row[i][idx];
      array_offset += matrix_dim;
    }
  } else { //peri-col
    idx=threadIdx.x - BLOCK_SIZE;
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    for(i=0; i < BLOCK_SIZE; i++){
      m[array_offset+idx] =  peri_col[i][idx];
      array_offset += matrix_dim;
    }
  }
}

__global__ void lud_perimeter_noshr(float *m, int matrix_dim, int offset)
{
  int i,j, array_offset;
  int idx;

  if (threadIdx.x < BLOCK_SIZE) { //peri-row
    array_offset = offset*matrix_dim+offset;
    idx=threadIdx.x;
    for(i=1; i < BLOCK_SIZE; i++){
      for (j=0; j < i; j++) {
        m[array_offset+i*matrix_dim+(blockIdx.x+1)*BLOCK_SIZE+idx] -= 
          m[array_offset+i*matrix_dim+j] * m[array_offset+j*matrix_dim+(blockIdx.x+1)*BLOCK_SIZE+idx];
      }
    }
  } else { //peri-col
    array_offset = (offset+(blockIdx.x+1)*BLOCK_SIZE)*matrix_dim+offset;
    idx=threadIdx.x - BLOCK_SIZE;
    for(i=0; i < BLOCK_SIZE; i++){
      for(j=0; j < i; j++) {
        m[array_offset+idx*matrix_dim+i] -= m[array_offset+idx*matrix_dim+j] * m[offset*matrix_dim+offset+j*matrix_dim+i];
      }
      m[array_offset+idx*matrix_dim+i] /= m[offset*matrix_dim+offset+i*matrix_dim+i];
    }
  }
}

__global__ void lud_internal(float *m, int matrix_dim, int offset)
{
  __shared__ float peri_row[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float peri_col[BLOCK_SIZE][BLOCK_SIZE];

  int i;
  float sum;

  int global_row_id = offset + (blockIdx.y+1)*BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x+1)*BLOCK_SIZE;

  peri_row[threadIdx.y][threadIdx.x] = m[(offset+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x];
  peri_col[threadIdx.y][threadIdx.x] = m[(global_row_id+threadIdx.y)*matrix_dim+offset+threadIdx.x];

  __syncthreads();

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++) {
    sum += peri_col[threadIdx.y][i] * peri_row[i][threadIdx.x];
  }
  m[(global_row_id+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x] -= sum;
}

__global__ void lud_internal_noshr(float *m, int matrix_dim, int offset)
{
  int i;
  float sum;

  int global_row_id = offset + (blockIdx.y+1)*BLOCK_SIZE;
  int global_col_id = offset + (blockIdx.x+1)*BLOCK_SIZE;

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++) {
    sum += m[(global_row_id+threadIdx.y)*matrix_dim+offset+i] * m[(offset+i)*matrix_dim+global_col_id+threadIdx.x];
  }
  m[(global_row_id+threadIdx.y)*matrix_dim+global_col_id+threadIdx.x] -= sum;
}

void lud_cuda(float *m, int matrix_dim, int do_shared)
{
  int i=0;
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  float *m_debug = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));

  if( do_shared) {
    printf("Executing kernels with shared memory!\n");
    for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
      lud_diagonal<<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
      lud_perimeter<<<(matrix_dim-i)/BLOCK_SIZE-1, BLOCK_SIZE*2>>>(m, matrix_dim, i);
      dim3 dimGrid((matrix_dim-i)/BLOCK_SIZE-1, (matrix_dim-i)/BLOCK_SIZE-1);
      lud_internal<<<dimGrid, dimBlock>>>(m, matrix_dim, i); 
    }
    lud_diagonal<<<1,BLOCK_SIZE>>>(m, matrix_dim, i);
  }
  else {
    printf("Executing kernels without shared memory!\n");
    
    cudaFuncSetCacheConfig("lud_diagonal_noshr", cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig("lud_perimeter_noshr", cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig("lud_internal_noshr", cudaFuncCachePreferL1);

    for (i=0; i < matrix_dim-BLOCK_SIZE; i += BLOCK_SIZE) {
      lud_diagonal_noshr<<<1, BLOCK_SIZE>>>(m, matrix_dim, i);
      lud_perimeter_noshr<<<(matrix_dim-i)/BLOCK_SIZE-1, BLOCK_SIZE*2>>>(m, matrix_dim, i);
      dim3 dimGrid((matrix_dim-i)/BLOCK_SIZE-1, (matrix_dim-i)/BLOCK_SIZE-1);
      lud_internal_noshr<<<dimGrid, dimBlock>>>(m, matrix_dim, i); 
    }
    lud_diagonal<<<1,BLOCK_SIZE>>>(m, matrix_dim, i);
  }
}

