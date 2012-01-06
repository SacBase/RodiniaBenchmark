#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "papi.h"

#ifndef N
#define N 1024
#endif
#define MAXLINE 128

int create_matrix_from_file(float **mp)
{
  int i, j, size;
  float *m;
  FILE *fp = NULL;
  char filename[MAXLINE];

  sprintf(filename, "../input/%d.dat", N); 

  fp = fopen(filename, "rb");
  if ( fp == NULL) {
    return( 0);
  }

  fscanf(fp, "%d\n", &size);

  if( size != N) {
    printf("Wrong file read. Expecting %d.dat but read %d.dat\n", N, size);
    return( 0);
  }

  m = (float*) malloc(sizeof(float)*N*N);
  if ( m == NULL) {
      fclose(fp);
      return( 0);
  }

  for (i=0; i < size; i++) {
    for (j=0; j < size; j++) {
      fscanf(fp, "%f ", m+i*size+j);
    }
  }

  fclose(fp);

  *mp = m;

  return( 1);
}

int main()
{
  int k, i, j, ret;
  float *mat, *mat_tmp1, *mat_tmp2;
  double runtime;
  struct timeval tv1, tv2;
  float kk;


  ret = create_matrix_from_file( &mat);
 
  if( !ret) {
    printf("Error has occured during reading file. Abort.\n");
    return( 1);
  } 

//  gettimeofday( &tv1, NULL);

#ifdef PAPI
  int retval, event_set, event_code;
  long long values[4];
  long long L1_misses = 0, L1_accesses = 0;
  long long L2_misses = 0, L2_accesses = 0;

  retval = PAPI_library_init( PAPI_VER_CURRENT);

  event_set = PAPI_NULL;
  retval = PAPI_create_eventset( &event_set);    
  if( retval != PAPI_OK) printf("error!\n");
 
  event_code = PAPI_L1_DCM;
  retval = PAPI_add_event( event_set, event_code);    
  if( retval != PAPI_OK) printf("L1_DCM error - code %d\n", retval);

  event_code = PAPI_L1_DCA;
  retval = PAPI_add_event( event_set, event_code);    
  if( retval != PAPI_OK) printf("L1_DCA error - code %d\n", retval);

  event_code = PAPI_L2_DCM;
  retval = PAPI_add_event( event_set, event_code);  
  if( retval != PAPI_OK) printf("L2_DCM error - code %d\n", retval);

  event_code = PAPI_L2_DCA;
  retval = PAPI_add_event( event_set, event_code);  
  if( retval != PAPI_OK) printf("L2_DCA error - code %d\n", retval);
#endif

  for( k = 0; k < N-1; k++) {

    mat_tmp1 = (float*)malloc(sizeof(float)*N*N);  

#ifdef PAPI
    retval = PAPI_start( event_set);  
    if( retval != PAPI_OK) printf("PAPI_start - code %d\n", retval);
#endif

    for( i = k+1; i < N; i++) {
      mat_tmp1[i*N+k] = mat[i*N+k]/mat[k*N+k];
    }

#ifdef PAPI
    retval = PAPI_stop( event_set, values);  
    if( retval != PAPI_OK) printf("PAPI_stop - code %d\n", retval);
    L1_misses += values[0];
    L1_accesses += values[1];
    L2_misses += values[2];
    L2_accesses += values[3];
    retval = PAPI_reset( event_set);  
    if( retval != PAPI_OK) printf("PAPI_reset - code %d\n", retval);
#endif

    free(mat);
    mat_tmp2 = (float*)malloc(sizeof(float)*N*N);  

#ifdef PAPI
    retval = PAPI_start( event_set);  
    if( retval != PAPI_OK) printf("PAPI_start - code %d\n", retval);
#endif

    for( i = k+1; i < N; i++) {
      for( j = k+1; j < N; j++) {
        mat_tmp2[i*N+j] = mat_tmp1[i*N+j]-mat_tmp1[i*N+k]*mat_tmp1[k*N+j];
      }
    } 

#ifdef PAPI
    retval = PAPI_stop( event_set, values);  
    if( retval != PAPI_OK) printf("PAPI_stop - code %d\n", retval);
    L1_misses += values[0];
    L1_accesses += values[1];
    L2_misses += values[2];
    L2_accesses += values[3];
    retval = PAPI_reset( event_set);  
    if( retval != PAPI_OK) printf("PAPI_reset - code %d\n", retval);
#endif

    free(mat_tmp1);
    mat = mat_tmp2;
  }

#ifdef PAPI
  printf( "L1_misses = %lld\n", L1_misses);
  printf( "L1_accesses = %lld\n",L1_accesses);
  printf( "L1 hit rate = %f\n", (float)(L1_accesses-L1_misses)/(float)L1_accesses);

  printf( "L2_misses = %lld\n", L2_misses);
  printf( "L2_accesses = %lld\n", L2_accesses);
  printf( "L2 hit rate = %f\n", (float)(L2_accesses-L2_misses)/(float)L2_accesses);
#endif

//  gettimeofday( &tv2, NULL);
//  runtime = ((tv2.tv_sec*1000.0 + tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0 + tv1.tv_usec/1000.0));
//  printf("Runtime(milli-seconds): %f\n", runtime);

#ifdef OUTPUT
  for( i = 0; i < N; i++) {
    for( j = 0; j < N; j++) {
      printf("%f ", mat[i*N+j]);
    }
    printf("\n");
  }
#else
  printf("%f\n", mat[0]);
#endif

  return( 0);
}



