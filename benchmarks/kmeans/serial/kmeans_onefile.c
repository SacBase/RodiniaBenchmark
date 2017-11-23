#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <omp.h>


#define RANDOM_MAX 2147483647


#define FLT_MAX ((3.40282347e+38))


#define NFEATURES 34
#define NCLUSTERS 5
#define THRESHOLD 0.001

#ifndef SIZE
#define SIZE 1048576
#endif

#ifndef ITER
#define ITER 500
#endif


int find_nearest_point( double  *feature,          
                        int     ith,
                        int     nfeatures,
                        double  *clusters,         
                        int     nclusters)
{
  int index, i, j;
  double max_dist=FLT_MAX;

  /* find the cluster center id with min distance to pt */
  for (i=0; i<nclusters; i++) {
    double dist = 0.0;
    for (j=0; j<nfeatures; j++) {
      dist += (feature[ith*nfeatures+j]-clusters[i*nfeatures+j]) * (feature[ith*nfeatures+j]-clusters[i*nfeatures+j]);
    }

    if (dist < max_dist) {
      max_dist = dist;
      index    = i;
    }
  }
  return(index);
}

int main()
{
  int i, j;
  double *feature, *new_centers, *clusters;
  int *membership, *new_centers_len;

  feature = (double*)malloc(sizeof(double)*SIZE*NFEATURES);

  srand(7);

  for( i = 0; i < SIZE; i++) {
    for( j = 0; j < NFEATURES; j++) {
      feature[i*NFEATURES+j] = (double)(rand()%100); 
    }
  }

  clusters = (double*) malloc(NCLUSTERS * NFEATURES * sizeof(double));

  srand(2012);
  /* randomly pick cluster centers */
  for (i=0; i<NCLUSTERS; i++) {
    int n = (int)rand() % SIZE;
    for (j=0; j<NFEATURES; j++) {
      clusters[i*NFEATURES+j] = feature[n*NFEATURES+j];
    }
  }

  /* need to initialize new_centers_len and new_centers[0] to all 0 */
  new_centers_len = (int*) calloc(NCLUSTERS, sizeof(int));
  new_centers = (double*) malloc(NCLUSTERS * NFEATURES * sizeof(double));

  membership = (int*)malloc(sizeof(int)*SIZE);
  for( i = 0; i < SIZE; i++) { 
    membership[i] = -1;
  }

  int cluster_id;
  int c = 0;
  int loop = -1;
  double delta;

  struct timeval tv1, tv2;
  gettimeofday( &tv1, NULL);

  do {
    delta = 0.0;

    for (i = 0; i < SIZE; i++) {

      /* find the index of nestest cluster center */
      int index = find_nearest_point(feature, i, NFEATURES, clusters, NCLUSTERS);

      /* if membership changes, increase delta by 1 */
      if (membership[i] != index) delta += 1.0;

      /* assign the membership to object i */
      membership[i] = index;

      /* update new cluster centers : sum of objects located within */
      new_centers_len[index]++;
      for (j = 0; j < NFEATURES; j++) {         
	new_centers[index*NFEATURES+j] += feature[i*NFEATURES+j];
      }
    }
/*
    for (i = 0; i < npoints; i++) {		
      cluster_id = membership_new[i];
      new_centers_len[cluster_id] = new_centers_len[cluster_id]+1;
      if( membership_new[i] != membership[i]) {
        delta += 1.0;
        membership[i] = membership_new[i];
      }

      for (j = 0; j < nfeatures; j++) {			
        new_centers[cluster_id][j] = new_centers[cluster_id][j] + feature[i][j];
      }
    } 
*/

    /* replace old cluster centers with new_centers */
    for (i = 0; i < NCLUSTERS; i++) {
      for (j = 0; j < NFEATURES; j++) {
	  if (new_centers_len[i] > 0) {
            clusters[i*NFEATURES+j] = new_centers[i*NFEATURES+j] / new_centers_len[i];
          }
          new_centers[i*NFEATURES+j] = 0.0;   /* set back to 0 */
      }
      new_centers_len[i] = 0;   /* set back to 0 */
    }

    c++;
    loop++;
  } while (delta > THRESHOLD && loop < ITER);
 
  //printf("iterated %d times\n", c);

  gettimeofday( &tv2, NULL);
  double runtime = ((tv2.tv_sec*1000.0+ tv2.tv_usec/1000.0)-(tv1.tv_sec*1000.0+ tv1.tv_usec/1000.0));
  printf("%f\n", runtime);

#ifdef OUTPUT
  for( i = 0; i < NCLUSTERS; i++) {
    for( j = 0; j < NFEATURES; j++) {
      printf("%f ", clusters[i*NFEATURES+j]);
    }
    printf("\n");
  }
#endif

  free(feature); 
  free(clusters);
  free(membership);
  free(new_centers);
  free(new_centers_len);

  return( 0);
}

