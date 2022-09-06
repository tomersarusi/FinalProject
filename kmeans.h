#ifndef KMEANS_H_
#define KMEANS_H_

double **kmeans(int _K, int _dim, int _numOfRows, int maxIter, double _eps, double **datapoints, double **kmeansList);
int getMinClusterIndex(int k, int dim, double *datapoint, double **kmeansList);
double calcDist(int dim, double *x, double *u);
int updateKmeans(int k, int dim, double epsilon, double **kmeansList, double **clusterSumList, int *clusterSizeList);

#endif