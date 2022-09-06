#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "kmeans.h"

#define PY_SSIZE_T_CLEAN

double **kmeans(int k, int dim, int numOfRows, int maxIter, double epsilon, double **datapoints, double **kmeansList){
    double **clusterSumList;
    double *datapoint;
    int *clusterSizeList;
    int curIter, minClusterIndex, i, j;
    for (curIter = 0; curIter < maxIter; curIter++)
    {
        clusterSumList = calloc(k, sizeof(double*));
        for (i = 0; i < k; i++)
        {
            clusterSumList[i] = calloc(dim, sizeof(double));
        }
        clusterSizeList = calloc(k, sizeof(int));
        for (i = 0; i < numOfRows; i++)
        {
            datapoint = datapoints[i];
            minClusterIndex = getMinClusterIndex(k, dim, datapoint, kmeansList); /*find the closest kmean to the datapoint*/
            clusterSizeList[minClusterIndex]++;
            for (j = 0; j < dim; j++)
            {
                clusterSumList[minClusterIndex][j] += datapoint[j]; /*Adding the datapoint's values to kmeans*/
            }
        }
        if (!updateKmeans(k, dim, epsilon, kmeansList, clusterSumList, clusterSizeList)) /*updating the kmeans to their new values according to the algorithm*/
        {
            break; /* If all of the kmeans didn't chenge their values more than epsilon than the algorithm is finished*/
        }    
    }

    for (i = 0; i < numOfRows; i++)
    {
        free(datapoints[i]);
    }
    free(datapoints);
    
    if (curIter > 0)
    {
        for (i = 0; i < k; i++)
        {
            free(clusterSumList[i]);
        }
        free(clusterSumList);
        free(clusterSizeList);
    }
    return kmeansList;
}

int getMinClusterIndex(int k, int dim, double *datapoint, double **kmeansList){
    double minDist = INFINITY, curDistance;
    int minClusterIndex = -1, i;
    for (i = 0; i < k; i++)
    {
        curDistance = calcDist(dim, datapoint, kmeansList[i]);
        if (curDistance < minDist)
        {
            minDist = curDistance;
            minClusterIndex = i;
        }
    }
    return minClusterIndex;    
}

double calcDist(int dim, double *x, double *u){
    double sumSquare = 0;
    int i;
    for (i = 0; i < dim; i++)
    {
        sumSquare += pow(x[i] - u[i], 2);
    }
    return sqrt(sumSquare);
}

int updateKmeans(int k, int dim, double epsilon, double **kmeansList, double **clusterSumList, int *clusterSizeList){
    double* prevKmean;
    int isAnyMoreThanEpsilon = 0, i, j;
    prevKmean = calloc(dim, sizeof(double));
    for (i = 0; i < k; i++)
    {
        for (j = 0; j < dim; j++)
        {
            prevKmean[j] = kmeansList[i][j];
            clusterSumList[i][j] /= clusterSizeList[i];
        }
        for (j = 0; j < dim; j++)
        {
            kmeansList[i][j] = clusterSumList[i][j];
        }
        if (calcDist(dim, kmeansList[i], prevKmean) >= epsilon
    )
        {
            isAnyMoreThanEpsilon = 1;
        }
    }
    free(prevKmean);
    return isAnyMoreThanEpsilon; /* return false if all of the kean's differences in values are less than epsilon*/
}