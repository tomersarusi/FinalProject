#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define EPSILON 0.00001
#define MAX_ITER 100


int determineK(double* eigenvaluesArr, int sizeOfArr){
    int i, delta, maxDelta = INT_MIN, maxI;
    qsort(eigenvaluesArr, sizeOfArr, sizeof(double), compEigenvalues);
    for (i = 0; i < floor(sizeOfArr/2); i++)
    {
        delta = eigenvaluesArr[i] - eigenvaluesArr[i+1];
        if (delta > maxDelta)
        {
            maxDelta = delta;
            maxI = i;
        }
    }
    return maxI;    
}

int compEigenvalues(const void* elem1, const void* elem2){
    double v1,v2;
    v1 = *((double*)elem1);
    v2 = *((double*)elem2);
    if (v1 > v2)
        return -1;
    if (v2 < v1)
        return 1;
    return 0;
}

double** computeJacobi(double** mtx, int sizeOfMtx){
    int i, isConverged = 0, iter = 0;
    double offA, offAtag, **output, **P, **Ptrans, **PA, **Atag, **prevOutput;
    output = callocMatrix(sizeOfMtx);
    for (i = 0; i < sizeOfMtx; i++)
        output[i][i] = 1;
    offA = calcOffSquared(mtx, sizeOfMtx);
    while (!isConverged && iter < MAX_ITER)
    {
        P = createRotationMatrix(mtx, sizeOfMtx);
        Ptrans = matrixTranspose(P, sizeOfMtx);
        PA = matrixMultiply(Ptrans, mtx, sizeOfMtx);
        Atag = matrixMultiply(PA, P, sizeOfMtx);
        prevOutput = output;
        output = matrixMultiply(prevOutput, P, sizeOfMtx);
        offAtag = calcOffSquared(mtx, sizeOfMtx);
        if ((offA - offAtag) < EPSILON)
            isConverged = 1;

        Neo(P, sizeOfMtx);
        Neo(Ptrans, sizeOfMtx);
        Neo(PA, sizeOfMtx);
        Neo(prevOutput, sizeOfMtx);
        Neo(mtx, sizeOfMtx);
        mtx = Atag;
        offA = offAtag;
        iter++;
    }
    return output;
}

double calcOffSquared(double** mtx, int sizeOfMtx){
    int i,j;
    double output = 0;
    for (i = 0; i < sizeOfMtx; i++)
    {
        for (j = 0; j < sizeOfMtx; j++)
        {
            if (i != j)
            {
                output += pow(mtx[i][j],2);
            }
        }
    }
    return output;
}

double** createRotationMatrix(double** mtx, int sizeOfMtx){
    int i,maxI,maxJ, *pvtPnt;
    double s,c, *sinCos, **output;
    output = callocMatrix(sizeOfMtx);
    pvtPnt = getPivotPoint(mtx, sizeOfMtx);
    maxI = pvtPnt[0];
    maxJ = pvtPnt[1];
    sinCos = getSinCos(mtx[maxI][maxI], mtx[maxJ][maxJ], mtx[maxI][maxJ]);
    s = sinCos[0];
    c = sinCos[1];
    for (int i = 0; i < sizeOfMtx; i++)
    {
        if (i == maxI || i == maxJ)
            output[i][i] = c;
        else
            output[i][i] = 1; 
    }
    output[maxI][maxJ] = s;
    output[maxJ][maxI] = -s;
    free(pvtPnt);
    free(sinCos);
    return output;
}

double* getSinCos(double Aii, double Ajj, double Aij){
    double theta, t, *output;
    theta = (Ajj-Aii)/(2*Aij);
    t = getSign(theta)/(abs(theta) + sqrt(pow(theta,2) + 1));
    output = calloc(2, sizeof(double));
    output[1] = 1/(sqrt(pow(t,2) + 1));
    output[0] = t*output[1];
    return output;
}

int getSign(double x){
    if (x >= 0)
    {
        return 1;
    }
    return 0;
}

int* getPivotPoint(double** mtx, int sizeOfMtx){
    int i,j, maxVal = INT_MIN, *output;
    output = calloc(2, sizeof(int));
    for (i = 0; i < sizeOfMtx; i++)
    {
        for (j = 0; j < sizeOfMtx; j++)
        {
            if (abs(mtx[i][j]) > maxVal)
            {
                maxVal = abs(mtx[i][j]);
                output[0] = i;
                output[1] = j;
            }
        }
    }
    return output;
}

double** computeNormalizedLaplacian(double** invSqrtDiagMtx, double** adjMatrix, int sizeOfMtx){
    int i,j;
    double **output, **DW, **DWD;
    output = callocMatrix(sizeOfMtx);
    DW = matrixMultiply(invSqrtDiagMtx, adjMatrix, sizeOfMtx);
    DWD = matrixMultiply(DW, invSqrtDiagMtx, sizeOfMtx);  
    for (i = 0; i < sizeOfMtx; i++)
    {
        output[i][i] = 1;
        for (j = 0; j < sizeOfMtx; j++)
        {
            output[i][j] -= DWD[i][j];
        }
    }
    Neo(DW, sizeOfMtx);
    Neo(DWD, sizeOfMtx);
    return output;
}

double** createInvSqrtDiagDegMatrix(double** adjMatrix, int sizeOfMtx){
    int i,z;
    double** output;
    output = callocMatrix(sizeOfMtx);
    for (i = 0; i < sizeOfMtx; i++)
    {
        for (z = 0; z < sizeOfMtx; z++)
        {
            output[i][i] += adjMatrix[i][z];
        }
        output[i][i] = pow(output[i][i],-0.5);
    }
    return output;
}

double** createAdjMatrix(double** arr, int sizeOfArr, int dim){
    int i,j;
    double** output;
    output = callocMatrix(sizeOfArr);
    for (i = 0; i < sizeOfArr; i++)
    {
        for (j = 0; j < sizeOfArr; j++)
        {
            output[i][j] = exp(-1*calcDistance(arr[i],arr[j], dim)/2);
        }
    }
    return output;
}

double calcDistance(double* vec1, double* vec2, int dim){
    int i;
    double output = 0;
    for (i = 0; i < dim; i++)
    {
        output += pow(vec1[i] - vec2[i], 2);
    }
    return sqrt(output);
}

double** matrixMultiply(double** mtx1, double** mtx2, int sizeOfMtx){
    int i,j,k;
    double** output;
    output = CallocMatrix(sizeOfMtx);
    for (i = 0; i < sizeOfMtx; i++)
    {
        for (j = 0; j < sizeOfMtx; j++)
        {
            for (k = 0; k < sizeOfMtx; k++)
            {
                output[i][j] += mtx1[i][k] * mtx2[k][j];
            }
        }
    }
    return output;
}

double** matrixTranspose(double** mtx, int sizeOfMtx){
    int i,j;
    double** output;
    output = callocMatrix(sizeOfMtx);
    for (i = 0; i < sizeOfMtx; i++)
    {
        for (j = 0; j < sizeOfMtx; j++)
        {
            output[i][j] = mtx[j][i];
        }
    }
    return output;    
}

double** callocMatrix(int sizeOfMtx){
    int i;
    double** output;
    output = calloc(sizeOfMtx, sizeof(double*));
    for (i = 0; i < sizeOfMtx; i++)
    {
        output[i] = calloc(sizeOfMtx, sizeof(double));
    }
    return output;
}

void Neo(double** mtx, int sizeOfMtx){ // Free The Matrix
    int i;
    for (i = 0; i < sizeOfMtx; i++)
    {
        free(mtx[i]);
    }
    free(mtx);
}