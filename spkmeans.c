#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "spkmeans.h"

#define EPSILON 0.00001
#define MAX_ITER 100
#ifndef INFINITY
#define INFINITY 1.0/0.0
#endif


int main(int argc, char const *argv[]){
    int n, dim, i;
    double **mtx, **wamMtx, **ddgMatrix, **lnormMatrix, *eigenvalues, **eigenvectors;
    if (argc != 3)
        throwError(0); 
    mtx = getMatrixFromFile(argv[2], &n, &dim);
    if (strcmp(argv[1], "wam") == 0){
        wamMtx = createWeightAdjMatrix(mtx, n, dim);
        printMatrix(wamMtx, n, n);
        Neo(wamMtx, n);
        Neo(mtx, n);
    }
    else if (strcmp(argv[1], "ddg") == 0){
        wamMtx = createWeightAdjMatrix(mtx, n, dim);
        ddgMatrix = diagDegMatrix(wamMtx, n);
        printMatrix(ddgMatrix, n, n);
        Neo(wamMtx, n);
        Neo(ddgMatrix, n);
        Neo(mtx, n);
    }
    else if (strcmp(argv[1], "lnorm") == 0){
        wamMtx = createWeightAdjMatrix(mtx, n, dim);
        ddgMatrix = diagDegMatrix(wamMtx, n);
        lnormMatrix = computeNormalizedLaplacian(ddgMatrix, wamMtx, n);
        printMatrix(lnormMatrix, n, n);
        Neo(wamMtx, n);
        Neo(ddgMatrix, n);
        Neo(lnormMatrix, n);
        Neo(mtx, n);
    }
    else if (strcmp(argv[1], "jacobi") == 0){
        eigenvalues = calloc(n, sizeof(double));
        eigenvectors = computeJacobi(mtx, n, eigenvalues);
        for (i = 0; i < n; i++)
        {
            printf("%.4f", eigenvalues[i]);
            if (i != n -1)
            {
                printf(",");
            }
        }
        printf("\n");
        printMatrix(eigenvectors, n, n);
        free(eigenvalues);
        Neo(eigenvectors, n);
        Neo(mtx, n);
    }
    else /*Error*/
    {
        Neo(mtx, n);
        throwError(0);
    }
    return 0;
}

double** getMatrixFromFile(const char* fileName, int* oN, int* oDim){
    int i, j = 0, n = 0, dim = 0;
    double tmp[1000][15], **output;
    char c;
    FILE* input = fopen(fileName, "r");
    if (!input)
        throwError(0);

    while (!feof(input)){
        if (fscanf(input, "%lf", &tmp[n][j]) != 1)
        {
            if (!feof(input))
            {
                throwError(0);
            }
        }
        j++;
        c = fgetc(input);
        if (c == '\n' || c == '\r')
        {
            dim = j;
            j = 0;
            n++;
        }
    }
    fclose(input);
    output = callocMatrix(n, dim);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < dim; j++)
        {
            output[i][j] = tmp[i][j];
        }
    }
    *oN = n;
    *oDim = dim;
    return output;
}

double** partialSpk(int *k, double** mtx, int n, int dim){
    int i, j, maxIndex;
    double maxVal, vecSum, **wamMtx, **ddgMatrix, **lnormMatrix, *eigenvalues, **eigenvectors, **output;
    wamMtx = createWeightAdjMatrix(mtx, n, dim);
    ddgMatrix = diagDegMatrix(wamMtx, n);
    lnormMatrix = computeNormalizedLaplacian(ddgMatrix, wamMtx, n);
    eigenvalues = calloc(n, sizeof(double));
    eigenvectors = computeJacobi(lnormMatrix, n, eigenvalues);
    if (*k == 0)
        *k = determineK(eigenvalues, n);
    output = callocMatrix(n, *k);
    for (i = 0; i < *k; i++) /* Calculate U */
    {
        maxVal = -INFINITY;
        maxIndex = 0;
        for (j = 0; j < n; j++)
        {
            if (eigenvalues[j] > maxVal)
            {
                maxVal = eigenvalues[j];
                maxIndex = j;
            }
        }
        eigenvalues[maxIndex] = -INFINITY;
        for (j = 0; j < n; j++)
            output[j][i] = eigenvectors[j][maxIndex];
    } /* End of calculate U */
    for (i = 0; i < n; i++)
    {
        vecSum = 0;
        for (j = 0; j < *k; j++)
        {
            vecSum += pow(output[i][j],2);
        }
        vecSum = sqrt(vecSum);
        for (j = 0; j < *k; j++)
        {
            if (vecSum != 0)
            {
                output[i][j] /= vecSum;
            }
        }
    }

    Neo(wamMtx, n);
    Neo(ddgMatrix, n);
    Neo(eigenvectors, n);
    free(eigenvalues);
    return output;
}

int determineK(double* eigenvaluesArr, int sizeOfArr){
    int i, maxI = -1;
    double delta, maxDelta = -INFINITY, *cpyOfArr;
    cpyOfArr = calloc(sizeOfArr, sizeof(double));
    for (i = 0; i < sizeOfArr; i++)
        cpyOfArr[i] = eigenvaluesArr[i];
    
    qsort(cpyOfArr, sizeOfArr, sizeof(double), compEigenvalues);
    for (i = 0; i < floor(sizeOfArr/2); i++)
    {
        delta = cpyOfArr[i] - cpyOfArr[i+1];
        if (delta > maxDelta)
        {
            maxDelta = delta;
            maxI = i;
        }
    }
    free(cpyOfArr);
    return maxI + 1;    
}

int compEigenvalues(const void* elem1, const void* elem2){
    if(*(double*)elem1 > *(double*)elem2)
        return -1;
    if(*(double*)elem1 < *(double*)elem2)
        return 1;
    return 0;
}

double** computeJacobi(double** mtx, int sizeOfMtx, double* eigenvalues){ /*DESTROYS ORIGINAL MTX*/
    int i, isConverged = 0, iter = 0;
    double offA, offAtag, **output, **P, **Ptrans, **PA, **Atag, **prevOutput;
    output = callocMatrix(sizeOfMtx, sizeOfMtx);
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
        offAtag = calcOffSquared(Atag, sizeOfMtx);
        if (fabs(offA - offAtag) <= EPSILON)
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
    for (i = 0; i < sizeOfMtx; i++)
    {
        eigenvalues[i] = mtx[i][i];
    }
    Neo(mtx, sizeOfMtx);
    return output;
}

/* void roundMatrix(double** mtx, int n, int dim, int digits){ in place
    int i,j, multiplier;
    multiplier = (int)pow(10, digits);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < dim; j++)
        {
            if (mtx[i][j] > 0)
            {
                mtx[i][j] = (double)floor(mtx[i][j] * multiplier) / multiplier;
            }
            if (mtx[i][j] < 0)
            {
                mtx[i][j] = (double)ceil(mtx[i][j] * multiplier) / multiplier;
            }
        }
    }
} */

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
    output = callocMatrix(sizeOfMtx, sizeOfMtx);
    pvtPnt = getPivotPoint(mtx, sizeOfMtx);
    maxI = pvtPnt[0];
    maxJ = pvtPnt[1];
    sinCos = getSinCos(mtx[maxI][maxI], mtx[maxJ][maxJ], mtx[maxI][maxJ]);
    s = sinCos[0];
    c = sinCos[1];
    for (i = 0; i < sizeOfMtx; i++)
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
    t = getSign(theta)/(fabs(theta) + sqrt(pow(theta,2) + 1));
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
    return -1;
}

int* getPivotPoint(double** mtx, int sizeOfMtx){
    int i,j, *output;
    double maxVal = -INFINITY;
    output = calloc(2, sizeof(int));
    for (i = 0; i < sizeOfMtx; i++)
    {
        for (j = 0; j < sizeOfMtx; j++)
        {
            if (i != j && fabs(mtx[i][j]) > maxVal)
            {
                maxVal = fabs(mtx[i][j]);
                output[0] = i;
                output[1] = j;
            }
        }
    }
    return output;
}

double** computeNormalizedLaplacian(double** diagMtx, double** adjMatrix, int sizeOfMtx){
    int i,j;
    double **output, **DW, **DWD;
    output = callocMatrix(sizeOfMtx, sizeOfMtx);
    for (i = 0; i < sizeOfMtx; i++)
        diagMtx[i][i] = 1/(sqrt(diagMtx[i][i]));
    DW = matrixMultiply(diagMtx, adjMatrix, sizeOfMtx);
    DWD = matrixMultiply(DW, diagMtx, sizeOfMtx);  
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

double** diagDegMatrix(double** adjMatrix, int sizeOfMtx){
    int i,z;
    double** output;
    output = callocMatrix(sizeOfMtx, sizeOfMtx);
    for (i = 0; i < sizeOfMtx; i++)
    {
        for (z = 0; z < sizeOfMtx; z++)
        {
            output[i][i] += adjMatrix[i][z];
        }
    }
    return output;
}

double** createWeightAdjMatrix(double** mtx, int n, int dim){
    int i,j;
    double** output;
    output = callocMatrix(n, n);
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i != j)
            {
                output[i][j] = exp((-1*calcDistance(mtx[i],mtx[j], dim))/2);
            }
        }
    }
    return output;
}

double calcDistance(double* vec1, double* vec2, int vecDim){
    int i;
    double output = 0;
    for (i = 0; i < vecDim; i++)
    {
        output += pow(vec1[i] - vec2[i], 2);
    }
    return sqrt(output);
}

double** matrixMultiply(double** mtx1, double** mtx2, int sizeOfMtx){
    int i,j,k;
    double** output;
    output = callocMatrix(sizeOfMtx, sizeOfMtx);
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
    output = callocMatrix(sizeOfMtx, sizeOfMtx);
    for (i = 0; i < sizeOfMtx; i++)
    {
        for (j = 0; j < sizeOfMtx; j++)
        {
            output[i][j] = mtx[j][i];
        }
    }
    return output;    
}

void printMatrix(double** mtx, int rowCnt, int clmCnt){
    int i,j;
    for (i = 0; i < rowCnt; i++)
    {
        for (j = 0; j < clmCnt; j++)
        {
            printf("%.4f", mtx[i][j]);
            if (j != clmCnt - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }
}

double** callocMatrix(int rowCnt, int clmCnt){
    int i;
    double** output;
    output = calloc(rowCnt, sizeof(double*));
    for (i = 0; i < rowCnt; i++)
    {
        output[i] = calloc(clmCnt, sizeof(double));
    }
    return output;
}

void Neo(double** mtx, int rowCnt){ /*Free The Matrix*/
    int i;
    for (i = 0; i < rowCnt; i++)
    {
        free(mtx[i]);
    }
    free(mtx);
}

void throwError(int errorCode){
    switch (errorCode)
    {
    case 0:
        printf("Invalid Input!\n");
        break;
    case 1:
        printf("An Error Has Occured\n");
        break;
    }
    exit(1);
}

enum goal{
    wam,
    ddg,
    lnorm,
    jacobi
};