#ifndef SPKMEANS_H_
#define SPKMEANS_H_

double** getMatrixFromFile(const char* fileName, int* oRowCnt, int* oClmCnt);
double** partialSpk(int *k, double** mtx, int rowCnt, int clmCnt);
int determineK(double* eigenvaluesArr, int sizeOfArr);
int compEigenvalues(const void* elem1, const void* elem2);
double** computeJacobi(double** mtx, int sizeOfMtx, double* eigenvalues);
double calcOffSquared(double** mtx, int sizeOfMtx);
double** createRotationMatrix(double** mtx, int sizeOfMtx);
double* getSinCos(double Aii, double Ajj, double Aij);
int getSign(double x);
int* getPivotPoint(double** mtx, int sizeOfMtx);
double** computeNormalizedLaplacian(double** invSqrtDiagMtx, double** adjMatrix, int sizeOfMtx);
double** diagDegMatrix(double** adjMatrix, int sizeOfMtx);
double** createWeightAdjMatrix(double** mtx, int rowsCnt, int clmCnt);
double calcDistance(double* vec1, double* vec2, int dim);
double** matrixMultiply(double** mtx1, double** mtx2, int sizeOfMtx);
double** matrixTranspose(double** mtx, int sizeOfMtx);
void printMatrix(double** mtx, int rowCnt, int clmCnt);
double** callocMatrix(int rowCnt, int clmCnt);
void Neo(double** mtx, int rowCnt);
void throwError(int errorCode);

#endif