#ifndef SPKMEANS_H_
#define SPKMEANS_H_

double** getMatrixFromFile(const char* fileName, int* oRowCnt, int* oClmCnt);
double** computeWeightAdjMatrix(double** mtx, int rowsCnt, int clmCnt);
double** computeDiagDegMatrix(double** adjMatrix, int sizeOfMtx);
double** computeNormalizedLaplacian(double** invSqrtDiagMtx, double** adjMatrix, int sizeOfMtx);
double** computeJacobi(double** mtx, int sizeOfMtx, double* eigenvalues);
double** computePartialSpk(int *k, double** mtx, int rowCnt, int clmCnt);
double** computeRotationMatrix(double** mtx, int sizeOfMtx);
int determineK(double* eigenvaluesArr, int sizeOfArr);
double* getSinCos(double Aii, double Ajj, double Aij);
int* getPivotPoint(double** mtx, int sizeOfMtx);
double calcDistance(double* vec1, double* vec2, int dim);
double calcOffSquared(double** mtx, int sizeOfMtx);
int compEigenvalues(const void* elem1, const void* elem2);
int getSign(double x);
double** callocMatrix(int rowCnt, int clmCnt);
double** copyMatrix(double** mtx, int n, int dim);
double** matrixMultiply(double** mtx1, double** mtx2, int sizeOfMtx);
double** matrixTranspose(double** mtx, int sizeOfMtx);
void printMatrix(double** mtx, int rowCnt, int clmCnt);
void Neo(double** mtx, int rowCnt);
void throwError(int errorCode);

#endif