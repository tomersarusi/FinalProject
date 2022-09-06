#include <Python.h>
#include "spkmeans.h"
#include "kmeans.h"

static PyObject *pyKmeans(PyObject *self, PyObject *args);
static PyObject *wam(PyObject *self, PyObject *args);
static PyObject *ddg(PyObject *self, PyObject *args);
static PyObject *lnorm(PyObject *self, PyObject *args);
static PyObject *jacobi(PyObject *self, PyObject *args);
static PyObject *pyPartialSpk(PyObject *self, PyObject *args);
static PyObject* pyMatrixFromDoubleArray(double **mtx, int rowCnt, int clmCnt);

static PyMethodDef methods[] = {
    {"kmeans", (PyCFunction) pyKmeans, METH_VARARGS, PyDoc_STR("kmeans")},
    {"wam", (PyCFunction) wam, METH_VARARGS, PyDoc_STR("wam")},
    {"ddg", (PyCFunction) ddg, METH_VARARGS, PyDoc_STR("ddg")},
    {"lnorm", (PyCFunction) lnorm, METH_VARARGS, PyDoc_STR("lnorm")},
    {"jacobi", (PyCFunction) jacobi, METH_VARARGS, PyDoc_STR("jacobi")},
    {"partialSpk", (PyCFunction) pyPartialSpk, METH_VARARGS, PyDoc_STR("partialSpk")},
    {NULL, NULL, 0 , NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "spkmeansmodule",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC
PyInit_spkmeansmodule(void){
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m)
    {
        return NULL;
    }
    return m;
}

static PyObject *pyKmeans(PyObject *self, PyObject *args){
    int i, j, k, dim, numOfRows, maxIter;
    PyObject *kmeansListFromPython, *datapointsFromPython, *resultPython;
    double epsilon, **kmeansList, **datapoints, **result;

    if (!PyArg_ParseTuple(args, "iiiidOO", &k, &dim, &numOfRows, &maxIter, &epsilon, &datapointsFromPython, &kmeansListFromPython)) /*deconstructing the input args*/
    {
        return NULL;
    }
    datapoints = calloc(numOfRows, sizeof(double*));
    for (i = 0; i < numOfRows; i++)
    {
        datapoints[i] = calloc(dim, sizeof(double));
        for (j = 0; j < dim; j++)
        {
            datapoints[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(datapointsFromPython, i), j)); /*building a 2d array of doubles, with the values being extracted from the pyObject representing the datapoints list*/
        }
    }
    kmeansList = calloc(k, sizeof(double*));
    for (i = 0; i < k; i++)
    {
        kmeansList[i] = calloc(dim, sizeof(double));
        for (j = 0; j < dim; j++)
        {
            
            kmeansList[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(kmeansListFromPython, i), j)); /*same as datapoints*/
        }
    }
    result = kmeans(k, dim, numOfRows, maxIter, epsilon, datapoints, kmeansList); /*run the kmeans algorithm*/
    resultPython = pyMatrixFromDoubleArray(result, k, dim);
    for (i = 0; i < k; i++)
    {
        free(result[i]);
    }
    free(result);

    return resultPython;
}

static PyObject *wam(PyObject *self, PyObject *args){
    int rowCnt, clmCnt;
    char* fileName;
    double **mtx, **wamMtx;
    PyObject *output;
    if (!PyArg_ParseTuple(args, "s", &fileName)) /*deconstructing the input args*/
    {
        return NULL;
    }
    mtx = getMatrixFromFile(fileName, &rowCnt, &clmCnt);
    wamMtx = createWeightAdjMatrix(mtx, rowCnt, clmCnt);
    output = pyMatrixFromDoubleArray(wamMtx, rowCnt, rowCnt);
    Neo(mtx, rowCnt);
    Neo(wamMtx, rowCnt);
    return output;
}

static PyObject *ddg(PyObject *self, PyObject *args){
    int rowCnt, clmCnt;
    char* fileName;
    double **mtx, **wamMtx, **ddgMatrix;
    PyObject *output;
    if (!PyArg_ParseTuple(args, "s", &fileName)) /*deconstructing the input args*/
    {
        return NULL;
    }
    mtx = getMatrixFromFile(fileName, &rowCnt, &clmCnt);
    wamMtx = createWeightAdjMatrix(mtx, rowCnt, clmCnt);
    ddgMatrix = diagDegMatrix(wamMtx, rowCnt);
    output = pyMatrixFromDoubleArray(ddgMatrix, rowCnt, rowCnt);
    Neo(mtx, rowCnt);
    Neo(wamMtx, rowCnt);
    Neo(ddgMatrix, rowCnt);
    return output;
}

static PyObject *lnorm(PyObject *self, PyObject *args){
    int rowCnt, clmCnt;
    char* fileName;
    double **mtx, **wamMtx, **ddgMatrix, **lnormMatrix;
    PyObject *output;
    if (!PyArg_ParseTuple(args, "s", &fileName)) /*deconstructing the input args*/
    {
        return NULL;
    }
    mtx = getMatrixFromFile(fileName, &rowCnt, &clmCnt);
    wamMtx = createWeightAdjMatrix(mtx, rowCnt, clmCnt);
    ddgMatrix = diagDegMatrix(wamMtx, rowCnt);
    lnormMatrix = computeNormalizedLaplacian(ddgMatrix, wamMtx, rowCnt);
    output = pyMatrixFromDoubleArray(lnormMatrix, rowCnt, rowCnt);
    Neo(mtx, rowCnt);
    Neo(wamMtx, rowCnt);
    Neo(ddgMatrix, rowCnt);
    Neo(lnormMatrix, rowCnt);
    return output;
}

static PyObject *jacobi(PyObject *self, PyObject *args){
    int i, rowCnt, clmCnt;
    char* fileName;
    double **mtx, *eigenvalues, **eigenvectors;
    PyObject *output, *row;
    if (!PyArg_ParseTuple(args, "s", &fileName)) /*deconstructing the input args*/
    {
        return NULL;
    }
    mtx = getMatrixFromFile(fileName, &rowCnt, &clmCnt);
    eigenvalues = calloc(rowCnt, sizeof(double));
    eigenvectors = computeJacobi(mtx, rowCnt, eigenvalues);
    output = pyMatrixFromDoubleArray(eigenvectors, rowCnt, rowCnt);
    row = PyList_New(0);
    for (i = 0; i < rowCnt; i++)
    {
        PyList_Append(row, PyFloat_FromDouble(eigenvalues[i]));
    }
    PyList_Insert(output, 0, row);
    free(eigenvalues);
    Neo(eigenvectors, rowCnt);
    return output;
}

static PyObject *pyPartialSpk(PyObject *self, PyObject *args){
    int k, n, dim;
    char* fileName;
    double **mtx, **kVectors;
    PyObject *output;
    if (!PyArg_ParseTuple(args, "is", &k, &fileName)) /*deconstructing the input args*/
    {
        return NULL;
    }
    mtx = getMatrixFromFile(fileName, &n, &dim);
    kVectors = partialSpk(&k, mtx, n, dim);
    output = pyMatrixFromDoubleArray(kVectors, n, k);
    Neo(kVectors, n);
    Neo(mtx, n);
    return output;
}

static PyObject* pyMatrixFromDoubleArray(double **mtx, int rowCnt, int clmCnt){
    int i, j;
    PyObject *pyMatrix, *row;
    pyMatrix = PyList_New(0); /*create an pyObject of an empty list*/
    for (i = 0; i < rowCnt; i++)
    {
        row = PyList_New(0);
        for (j = 0; j < clmCnt; j++)
        {
            PyList_Append(row, PyFloat_FromDouble(mtx[i][j])); /*appending the kmeans values to the pyobject list*/
        }
        PyList_Append(pyMatrix, row);
    }
    return pyMatrix;
}