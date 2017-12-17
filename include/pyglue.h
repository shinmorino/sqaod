/* -*- c++ -*- */
#ifndef QD_PYGLUE_H__
#define QD_PYGLUE_H__

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

template<class real>
struct NpMatrixT {
    NpMatrixT() { }
    NpMatrixT(PyObject *obj);
    
    real *data;
    int nDims;
    npy_intp dims[2];

    /* accessor for ease of coding. */
    operator real*() {
        return data;
    }
    operator real*() const {
        return data;
    }
};

template<class real>
inline
NpMatrixT<real>::NpMatrixT(PyObject *obj) {
    dims[0] = dims[1] = 1;
    PyArrayObject *arr = (PyArrayObject*)obj;
    data = (real*)PyArray_DATA(arr);
    nDims = PyArray_NDIM(arr);
    for (int idx = 0; idx < nDims; ++idx)
        dims[idx] = PyArray_SHAPE(arr)[idx];
}


struct NpBitMatrix : public NpMatrixT<char> {
    NpBitMatrix() { }
    NpBitMatrix(PyObject *obj) : NpMatrixT<char>(obj) { }
    void allocate(int nRows, int nCols) {
        /* new array object */
        dims[0] = nRows;
        dims[1] = nCols;
        obj = PyArray_EMPTY(2, dims, NPY_INT8, 0);
        PyArrayObject *arr = (PyArrayObject*)obj;
        /* setup members */
        data = (char*)PyArray_DATA(arr);
    }
    PyObject *obj;
};




template<class real>
struct NpConstScalarT {
    NpConstScalarT(PyObject *obj);
    
    operator real() {
        return data;
    }
    operator real() const {
        return data;
    }
    real data;
};

template<> inline
NpConstScalarT<double>::NpConstScalarT(PyObject *obj) {
    PyFloat64ScalarObject *fpObj = (PyFloat64ScalarObject*)obj;
    data = fpObj->obval;
}

template<> inline
NpConstScalarT<float>::NpConstScalarT(PyObject *obj) {
    PyFloat32ScalarObject *fpObj = (PyFloat32ScalarObject*)obj;
    data = fpObj->obval;
}


inline
PyObject *newScalarObj(double v) {
    PyObject *obj = PyArrayScalar_New(Float64);
    PyArrayScalar_ASSIGN(obj, Float64, (npy_float64)v);
    return obj;
}

inline
PyObject *newScalarObj(float v) {
    PyObject *obj = PyArrayScalar_New(Float32);
    PyArrayScalar_ASSIGN(obj, Float32, (npy_float32)v);
    return obj;
}



/* Helpers for dtypes */
inline
bool isFloat64(PyObject *dtype) {
    return dtype == (PyObject*)&PyFloat64ArrType_Type;
}
inline
bool isFloat32(PyObject *dtype) {
    return dtype == (PyObject*)&PyFloat32ArrType_Type;
}


#endif
