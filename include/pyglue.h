/* -*- c++ -*- */
#ifndef QD_PYGLUE_H__
#define QD_PYGLUE_H__

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

template<class real>
struct NpMatrixT {
    NpMatrixT(PyObject *obj);
    
    real *data;
    int nDims;
    int dims[2];

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
    dims[0] = dims[1] = -1;
    PyArrayObject *arr = (PyArrayObject*)obj;
    data = (real*)PyArray_DATA(arr);
    nDims = PyArray_NDIM(arr);
    for (int idx = 0; idx < nDims; ++idx)
        dims[idx] = PyArray_SHAPE(arr)[idx];
}


typedef NpMatrixT<char> NpBitMatrix;




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


#endif
