/* -*- c++ -*- */
#ifndef QD_PYGLUE_H__
#define QD_PYGLUE_H__

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <common/Matrix.h>
#include <common/Common.h>
#include <algorithm>


template<class real>
struct NpMatrixType {
    typedef sqaod::MatrixType<real> Matrix;
    NpMatrixType(PyObject *pyObj) {
        PyArrayObject *arr = (PyArrayObject*)pyObj;
        real *data = (real*)PyArray_DATA(arr);
        assert(PyArray_NDIM(arr) == 2);
        mat.map(data, PyArray_SHAPE(arr)[0], PyArray_SHAPE(arr)[1]);
    }

    void allocate(int nRows, int nCols) {
        /* new array object */
        npy_intp dims[2];
        dims[0] = nRows;
        dims[1] = nCols;
        obj = PyArray_EMPTY(2, dims, NPY_INT8, 0);
        PyArrayObject *arr = (PyArrayObject*)obj;
        /* setup members */
        char *data = (char*)PyArray_DATA(arr);
        mat.set(data, nRows, nCols);
    }
    
    /* accessor for ease of coding. */
    operator const Matrix&() const {
        return mat;
    }

    Matrix *operator&() {
        return &mat;
    }
    
    Matrix mat;
    PyObject *obj;
};


template<class real>
struct NpVectorType {
    typedef sqaod::VectorType<real> Vector;

    /* FIXME: npyType should be obtained by the type of real */
    NpVectorType(int _size, int npyType) {
        /* new array object */
        npy_intp size = _size;
        /* FIXME: get NPY_xx type from C++ type */
        obj = PyArray_EMPTY(1, &size, npyType, 0);
        PyArrayObject *arr = (PyArrayObject*)obj;
        /* setup members */
        real *data = (real*)PyArray_DATA(arr);
        vec.set(data, _size);
    }


    NpVectorType(PyObject *pyObj) {
        obj = pyObj;
        PyArrayObject *arr = (PyArrayObject*)pyObj;
        real *data = (real*)PyArray_DATA(arr);
        int size;
        throwErrorIf(3 <= PyArray_NDIM(arr), "ndarray is not 1-diemsional.");
        if (PyArray_NDIM(arr) == 2) {
            int rows = PyArray_SHAPE(arr)[0];
            int cols = PyArray_SHAPE(arr)[1];
            throwErrorIf((rows != 1) && (cols != 1), "ndarray is not 1-diemsional.");
            size = std::max(rows, cols);
        }
        else /*if (PyArray_NDIM(arr) == 1) */  {
            size = PyArray_SHAPE(arr)[0];
        }
        vec.set(data, size);
    }

    /* accessor for ease of coding. */
    operator const Vector&() const {
        return vec;
    }
    Vector *operator&() {
        return &vec;
    }
    
    
    Vector vec;
    PyObject *obj;
};


template<class real>
struct NpScalarRefType {
    typedef sqaod::VectorType<real> Vector;

    NpScalarRefType(PyObject *pyObj) {
        obj = pyObj;
        PyArrayObject *arr = (PyArrayObject*)pyObj;
        throwErrorIf(3 <= PyArray_NDIM(arr), "not a scalar.");
        if (PyArray_NDIM(arr) == 2) {
            int rows = PyArray_SHAPE(arr)[0];
            int cols = PyArray_SHAPE(arr)[1];
            throwErrorIf((rows != 1) || (cols != 1), "not a scalar.");
        }
        else if (PyArray_NDIM(arr) == 1) {
            int size = PyArray_SHAPE(arr)[0];
            throwErrorIf(size != 1, "not a scalar.");
        }
        data = (real*)PyArray_DATA(arr);
    }

    /* accessor for ease of coding. */
    operator real&() {
        return *data;
    }
    real *operator&() {
        return data;
    }
    
    real *data;
    PyObject *obj;
};


/* Const Scalar */

template<class real>
struct NpConstScalarType {
    NpConstScalarType(PyObject *obj);
    
    operator real() {
        return data;
    }
    operator real() const {
        return data;
    }
    real data;
};

template<> inline
NpConstScalarType<double>::NpConstScalarType(PyObject *obj) {
    PyFloat64ScalarObject *fpObj = (PyFloat64ScalarObject*)obj;
    data = fpObj->obval;
}

template<> inline
NpConstScalarType<float>::NpConstScalarType(PyObject *obj) {
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


typedef NpMatrixType<char> NpBitMatrix;
typedef NpVectorType<char> NpBitVector;


#endif
