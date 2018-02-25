/* -*- c++ -*- */
#pragma once

#if defined(_WIN32) && defined(_DEBUG)
#  undef _DEBUG
#  include <Python.h>
#  define _DEBUG
#else
#  include <Python.h>
#endif


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <sqaodc/sqaodc.h>
#include <sqaodc/common/Common.h>
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
        vec.map(data, _size);
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
        vec.map(data, size);
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
            int rows = (int)PyArray_SHAPE(arr)[0];
            int cols = (int)PyArray_SHAPE(arr)[1];
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

/* Preference */

int parsePreference(const char *key, PyObject *valueObj,
                    sqaod::Preference *pref, PyObject *errObj) {

    sqaod::PreferenceName prefName = sqaod::preferenceNameFromString(key);
    switch (prefName) {
    case sqaod::pnAlgorithm: {
        if (!PyString_Check(valueObj)) {
            PyErr_SetString(errObj, "algorithm value is not a string");
            return -1;
        }
        sqaod::Algorithm algo = sqaod::algorithmFromString(PyString_AsString(valueObj));
        *pref = sqaod::Preference(sqaod::pnAlgorithm, algo);
        return 0;
    }
    case sqaod::pnNumTrotters:
    case sqaod::pnTileSize:
    case sqaod::pnTileSize0:
    case sqaod::pnTileSize1: {
        if (PyInt_Check(valueObj)) {
            PyIntObject *intObj = (PyIntObject*)valueObj;
            *pref = sqaod::Preference(prefName, (sqaod::SizeType)intObj->ob_ival);
            return 0;
        }
        else if (PyLong_Check(valueObj)) {
            *pref = sqaod::Preference(prefName, PyLong_AsLong(valueObj));
            return 0;
        }
        else {
            PyErr_SetString(errObj, "Not an integer value.");
            return -1;
        }
    }
    default:
        PyErr_SetString(errObj, "unknown preference name");
        return -1;
    }
}



inline
int parsePreferences(PyObject *pyObj, sqaod::Preferences *prefs, PyObject *errObj) {

    if (!PyDict_Check(pyObj)) {
        abort_("Unexpected object.");
        return -1;
    }

    PyListObject *list = (PyListObject*)PyDict_Items(pyObj);
    int nItems = (int)PyList_GET_SIZE(list);
    prefs->reserve(nItems);
    for (int idx = 0; idx < nItems; ++idx) {
        PyObject *item = PyList_GET_ITEM(list, idx);
        assert(PyTuple_Check(item));
        PyTupleObject *tuple = (PyTupleObject*)item;
        PyObject *nameObj = PyTuple_GET_ITEM(tuple, 0);
        assert(PyString_Check(nameObj));
        const char *name = PyString_AsString(nameObj);
        PyObject *valueObj = PyTuple_GET_ITEM(tuple, 1);

        sqaod::Preference pref;
        if (parsePreference(name, valueObj, &pref, errObj) == -1)
            return -1;
        prefs->pushBack(pref);
    }
    Py_DECREF(list);
    return nItems;
}


PyObject *createPreferenceValue(const sqaod::Preference &pref) {
    switch (pref.name) {
    case sqaod::pnAlgorithm: {
        const char *algoName = sqaod::algorithmToString(pref.algo);
        return Py_BuildValue("s", algoName);
    }
    case sqaod::pnNumTrotters:
    case sqaod::pnTileSize:
    case sqaod::pnTileSize0:
    case sqaod::pnTileSize1: {
        return Py_BuildValue("i", pref.size);
    }
    default:
        abort_("Must not reach here.");
        return NULL;
    }
}

inline
PyObject *createPreferences(const sqaod::Preferences &prefs) {
    PyObject *dictObj = PyDict_New();
    for (int idx = 0; idx < (sqaod::IdxType)prefs.size(); ++idx) {
        const sqaod::Preference &pref = prefs[idx];
        const char *name = sqaod::preferenceNameToString(pref.name);
        PyObject *valueObj = createPreferenceValue(pref);
        PyDict_SetItemString(dictObj, name, valueObj);
    }
    return dictObj;
}


/* exception handling macro */

#define TRY try
#define CATCH_ERROR_AND_RETURN(errObj)                      \
        catch (const std::exception &e) {                   \
            PyErr_SetString(errObj, e.what());              \
            return NULL;                                    \
        }

#define RAISE_INVALID_DTYPE(dtype, errObj)                                     \
        {                                                               \
            PyErr_SetString(errObj, "dtype must be numpy.float64 or numpy.float32."); \
            return NULL; \
        }

/* references 
 * http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
 * http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html
 *
 * Input validation is minimal in C++ side, assuming required checkes are done in python.
 */
