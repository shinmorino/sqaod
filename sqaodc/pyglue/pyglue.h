/* -*- c++ -*- */
#pragma once

#if defined(_WIN32) && defined(_DEBUG)
#  undef _DEBUG
#  include <Python.h>
#  define _DEBUG
#else
#  include <Python.h>
#endif
#include <bytesobject.h>


#if PY_MAJOR_VERSION >= 3

#define IsIntegerType(o) (PyLong_Check(o))

#else

#define IsIntegerType(o) (PyLong_Check(o) || PyInt_Check(o))

#endif


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>
#include <sqaodc/sqaodc.h>
#include <sqaodc/common/Common.h>
#include <algorithm>

namespace sq = sqaod;

template<class real>
struct NpMatrixType {
    typedef sqaod::MatrixType<real> Matrix;
    NpMatrixType(PyObject *pyObj) {
        PyArrayObject *arr = (PyArrayObject*)pyObj;
        real *data = (real*)PyArray_DATA(arr);
        assert(PyArray_NDIM(arr) == 2);
        sq::SizeType stride = (sq::SizeType)PyArray_STRIDE(arr, 0) / sizeof(real);
        mat.map(data, (sq::SizeType)PyArray_SHAPE(arr)[0], (sq::SizeType)PyArray_SHAPE(arr)[1], stride);
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
            int rows = (int)PyArray_SHAPE(arr)[0];
            int cols = (int)PyArray_SHAPE(arr)[1];
            throwErrorIf((rows != 1) && (cols != 1), "ndarray is not 1-diemsional.");
            size = std::max(rows, cols);
        }
        else /*if (PyArray_NDIM(arr) == 1) */  {
            size = (int)PyArray_SHAPE(arr)[0];
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
            int size = (int)PyArray_SHAPE(arr)[0];
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
    NpConstScalarType(PyObject *obj) {
        err = false;
        if (PyFloat_Check(obj)) {
            data = (real)PyFloat_AS_DOUBLE(obj);
        }
        else {
            data = (real)PyFloat_AsDouble(obj);
            if (data == -1.)
                err = (PyErr_Occurred() != NULL);
        }
    }
    
    operator real() {
        return data;
    }
    operator real() const {
        return data;
    }
    real data;
    bool err;
};


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

#define ASSERT_DTYPE(dtype) if (!isFloat32(dtype) && !isFloat64(dtype)) \
        {   PyErr_SetString(PyExc_RuntimeError, "dtype is not float64 nor float32."); \
            return NULL; }


typedef NpMatrixType<char> NpBitMatrix;
typedef NpVectorType<char> NpBitVector;


/* Helpers for String */
const char *getStringFromObject(PyObject *pyObj) {
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_AsUTF8(pyObj);
#else
    return PyString_AsString(pyObj);
#endif
}

/* Helpers for String */
bool isStringObject(PyObject *pyObj) {
#if PY_MAJOR_VERSION >= 3
    return PyUnicode_Check(pyObj);
#else
    return PyString_Check(pyObj);
#endif
}




/* Preference */

int parsePreference(const char *key, PyObject *valueObj, sqaod::Preference *pref) {

    sqaod::PreferenceName prefName = sqaod::preferenceNameFromString(key);
    switch (prefName) {
    case sqaod::pnAlgorithm: {
        if (!isStringObject(valueObj)) {
            PyErr_SetString(PyExc_RuntimeError, "algorithm value is not a string");
            return -1;
        }
        sqaod::Algorithm algo = sqaod::algorithmFromString(getStringFromObject(valueObj));
        *pref = sqaod::Preference(sqaod::pnAlgorithm, algo);
        return 0;
    }
    case sqaod::pnNumTrotters:
    case sqaod::pnTileSize:
    case sqaod::pnTileSize0:
    case sqaod::pnTileSize1: {
        if (IsIntegerType(valueObj)) {
            *pref = sqaod::Preference(prefName, PyLong_AsLong(valueObj));
            return 0;
        }
        else if (PyLong_Check(valueObj)) {
            *pref = sqaod::Preference(prefName, PyLong_AsLong(valueObj));
            return 0;
        }
        else {
            PyErr_SetString(PyExc_RuntimeError, "Not an integer value.");
            return -1;
        }
    }
    default:
        PyErr_SetString(PyExc_RuntimeError, "unknown preference name");
        return -1;
    }
}



inline
int parsePreferences(PyObject *pyObj, sqaod::Preferences *prefs) {

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
        assert(isStringObject(nameObj));
        const char *name = getStringFromObject(nameObj);
        PyObject *valueObj = PyTuple_GET_ITEM(tuple, 1);

        sqaod::Preference pref;
        if (parsePreference(name, valueObj, &pref) == -1)
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
    case sqaod::pnPrecision : {
        return Py_BuildValue("s", pref.precision);
    }
    case sqaod::pnDevice : {
        return Py_BuildValue("s", pref.device);
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
#define CATCH_ERROR_AND_RETURN                      \
        catch (const std::exception &e) {                   \
            PyErr_SetString(PyExc_RuntimeError, e.what());  \
            return NULL;                                    \
        }



#if PY_MAJOR_VERSION >= 3

#define INITFUNCNAME(name) PyInit_##name

#else

#define INITFUNCNAME(name) init##name

#endif



/* references 
 * http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
 * http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html
 *
 * Input validation is minimal in C++ side, assuming required checkes are done in python.
 */
