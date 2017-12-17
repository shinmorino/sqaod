#include <pyglue.h>
#include <cpu/Traits.h>


// http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
// http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

/* NOTE: Value type checks for python objs have been already done in python glue, 
 * Here we only get entities needed. */


static PyObject *Cpu_NativeError;
namespace sqd = sqaod;



namespace {


void setErrInvalidDtype(PyObject *dtype) {
    PyErr_SetString(Cpu_NativeError, "dtype must be numpy.float64 or numpy.float32.");
}

#define RAISE_INVALID_DTYPE(dtype) {setErrInvalidDtype(dtype); return NULL; }
    
    
template<class real>
void internal_dense_graph_calculate_E(PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix E(objE), W(objW);
    NpBitMatrix x(objX);
    /* do the native job */
    sqd::DGFuncs<real>::calculate_E(E, W, x, x.dims[0]);
}

    
extern "C"
PyObject *cpu_native_dense_graph_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE, *objW, *objX;
    PyObject *dtype;
    
    if (!PyArg_ParseTuple(args, "OOOO", &objE, &objW, &objX, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_dense_graph_calculate_E<double>(objE, objW, objX);
    else if (isFloat32(dtype))
        internal_dense_graph_calculate_E<float>(objE, objW, objX);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dense_graph_batch_calculate_E(PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixT<real> NpMatrix;
    
    NpMatrix E(objE), W(objW);
    NpBitMatrix x(objX);
    /* do the native job */
    sqd::DGFuncs<real>::batchCalculate_E(E, W, x, x.dims[1], x.dims[0]);
}

extern "C"
PyObject *cpu_native_dense_graph_batch_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE, *objW, *objX;
    PyObject *dtype = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &objE, &objW, &objX, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_dense_graph_batch_calculate_E<double>(objE, objW, objX);
    else if (isFloat32(dtype))
        internal_dense_graph_batch_calculate_E<float>(objE, objW, objX);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    


template<class real>
void internal_dense_graph_calculate_hJc(PyObject *objH, PyObject *objJ, PyObject *objC,
                                        PyObject *objW) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix h(objH), J(objJ), c(objC), W(objW);
    /* do the native job */
    sqd::DGFuncs<real>::calculate_hJc(h, J, c, W, W.dims[0]);
}


extern "C"
PyObject *cpu_native_dense_graph_calculate_hJc(PyObject *module, PyObject *args) {
    PyObject *objH, *objJ, *objC, *objW;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objH, &objJ, &objC, &objW, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_dense_graph_calculate_hJc<double>(objH, objJ, objC, objW);
    else if (isFloat32(dtype))
        internal_dense_graph_calculate_hJc<float>(objH, objJ, objC, objW);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
template<class real> void
internal_dense_graph_calculate_E_from_qbits(PyObject *objE,
                                            PyObject *objH, PyObject *objJ, PyObject *objC,
                                            PyObject *objQ) {
    typedef NpMatrixT<real> NpMatrix;
    typedef NpConstScalarT<real> NpConstScalar;
    NpMatrix E(objE), h(objH), J(objJ);
    NpConstScalar c(objC);
    NpBitMatrix q(objQ);
    /* do the native job */
    sqd::DGFuncs<real>::calculate_E_fromQbits(E, h, J, c, q, q.dims[0]);
}
    

extern "C"
PyObject *cpu_native_dense_graph_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE, *objH, *objJ, *objC, *objQ;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objE, &objH, &objJ, &objC, &objQ, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_dense_graph_calculate_E_from_qbits<double>(objE, objH, objJ, objC, objQ);
    else if (isFloat32(dtype))
        internal_dense_graph_calculate_E_from_qbits<float>(objE, objH, objJ, objC, objQ);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real> void
internal_dense_graph_batch_calculate_E_from_qbits(PyObject *objE,
                                                  PyObject *objH, PyObject *objJ, PyObject *objC,
                                                  PyObject *objQ) {
    typedef NpMatrixT<real> NpMatrix;
    typedef NpConstScalarT<real> NpConstScalar;
    NpMatrix E(objE), h(objH), J(objJ);
    NpConstScalar c(objC);
    NpBitMatrix q(objQ);
    /* do the native job */
    sqd::DGFuncs<real>::batchCalculate_E_fromQbits(E, h, J, c, q, q.dims[1], q.dims[0]);
}
    
extern "C"
PyObject *cpu_native_dense_graph_batch_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE, *objH, *objJ, *objC, *objQ;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objE, &objH, &objJ, &objC, &objQ, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_dense_graph_batch_calculate_E_from_qbits<double>(objE, objH, objJ, objC, objQ);
    else if (isFloat32(dtype))
        internal_dense_graph_batch_calculate_E_from_qbits<float>(objE, objH, objJ, objC, objQ);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


/* RBM */

    

template<class real> void
internal_rbm_calculate_E(PyObject *objE,
                         PyObject *objB0, PyObject *objB1, PyObject *objW,
                         PyObject *objX0, PyObject *objX1) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix b0(objB0), b1(objB1), W(objW);
    const NpMatrix E(objE); 
    NpBitMatrix x0(objX0), x1(objX1);
    /* do the native job */
    sqd::RBMFuncs<real>::calculate_E(E, b0, b1, W, x0, x1,
                                       x0.dims[x0.nDims - 1], x1.dims[x1.nDims - 1]);
}
    
extern "C"
PyObject *cpu_native_rbm_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE, *objB0, *objB1, *objW, *objX0, *objX1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_rbm_calculate_E<double>(objE, objB0, objB1, objW, objX0, objX1);
    else if (isFloat32(dtype))
        internal_rbm_calculate_E<float>(objE, objB0, objB1, objW, objX0, objX1);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real> void
internal_rbm_batch_calculate_E(PyObject *objE,
                               PyObject *objB0, PyObject *objB1, PyObject *objW,
                               PyObject *objX0, PyObject *objX1) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix E(objE), b0(objB0), b1(objB1), W(objW);
    NpBitMatrix x0(objX0), x1(objX1);
    /* do the native job */
    int N0, N1, nBatch0, nBatch1;
    if (x0.nDims == 1) {
        N0 = x0.dims[0];
        nBatch0 = 1;
    }
    else {
        N0 = x0.dims[1];
        nBatch0 = x0.dims[0];
    }
    if (x1.nDims == 1) {
        N1 = x1.dims[0];
        nBatch1 = 1;
    }
    else {
        N1 = x1.dims[1];
        nBatch1 = x1.dims[0];
    }
    sqd::RBMFuncs<real>::batchCalculate_E(E, b0, b1, W, x0, x1,
                                            N0, N1, nBatch0, nBatch1);
}
    
extern "C"
PyObject *cpu_native_rbm_batch_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE, *objB0, *objB1, *objW, *objX0, *objX1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_rbm_batch_calculate_E<double>(objE, objB0, objB1, objW, objX0, objX1);
    else if (isFloat32(dtype))
        internal_rbm_batch_calculate_E<float>(objE, objB0, objB1, objW, objX0, objX1);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
    
template<class real>
void internal_rbm_calculate_hJc(PyObject *objH0, PyObject *objH1, PyObject *objJ,
                                PyObject *objC,
                                PyObject *objB0, PyObject *objB1, PyObject *objW) {
    typedef NpMatrixT<real> NpMatrix;
    const NpMatrix b0(objB0), b1(objB1), W(objW);
    NpMatrix h0(objH0), h1(objH1), J(objJ), c(objC);
    /* do the native job */
    sqd::RBMFuncs<real>::calculate_hJc(h0, h1, J, c, b0, b1, W, W.dims[1], W.dims[0]);
}


extern "C"
PyObject *cpu_native_rbm_calculate_hJc(PyObject *module, PyObject *args) {
    PyObject *objH0, *objH1, *objJ, *objC, *objB0, *objB1, *objW;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &objH0, &objH1, &objJ, &objC,
                          &objB0, &objB1, &objW, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_rbm_calculate_hJc<double>(objH0, objH1, objJ, objC,
                                           objB0, objB1, objW);
    else if (isFloat32(dtype))
        internal_rbm_calculate_hJc<float>(objH0, objH1, objJ, objC,
                                          objB0, objB1, objW);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real> void
internal_rbm_calculate_E_from_qbits(PyObject *objE,
                                    PyObject *objH0, PyObject *objH1, PyObject *objJ, PyObject *objC,
                                    PyObject *objQ0, PyObject *objQ1) {
    typedef NpMatrixT<real> NpMatrix;
    typedef NpConstScalarT<real> NpConstScalar;
    NpMatrix E(objE);
    const NpMatrix h0(objH0), h1(objH1), J(objJ);
    NpConstScalar c(objC);
    const NpBitMatrix q0(objQ0), q1(objQ1);
    /* do the native job */
    sqd::RBMFuncs<real>::calculate_E_fromQbits(E, h0, h1, J, c, q0, q1, q0.dims[0], q1.dims[0]);
}
    
extern "C"
PyObject *cpu_native_rbm_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE, *objH0, *objH1, *objJ, *objC, *objQ0, *objQ1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &objE, &objH0, &objH1, &objJ, &objC,
                          &objQ0, &objQ1, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_rbm_calculate_E_from_qbits<double>(objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    else if (isFloat32(dtype))
        internal_rbm_calculate_E_from_qbits<float>(objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real> void
internal_rbm_batch_calculate_E_from_qbits(PyObject *objE,
                                          PyObject *objH0, PyObject *objH1, PyObject *objJ, PyObject *objC,
                                          PyObject *objQ0, PyObject *objQ1) {
    typedef NpMatrixT<real> NpMatrix;
    typedef NpConstScalarT<real> NpConstScalar;
    NpMatrix E(objE), h0(objH0), h1(objH1), J(objJ);
    NpConstScalar c(objC);
    NpBitMatrix q0(objQ0), q1(objQ1);
    /* do the native job */
    int N0 = J.dims[1], N1 = J.dims[0];
    int nBatch0 = (q0.nDims == 1) ? 1 : q0.dims[0];
    int nBatch1 = (q1.nDims == 1) ? 1 : q1.dims[0];
    sqd::RBMFuncs<real>::
        batchCalculate_E_fromQbits(E, h0, h1, J, c, q0, q1, N0, N1, nBatch0, nBatch1);
}
    
extern "C"
PyObject *cpu_native_rbm_batch_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE, *objH0, *objH1, *objJ, *objC, *objQ0, *objQ1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &objE, &objH0, &objH1, &objJ, &objC,
                          &objQ0, &objQ1, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        internal_rbm_batch_calculate_E_from_qbits<double>(objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    else if (isFloat32(dtype))
        internal_rbm_batch_calculate_E_from_qbits<float>(objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}



/* Solver */

template<class real>
PyObject *internal_dense_graph_batch_search(PyObject *objE, PyObject *objW, int xBegin, int xEnd) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix E(objE), W(objW);
    int N = W.dims[0];
    /* do the native job */
    sqd::PackedBitsArray xList;
    sqd::DGFuncs<real>::batchSearch(E, &xList, W, N, xBegin, xEnd);
    /* copy values to PyArray(int8). */
    npy_intp dims[2] = {(int)xList.size(), N};
    PyArrayObject *objX = (PyArrayObject*)PyArray_EMPTY(2, dims, NPY_INT8, 0);
    char *data = (char*)PyArray_DATA(objX);
    memcpy(data, xList.data(), sizeof(char) * xList.size() * N);
    return (PyObject*)objX;
}
    

extern "C"
PyObject *cpu_native_dense_graph_batch_search(PyObject *module, PyObject *args) {
    PyObject *objE, *objW, *objX;
    PyObject *dtype;
    unsigned long long xBegin = 0, xEnd = 0;
    std::vector<unsigned long long> xList;
    
    if (!PyArg_ParseTuple(args, "OOiiO", &objE, &objW, &xBegin, &xEnd, &dtype))
        return NULL;
    
    if (isFloat64(dtype))
        objX = internal_dense_graph_batch_search<double>(objE, objW, xBegin, xEnd);
    else if (isFloat32(dtype))
        objX = internal_dense_graph_batch_search<float>(objE, objW, xBegin, xEnd);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(objX);
    return objX;    
}
}



static
PyMethodDef annealermethods[] = {
	{"dense_graph_calculate_E", cpu_native_dense_graph_calculate_E, METH_VARARGS},
	{"dense_graph_batch_calculate_E", cpu_native_dense_graph_batch_calculate_E, METH_VARARGS},
	{"dense_graph_calculate_hJc", cpu_native_dense_graph_calculate_hJc, METH_VARARGS},
	{"dense_graph_calculate_E_from_qbits", cpu_native_dense_graph_calculate_E_from_qbits, METH_VARARGS},
	{"dense_graph_batch_calculate_E_from_qbits", cpu_native_dense_graph_batch_calculate_E_from_qbits, METH_VARARGS},
	{"rbm_calculate_E", cpu_native_rbm_calculate_E, METH_VARARGS},
	{"rbm_batch_calculate_E", cpu_native_rbm_batch_calculate_E, METH_VARARGS},
	{"rbm_calculate_hJc", cpu_native_rbm_calculate_hJc, METH_VARARGS},
	{"rbm_calculate_E_from_qbits", cpu_native_rbm_calculate_E_from_qbits, METH_VARARGS},
	{"rbm_batch_calculate_E_from_qbits", cpu_native_rbm_batch_calculate_E_from_qbits, METH_VARARGS},
	{"dense_graph_batch_search", cpu_native_dense_graph_batch_search, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcpu_native(void) {
    PyObject *m;
    
    m = Py_InitModule("cpu_native", annealermethods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cpu_native.error";
    Cpu_NativeError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cpu_NativeError);
    PyModule_AddObject(m, "error", Cpu_NativeError);
}
