#include <pyglue.h>
#include <cpu/Traits.h>


// http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
// http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

/* NOTE: Value type checks for python objs have been already done in python glue, 
 * Here we only get entities needed. */


static PyObject *Cpu_NativeError;
namespace qdcpu = quantd_cpu;

#define DONT_REACH_HERE {printf("Don't reach here, %s:%d\n", __FILE__, (int)__LINE__); abort();}


namespace {

    
template<class real>
void internal_dense_graph_calculate_E(PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix E(objE), W(objW);
    NpBitMatrix x(objX);
    /* do the native job */
    qdcpu::SolverTraits<real>::denseGraphCalculate_E(E, W, x, x.dims[0]);
}

    
extern "C"
PyObject *cpu_native_dense_graph_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE = NULL, *objW = NULL, *objX = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOi", &objE, &objW, &objX, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_dense_graph_calculate_E<double>(objE, objW, objX);
    else if (nPrec == 32)
        internal_dense_graph_calculate_E<float>(objE, objW, objX);
    else
        DONT_REACH_HERE;

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dense_graph_batch_calculate_E(PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixT<real> NpMatrix;
    
    NpMatrix E(objE), W(objW);
    NpBitMatrix x(objX);
    /* do the native job */
    qdcpu::SolverTraits<real>::denseGraphBatchCalculate_E(E, W, x, x.dims[1], x.dims[0]);
}

extern "C"
PyObject *cpu_native_dense_graph_batch_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE = NULL, *objW = NULL, *objX = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOi", &objE, &objW, &objX, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_dense_graph_batch_calculate_E<double>(objE, objW, objX);
    else if (nPrec == 32)
        internal_dense_graph_batch_calculate_E<float>(objE, objW, objX);
    else
        DONT_REACH_HERE;

    Py_INCREF(Py_None);
    return Py_None;    
}
    


template<class real>
void internal_dense_graph_calculate_hJc(PyObject *objH, PyObject *objJ, PyObject *objC,
                                        PyObject *objW) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix h(objH), J(objJ), c(objC), W(objW);
    /* do the native job */
    qdcpu::SolverTraits<real>::denseGraphCalculate_hJc(h, J, c, W, W.dims[0]);
}


extern "C"
PyObject *cpu_native_dense_graph_calculate_hJc(PyObject *module, PyObject *args) {
    PyObject *objH = NULL, *objJ = NULL, *objC = NULL, *objW = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOOi", &objH, &objJ, &objC, &objW, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_dense_graph_calculate_hJc<double>(objH, objJ, objC, objW);
    else if (nPrec == 32)
        internal_dense_graph_calculate_hJc<float>(objH, objJ, objC, objW);
    else
        DONT_REACH_HERE;

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
    qdcpu::SolverTraits<real>::denseGraphCalculate_E_fromQbits(E, h, J, c, q, q.dims[0]);
}
    

extern "C"
PyObject *cpu_native_dense_graph_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE = NULL, *objH = NULL, *objJ = NULL, *objC = NULL, *objQ = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOOOi", &objE, &objH, &objJ, &objC, &objQ, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_dense_graph_calculate_E_from_qbits<double>(objE, objH, objJ, objC, objQ);
    else if (nPrec == 32)
        internal_dense_graph_calculate_E_from_qbits<float>(objE, objH, objJ, objC, objQ);
    else
        DONT_REACH_HERE;

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
    qdcpu::SolverTraits<real>::denseGraphBatchCalculate_E_fromQbits(E, h, J, c, q, q.dims[1], q.dims[0]);
}
    
extern "C"
PyObject *cpu_native_dense_graph_batch_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE = NULL, *objH = NULL, *objJ = NULL, *objC = NULL, *objQ = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOOOi", &objE, &objH, &objJ, &objC, &objQ, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_dense_graph_batch_calculate_E_from_qbits<double>(objE, objH, objJ, objC, objQ);
    else if (nPrec == 32)
        internal_dense_graph_batch_calculate_E_from_qbits<float>(objE, objH, objJ, objC, objQ);
    else
        DONT_REACH_HERE;

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
    qdcpu::SolverTraits<real>::rbmCalculate_E(E, b0, b1, W, x0, x1,
                                              x0.dims[x0.nDims - 1], x1.dims[x1.nDims - 1]);
}
    
extern "C"
PyObject *cpu_native_rbm_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE = NULL,
        *objB0 = NULL, *objB1 = NULL, *objW = NULL,
        *objX0 = NULL, *objX1 = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOi",
                          &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_rbm_calculate_E<double>(objE, objB0, objB1, objW, objX0, objX1);
    else if (nPrec == 32)
        internal_rbm_calculate_E<float>(objE, objB0, objB1, objW, objX0, objX1);
    else
        DONT_REACH_HERE;

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
    qdcpu::SolverTraits<real>::rbmBatchCalculate_E(E, b0, b1, W, x0, x1,
                                                   N0, N1, nBatch0, nBatch1);
}
    
extern "C"
PyObject *cpu_native_rbm_batch_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE = NULL,
        *objB0 = NULL, *objB1 = NULL, *objW = NULL,
        *objX0 = NULL, *objX1 = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOi",
                          &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_rbm_batch_calculate_E<double>(objE, objB0, objB1, objW, objX0, objX1);
    else if (nPrec == 32)
        internal_rbm_batch_calculate_E<float>(objE, objB0, objB1, objW, objX0, objX1);
    else
        DONT_REACH_HERE;

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
    qdcpu::SolverTraits<real>::rbmCalculate_hJc(h0, h1, J, c, b0, b1, W, W.dims[1], W.dims[0]);
}


extern "C"
PyObject *cpu_native_rbm_calculate_hJc(PyObject *module, PyObject *args) {
    PyObject *objH0 = NULL, *objH1 = NULL, *objJ = NULL, *objC = NULL,
        *objB0 = NULL, *objB1 = NULL, *objW = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOi", &objH0, &objH1, &objJ, &objC,
                          &objB0, &objB1, &objW, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_rbm_calculate_hJc<double>(objH0, objH1, objJ, objC,
                                           objB0, objB1, objW);
    else if (nPrec == 32)
        internal_rbm_calculate_hJc<float>(objH0, objH1, objJ, objC,
                                          objB0, objB1, objW);
    else
        DONT_REACH_HERE;

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
    qdcpu::SolverTraits<real>::rbmCalculate_E_fromQbits(E, h0, h1, J, c, q0, q1, q0.dims[0], q1.dims[0]);
}
    
extern "C"
PyObject *cpu_native_rbm_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE = NULL,
        *objH0 = NULL, *objH1 = NULL, *objJ = NULL, *objC = NULL,
        *objQ0 = NULL, *objQ1 = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOi",
                          &objE, &objH0, &objH1, &objJ, &objC,
                          &objQ0, &objQ1, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_rbm_calculate_E_from_qbits<double>(objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    else if (nPrec == 32)
        internal_rbm_calculate_E_from_qbits<float>(objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    else
        DONT_REACH_HERE;

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
    qdcpu::SolverTraits<real>::
        rbmBatchCalculate_E_fromQbits(E, h0, h1, J, c, q0, q1, N0, N1, nBatch0, nBatch1);
}
    
extern "C"
PyObject *cpu_native_rbm_batch_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE = NULL,
        *objH0 = NULL, *objH1 = NULL, *objJ = NULL, *objC = NULL,
        *objQ0 = NULL, *objQ1 = NULL;
    int nPrec = 0;
    if (!PyArg_ParseTuple(args, "OOOOOOOi",
                          &objE, &objH0, &objH1, &objJ, &objC,
                          &objQ0, &objQ1, &nPrec))
        return NULL;
    
    if (nPrec == 64)
        internal_rbm_batch_calculate_E_from_qbits<double>(objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    else if (nPrec == 32)
        internal_rbm_batch_calculate_E_from_qbits<float>(objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    else
        DONT_REACH_HERE;

    Py_INCREF(Py_None);
    return Py_None;    
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
