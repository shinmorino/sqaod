#include <pyglue.h>
#include <cpu/Traits.h>


// http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
// http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

/* NOTE: Value type checks for python objs have been already done in python glue, 
 * Here we only get entities needed. */


static PyObject *CpuNativeError;
namespace qdcpu = quantd_cpu;

#define DONT_REACH_HERE {printf("Don't reach here, %s:%d\n", __FILE__, (int)__LINE__); abort();}


namespace {

template<class real>
void internal_dense_graph_calculate_hJc(PyObject *objH, PyObject *objJ, PyObject *objC,
                                        PyObject *objW) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix h(objH), J(objJ), c(objC), W(objW);
    /* do the native job */
    qdcpu::SolverTraits<real>::denseGraphCalculate_hJc(h, J, c, W, W.dims[0]);
}


extern "C"
PyObject *cpunative_dense_graph_calculate_hJc(PyObject *module, PyObject *args) {
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

    
template<class real>
void internal_dense_graph_calculate_E(PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix E(objE), W(objW);
    NpBitMatrix x(objX);
    /* do the native job */
    qdcpu::SolverTraits<real>::denseGraphCalculate_E(E, W, x, x.dims[0]);
}

    
extern "C"
PyObject *cpunative_dense_graph_calculate_E(PyObject *module, PyObject *args) {
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
PyObject *cpunative_dense_graph_batch_calculate_E(PyObject *module, PyObject *args) {
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
PyObject *cpunative_dense_graph_calculate_E_from_qbits(PyObject *module, PyObject *args) {
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
PyObject *cpunative_dense_graph_batch_calculate_E_from_qbits(PyObject *module, PyObject *args) {
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

}



static
PyMethodDef annealermethods[] = {
	{"dense_graph_calculate_hJc", cpunative_dense_graph_calculate_hJc, METH_VARARGS},
	{"dense_graph_calculate_E", cpunative_dense_graph_calculate_E, METH_VARARGS},
	{"dense_graph_batch_calculate_E", cpunative_dense_graph_batch_calculate_E, METH_VARARGS},
	{"dense_graph_calculate_E_from_qbits", cpunative_dense_graph_calculate_E_from_qbits, METH_VARARGS},
	{"dense_graph_batch_calculate_E_from_qbits", cpunative_dense_graph_batch_calculate_E_from_qbits, METH_VARARGS},
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
    
    char name[] = "cpunative.error";
    CpuNativeError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(CpuNativeError);
    PyModule_AddObject(m, "error", CpuNativeError);
}
