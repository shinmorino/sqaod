#include <pyglue.h>
#include <cuda/CUDAFormulas.h>


static PyObject *Cuda_FormulasError;
namespace sq = sqaod;
namespace sqcu = sqaod_cuda;


namespace {

sqcu::CUDADenseGraphFormulas<float>  dgFuncsFp64_;
sqcu::CUDADenseGraphFormulas<double> dgFuncsFp32_;

template<class real> sqcu::CUDADenseGraphFormulas<real> &dgFuncs();
template<> sqcu::CUDADenseGraphFormulas<float> &dgFuncs<float>() { return dgFuncsFp64_; }
template<> sqcu::CUDADenseGraphFormulas<double> &dgFuncs<double>() { return dgFuncsFp32_; }

sqcu::CUDABipartiteGraphFormulas<float>  bgFuncsFp64_;
sqcu::CUDABipartiteGraphFormulas<double> bgFuncsFp32_;

template<class real> sqcu::CUDABipartiteGraphFormulas<real> &bgFuncs();
template<> sqcu::CUDABipartiteGraphFormulas<float> &bgFuncs<float>() { return bgFuncsFp64_; }
template<> sqcu::CUDABipartiteGraphFormulas<double> &bgFuncs<double>() { return bgFuncsFp32_; }



template<class real>
void internal_dense_graph_calculate_E(PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;

    const NpMatrix W(objW);
    NpVector E(objE);
    NpVectorType<char> x(objX);
    /* do the native job */
    dgFuncs<real>().calculate_E(E.vec.data, W, sq::cast<real>(x.vec));
}

    
extern "C"
PyObject *cuda_formulas_dense_graph_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE, *objW, *objX;
    PyObject *dtype;
    
    if (!PyArg_ParseTuple(args, "OOOO", &objE, &objW, &objX, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_dense_graph_calculate_E<double>(objE, objW, objX);
        else if (isFloat32(dtype))
            internal_dense_graph_calculate_E<float>(objE, objW, objX);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dense_graph_batch_calculate_E(PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    
    const NpMatrix W(objW);
    NpVector E(objE); 
    const NpBitMatrix x(objX);
    /* do the native job */
    dgFuncs<real>().calculate_E(&E, W, sq::cast<real>(x.mat));
}

extern "C"
PyObject *cuda_formulas_dense_graph_batch_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE, *objW, *objX;
    PyObject *dtype = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &objE, &objW, &objX, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_dense_graph_batch_calculate_E<double>(objE, objW, objX);
        else if (isFloat32(dtype))
            internal_dense_graph_batch_calculate_E<float>(objE, objW, objX);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    


template<class real>
void internal_dense_graph_calculate_hJc(PyObject *objH, PyObject *objJ, PyObject *objC,
                                        PyObject *objW) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpVector h(objH), c(objC);
    NpMatrix J(objJ);
    const NpMatrix W(objW);
    /* do the native job */
    dgFuncs<real>().calculate_hJc(&h, &J, c.vec.data, W);
}


extern "C"
PyObject *cuda_formulas_dense_graph_calculate_hJc(PyObject *module, PyObject *args) {
    PyObject *objH, *objJ, *objC, *objW;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objH, &objJ, &objC, &objW, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_dense_graph_calculate_hJc<double>(objH, objJ, objC, objW);
        else if (isFloat32(dtype))
            internal_dense_graph_calculate_hJc<float>(objH, objJ, objC, objW);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
template<class real> void
internal_dense_graph_calculate_E_from_qbits(PyObject *objE,
                                            PyObject *objH, PyObject *objJ, PyObject *objC,
                                            PyObject *objQ) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpScalarRefType<real> NpScalarRef;
    typedef NpConstScalarType<real> NpConstScalar;
    NpScalarRef E(objE);
    const NpVector h(objH);
    const NpMatrix J(objJ);
    NpConstScalar c(objC);
    const NpBitVector q(objQ);
    /* do the native job */
    dgFuncs<real>().calculate_E(&E, h, J, c, sq::cast<real>(q.vec));
}
    

extern "C"
PyObject *cuda_formulas_dense_graph_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE, *objH, *objJ, *objC, *objQ;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objE, &objH, &objJ, &objC, &objQ, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_dense_graph_calculate_E_from_qbits<double>(objE, objH, objJ, objC, objQ);
        else if (isFloat32(dtype))
            internal_dense_graph_calculate_E_from_qbits<float>(objE, objH, objJ, objC, objQ);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dense_graph_batch_calculate_E_from_qbits(PyObject *objE,
                                                       PyObject *objH, PyObject *objJ, PyObject *objC,
                                                       PyObject *objQ) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpConstScalarType<real> NpConstScalar;
    NpVector E(objE);
    const NpVector h(objH);
    const NpMatrix J(objJ);
    NpConstScalar c(objC);
    const NpBitMatrix q(objQ);
    /* do the native job */
    dgFuncs<real>().calculate_E(&E, h, J, c, sq::cast<real>(q.mat));
}

extern "C"
PyObject *cuda_formulas_dense_graph_batch_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE, *objH, *objJ, *objC, *objQ;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objE, &objH, &objJ, &objC, &objQ, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_dense_graph_batch_calculate_E_from_qbits<double>(objE, objH, objJ, objC, objQ);
        else if (isFloat32(dtype))
            internal_dense_graph_batch_calculate_E_from_qbits<float>(objE, objH, objJ, objC, objQ);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    

/* Bipartite graph */

    

template<class real> void
internal_bipartite_graph_calculate_E(PyObject *objE,
                                     PyObject *objB0, PyObject *objB1, PyObject *objW,
                                     PyObject *objX0, PyObject *objX1) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpScalarRefType<real> NpScalarRef;
    const NpVector b0(objB0), b1(objB1);
    const NpMatrix W(objW);
    NpScalarRef E(objE);
    const NpBitVector x0(objX0), x1(objX1);
    /* do the native job */
    bgFuncs<real>().calculate_E(&E, b0, b1, W,
                                sq::cast<real>(x0.vec), sq::cast<real>(x1.vec));
}
    
extern "C"
PyObject *cuda_formulas_bipartite_graph_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE, *objB0, *objB1, *objW, *objX0, *objX1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_calculate_E<double>(objE, objB0, objB1, objW, objX0, objX1);
        else if (isFloat32(dtype))
            internal_bipartite_graph_calculate_E<float>(objE, objB0, objB1, objW, objX0, objX1);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real> void
internal_bipartite_graph_batch_calculate_E(PyObject *objE,
                                           PyObject *objB0, PyObject *objB1, PyObject *objW,
                                           PyObject *objX0, PyObject *objX1) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpVector E(objE);
    const NpVector b0(objB0), b1(objB1);
    const NpMatrix W(objW);
    const NpBitMatrix x0(objX0), x1(objX1);
    bgFuncs<real>().calculate_E(&E, b0, b1, W,
                                sq::cast<real>(x0.mat), sq::cast<real>(x1.mat));
}
    
extern "C"
PyObject *cuda_formulas_bipartite_graph_batch_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objE, *objB0, *objB1, *objW, *objX0, *objX1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_batch_calculate_E<double>
                    (objE, objB0, objB1, objW, objX0, objX1);
        else if (isFloat32(dtype))
            internal_bipartite_graph_batch_calculate_E<float>
                    (objE, objB0, objB1, objW, objX0, objX1);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real> void
internal_bipartite_graph_batch_calculate_E_2d(PyObject *objE,
                                              PyObject *objB0, PyObject *objB1, PyObject *objW,
                                              PyObject *objX0, PyObject *objX1) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpMatrix E(objE);
    const NpVector b0(objB0), b1(objB1);
    const NpMatrix W(objW);
    const NpBitMatrix x0(objX0), x1(objX1);
    bgFuncs<real>().calculate_E_2d(&E, b0, b1, W,
                                   sq::cast<real>(x0.mat), sq::cast<real>(x1.mat));
}
    
extern "C"
PyObject *cuda_formulas_bipartite_graph_batch_calculate_E_2d(PyObject *module, PyObject *args) {
    PyObject *objE, *objB0, *objB1, *objW, *objX0, *objX1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_batch_calculate_E_2d<double>
                    (objE, objB0, objB1, objW, objX0, objX1);
        else if (isFloat32(dtype))
            internal_bipartite_graph_batch_calculate_E_2d<float>
                    (objE, objB0, objB1, objW, objX0, objX1);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);
        
    Py_INCREF(Py_None);
    return Py_None;    
}
    
template<class real>
void internal_bipartite_graph_calculate_hJc(PyObject *objH0, PyObject *objH1, PyObject *objJ,
                                            PyObject *objC,
                                            PyObject *objB0, PyObject *objB1, PyObject *objW) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpScalarRefType<real> NpScalarRef;
    const NpVector b0(objB0), b1(objB1);
    const NpMatrix W(objW);
    NpVector h0(objH0), h1(objH1);
    NpScalarRef c(objC);
    NpMatrix J(objJ);
    /* do the native job */
    bgFuncs<real>().calculate_hJc(&h0, &h1, &J, &c, b0, b1, W);
}


extern "C"
PyObject *cuda_formulas_bipartite_graph_calculate_hJc(PyObject *module, PyObject *args) {
    PyObject *objH0, *objH1, *objJ, *objC, *objB0, *objB1, *objW;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO", &objH0, &objH1, &objJ, &objC,
                          &objB0, &objB1, &objW, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_calculate_hJc<double>(objH0, objH1, objJ, objC,
                                                           objB0, objB1, objW);
        else if (isFloat32(dtype))
            internal_bipartite_graph_calculate_hJc<float>(objH0, objH1, objJ, objC,
                                                          objB0, objB1, objW);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real> void
internal_bipartite_graph_calculate_E_from_qbits(PyObject *objE,
                                                PyObject *objH0, PyObject *objH1, PyObject *objJ, PyObject *objC,
                                                PyObject *objQ0, PyObject *objQ1) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpConstScalarType<real> NpConstScalar;
    NpVector E(objE);
    const NpVector h0(objH0), h1(objH1);
    const NpMatrix J(objJ);
    NpConstScalar c(objC);
    const NpBitVector q0(objQ0), q1(objQ1);
    /* do the native job */
    bgFuncs<real>().calculate_E(E.vec.data, h0, h1, J, c,
                                sq::cast<real>(q0.vec), sq::cast<real>(q1.vec));
}
    
extern "C"
PyObject *cuda_formulas_bipartite_graph_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE, *objH0, *objH1, *objJ, *objC, *objQ0, *objQ1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &objE, &objH0, &objH1, &objJ, &objC,
                          &objQ0, &objQ1, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_calculate_E_from_qbits<double>
                    (objE, objH0, objH1, objJ, objC, objQ0, objQ1);
        else if (isFloat32(dtype))
            internal_bipartite_graph_calculate_E_from_qbits<float>
                    (objE, objH0, objH1, objJ, objC, objQ0, objQ1);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real> void
internal_bipartite_graph_batch_calculate_E_from_qbits(PyObject *objE,
                                                      PyObject *objH0, PyObject *objH1, PyObject *objJ, PyObject *objC,
                                                      PyObject *objQ0, PyObject *objQ1) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpConstScalarType<real> NpConstScalar;
    NpVector E(objE);
    const NpVector h0(objH0), h1(objH1);
    const NpMatrix J(objJ);
    NpConstScalar c(objC);
    const NpBitMatrix q0(objQ0), q1(objQ1);
    /* do the native job */
    bgFuncs<real>().calculate_E(&E, h0, h1, J, c,
                                sq::cast<real>(q0.mat), sq::cast<real>(q1.mat));
}

extern "C"
PyObject *cuda_formulas_bipartite_graph_batch_calculate_E_from_qbits(PyObject *module, PyObject *args) {
    PyObject *objE, *objH0, *objH1, *objJ, *objC, *objQ0, *objQ1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &objE, &objH0, &objH1, &objJ, &objC,
                          &objQ0, &objQ1, &dtype))
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_batch_calculate_E_from_qbits<double>
                    (objE, objH0, objH1, objJ, objC, objQ0, objQ1);
        else if (isFloat32(dtype))
            internal_bipartite_graph_batch_calculate_E_from_qbits<float>
                    (objE, objH0, objH1, objJ, objC, objQ0, objQ1);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_FormulasError);
    } CATCH_ERROR_AND_RETURN(Cuda_FormulasError);
        
    Py_INCREF(Py_None);
    return Py_None;    
}

}



static
PyMethodDef annealermethods[] = {
	{"dense_graph_calculate_E", cuda_formulas_dense_graph_calculate_E, METH_VARARGS},
	{"dense_graph_batch_calculate_E", cuda_formulas_dense_graph_batch_calculate_E, METH_VARARGS},
	{"dense_graph_calculate_hJc", cuda_formulas_dense_graph_calculate_hJc, METH_VARARGS},
	{"dense_graph_calculate_E_from_qbits", cuda_formulas_dense_graph_calculate_E_from_qbits, METH_VARARGS},
	{"dense_graph_batch_calculate_E_from_qbits", cuda_formulas_dense_graph_batch_calculate_E_from_qbits, METH_VARARGS},
	{"bipartite_graph_calculate_E", cuda_formulas_bipartite_graph_calculate_E, METH_VARARGS},
	{"bipartite_graph_batch_calculate_E", cuda_formulas_bipartite_graph_batch_calculate_E, METH_VARARGS},
	{"bipartite_graph_batch_calculate_E_2d", cuda_formulas_bipartite_graph_batch_calculate_E_2d, METH_VARARGS},
	{"bipartite_graph_calculate_hJc", cuda_formulas_bipartite_graph_calculate_hJc, METH_VARARGS},
	{"bipartite_graph_calculate_E_from_qbits", cuda_formulas_bipartite_graph_calculate_E_from_qbits, METH_VARARGS},
	{"bipartite_graph_batch_calculate_E_from_qbits", cuda_formulas_bipartite_graph_batch_calculate_E_from_qbits, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcuda_formulas(void) {
    PyObject *m;
    
    m = Py_InitModule("cuda_formulas", annealermethods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cuda_formulas.error";
    Cuda_FormulasError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cuda_FormulasError);
    PyModule_AddObject(m, "error", Cuda_FormulasError);
}
