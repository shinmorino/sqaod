/* -*- c++ -*- */

namespace {

template<template <class> class Formulas>
struct FormulasPtrPair {
    Formulas<float> *fltFormulas;
    Formulas<double> *dblFormulas;
};


template<class real>
DGFormulas<real> *pyObjToDGFormulas(PyObject *obj);

template<>
DGFormulas<float> *pyObjToDGFormulas(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<FormulasPtrPair<DGFormulas>*>(val)->fltFormulas;
}

template<>
DGFormulas<double> *pyObjToDGFormulas(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<FormulasPtrPair<DGFormulas>*>(val)->dblFormulas;
}

extern "C"
PyObject *dg_formulas_new(PyObject *module, PyObject *args) {
    FormulasPtrPair<DGFormulas> *pair = new FormulasPtrPair<DGFormulas>();
    pair->fltFormulas = newDGFormulas<float>();
    pair->dblFormulas = newDGFormulas<double>();
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)pair);
    return obj;
}

extern "C"
PyObject *dg_formulas_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    
    npy_uint64 val = PyArrayScalar_VAL(objExt, UInt64);
    FormulasPtrPair<DGFormulas> *pair = reinterpret_cast<FormulasPtrPair<DGFormulas>*>(val);
    delete(pair->fltFormulas);
    delete(pair->dblFormulas);
    delete pair;

    Py_INCREF(Py_None);
    return Py_None;
}

#ifdef CUDA_FORMULAS

namespace sqcu = sqaod::cuda;

extern "C"
PyObject *dg_formulas_assign_device(PyObject *module, PyObject *args) {
    PyObject *objExt, *objDevice;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &objDevice))
        return NULL;
    
    sqcu::Device *device = (sqcu::Device*)PyArrayScalar_VAL(objDevice, UInt64);
    TRY {
        pyObjToDGFormulas<float>(objExt)->assignDevice(*device);
        pyObjToDGFormulas<double>(objExt)->assignDevice(*device);
    } CATCH_ERROR_AND_RETURN;
    
    Py_INCREF(Py_None);
    return Py_None;    
}

#endif
    
template<class real>
void internal_dense_graph_calculate_E(PyObject *objExt, PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;

    const NpMatrix W(objW);
    NpVector E(objE);
    NpVectorType<char> x(objX);
    /* do the native job */
    pyObjToDGFormulas<real>(objExt)->calculate_E(E.vec.data, W, sq::cast<real>(x.vec));
}

    
extern "C"
PyObject *dense_graph_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objW, *objX;
    PyObject *dtype;
    
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objE, &objW, &objX, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_dense_graph_calculate_E<double>(objExt, objE, objW, objX);
        else // if (isFloat32(dtype))
            internal_dense_graph_calculate_E<float>(objExt, objE, objW, objX);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dense_graph_batch_calculate_E(PyObject *objExt, PyObject *objE, PyObject *objW, PyObject *objX) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    
    const NpMatrix W(objW);
    NpVector E(objE); 
    const NpBitMatrix x(objX);
    /* do the native job */
    pyObjToDGFormulas<real>(objExt)->calculate_E(&E, W, sq::cast<real>(x.mat));
}

extern "C"
PyObject *dense_graph_batch_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objW, *objX;
    PyObject *dtype = NULL;
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objE, &objW, &objX, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_dense_graph_batch_calculate_E<double>(objExt, objE, objW, objX);
        else // if (isFloat32(dtype))
            internal_dense_graph_batch_calculate_E<float>(objExt, objE, objW, objX);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}
    


template<class real>
void internal_dense_graph_calculate_hamiltonian(PyObject *objExt,
                                                PyObject *objH, PyObject *objJ, PyObject *objC,
                                                PyObject *objW) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpVector h(objH), c(objC);
    NpMatrix J(objJ);
    const NpMatrix W(objW);
    /* do the native job */
    pyObjToDGFormulas<real>(objExt)->calculateHamiltonian(&h, &J, c.vec.data, W);
}


extern "C"
PyObject *dense_graph_calculate_hamiltonian(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH, *objJ, *objC, *objW;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objExt, &objH, &objJ, &objC, &objW, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_dense_graph_calculate_hamiltonian<double>(objExt, objH, objJ, objC, objW);
        else // if (isFloat32(dtype))
            internal_dense_graph_calculate_hamiltonian<float>(objExt, objH, objJ, objC, objW);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}

    
template<class real> PyObject *
internal_dense_graph_calculate_E_from_spin(PyObject *objExt, PyObject *objE,
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

    if (c.err)
        return NULL;
    /* do the native job */
    pyObjToDGFormulas<real>(objExt)->calculate_E(&E, h, J, c, sq::cast<real>(q.vec));
    Py_INCREF(Py_None);
    return Py_None;    
}
    

extern "C"
PyObject *dense_graph_calculate_E_from_spin(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objH, *objJ, *objC, *objQ;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &objExt, &objE, &objH, &objJ, &objC, &objQ, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            return internal_dense_graph_calculate_E_from_spin<double>(objExt, objE, objH, objJ, objC, objQ);
        else // if (isFloat32(dtype))
            return internal_dense_graph_calculate_E_from_spin<float>(objExt, objE, objH, objJ, objC, objQ);
    } CATCH_ERROR_AND_RETURN;
    return NULL;
}


template<class real>
PyObject *internal_dense_graph_batch_calculate_E_from_spin(PyObject *objExt, PyObject *objE,
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
    if (c.err)
        return NULL;
    /* do the native job */
    pyObjToDGFormulas<real>(objExt)->calculate_E(&E, h, J, c, sq::cast<real>(q.mat));
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dense_graph_batch_calculate_E_from_spin(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objH, *objJ, *objC, *objQ;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOO", &objExt, &objE, &objH, &objJ, &objC, &objQ, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            return internal_dense_graph_batch_calculate_E_from_spin<double>(objExt, objE, objH, objJ, objC, objQ);
        else // if (isFloat32(dtype))
            return internal_dense_graph_batch_calculate_E_from_spin<float>(objExt, objE, objH, objJ, objC, objQ);
    } CATCH_ERROR_AND_RETURN;
    return NULL;
}
    

/* Bipartite graph */


template<class real>
BGFormulas<real> *pyObjToBGFormulas(PyObject *obj);

template<>
BGFormulas<float> *pyObjToBGFormulas(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<FormulasPtrPair<BGFormulas>*>(val)->fltFormulas;
}

template<>
BGFormulas<double> *pyObjToBGFormulas(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<FormulasPtrPair<BGFormulas>*>(val)->dblFormulas;
}


extern "C"
PyObject *bg_formulas_new(PyObject *module, PyObject *args) {
    FormulasPtrPair<BGFormulas> *pair = new FormulasPtrPair<BGFormulas>();
    pair->fltFormulas = newBGFormulas<float>();
    pair->dblFormulas = newBGFormulas<double>();
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)pair);
    return obj;
}

extern "C"
PyObject *bg_formulas_delete(PyObject *module, PyObject *args) {
    PyObject *objExt, *objDevice;
    if (!PyArg_ParseTuple(args, "O", &objExt, &objDevice))
        return NULL;
    
    npy_uint64 val = PyArrayScalar_VAL(objExt, UInt64);
    FormulasPtrPair<BGFormulas> *pair = reinterpret_cast<FormulasPtrPair<BGFormulas>*>(val);
    delete pair->fltFormulas;
    delete pair->dblFormulas;
    delete pair;

    Py_INCREF(Py_None);
    return Py_None;
}


#ifdef CUDA_FORMULAS

namespace sqcu = sqaod::cuda;

extern "C"
PyObject *bg_formulas_assign_device(PyObject *module, PyObject *args) {
    PyObject *objExt, *objDevice;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &objDevice))
        return NULL;
    
    sqcu::Device *device = (sqcu::Device*)PyArrayScalar_VAL(objDevice, UInt64);
    TRY {
        pyObjToBGFormulas<float>(objExt)->assignDevice(*device);
        pyObjToBGFormulas<double>(objExt)->assignDevice(*device);
    } CATCH_ERROR_AND_RETURN;
    
    Py_INCREF(Py_None);
    return Py_None;    
}

#endif

template<class real> void
internal_bipartite_graph_calculate_E(PyObject *objExt, PyObject *objE,
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
    pyObjToBGFormulas<real>(objExt)->calculate_E(&E, b0, b1, W,
                                                 sq::cast<real>(x0.vec), sq::cast<real>(x1.vec));
}
    
extern "C"
PyObject *bipartite_graph_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objB0, *objB1, *objW, *objX0, *objX1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &objExt, &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_calculate_E<double>(objExt, objE, objB0, objB1, objW, objX0, objX1);
        else if (isFloat32(dtype))
            internal_bipartite_graph_calculate_E<float>(objExt, objE, objB0, objB1, objW, objX0, objX1);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real> void
internal_bipartite_graph_batch_calculate_E(PyObject *objExt, PyObject *objE,
                                           PyObject *objB0, PyObject *objB1, PyObject *objW,
                                           PyObject *objX0, PyObject *objX1) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpVector E(objE);
    const NpVector b0(objB0), b1(objB1);
    const NpMatrix W(objW);
    const NpBitMatrix x0(objX0), x1(objX1);
    pyObjToBGFormulas<real>(objExt)->calculate_E(&E, b0, b1, W,
                                                 sq::cast<real>(x0.mat), sq::cast<real>(x1.mat));
}
    
extern "C"
PyObject *bipartite_graph_batch_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objB0, *objB1, *objW, *objX0, *objX1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &objExt, &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_batch_calculate_E<double>
                    (objExt, objE, objB0, objB1, objW, objX0, objX1);
        else // if (isFloat32(dtype))
            internal_bipartite_graph_batch_calculate_E<float>
                    (objExt, objE, objB0, objB1, objW, objX0, objX1);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real> void
internal_bipartite_graph_batch_calculate_E_2d(PyObject *objExt, PyObject *objE,
                                              PyObject *objB0, PyObject *objB1, PyObject *objW,
                                              PyObject *objX0, PyObject *objX1) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpMatrix E(objE);
    const NpVector b0(objB0), b1(objB1);
    const NpMatrix W(objW);
    const NpBitMatrix x0(objX0), x1(objX1);
    pyObjToBGFormulas<real>(objExt)->calculate_E_2d(&E, b0, b1, W,
                                       sq::cast<real>(x0.mat), sq::cast<real>(x1.mat));
}
    
extern "C"
PyObject *bipartite_graph_batch_calculate_E_2d(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objB0, *objB1, *objW, *objX0, *objX1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOO",
                          &objExt, &objE, &objB0, &objB1, &objW,
                          &objX0, &objX1, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_batch_calculate_E_2d<double>
                    (objExt, objE, objB0, objB1, objW, objX0, objX1);
        else // if (isFloat32(dtype))
            internal_bipartite_graph_batch_calculate_E_2d<float>
                    (objExt, objE, objB0, objB1, objW, objX0, objX1);
    } CATCH_ERROR_AND_RETURN;
        
    Py_INCREF(Py_None);
    return Py_None;    
}
    
template<class real>
void internal_bipartite_graph_calculate_hamiltonian(PyObject *objExt,
                                                    PyObject *objH0, PyObject *objH1, PyObject *objJ,
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
    pyObjToBGFormulas<real>(objExt)->calculateHamiltonian(&h0, &h1, &J, &c, b0, b1, W);
}


extern "C"
PyObject *bipartite_graph_calculate_hamiltonian(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH0, *objH1, *objJ, *objC, *objB0, *objB1, *objW;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOOO", &objExt, &objH0, &objH1, &objJ, &objC,
                          &objB0, &objB1, &objW, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_bipartite_graph_calculate_hamiltonian<double>(objExt,
                                                                   objH0, objH1, objJ, objC,
                                                                   objB0, objB1, objW);
        else // if (isFloat32(dtype))
            internal_bipartite_graph_calculate_hamiltonian<float>(objExt,
                                                                  objH0, objH1, objJ, objC,
                                                                  objB0, objB1, objW);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real> PyObject *
internal_bipartite_graph_calculate_E_from_spin(PyObject *objExt, PyObject *objE,
                                               PyObject *objH0, PyObject *objH1,
                                               PyObject *objJ, PyObject *objC,
                                               PyObject *objQ0, PyObject *objQ1) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpConstScalarType<real> NpConstScalar;
    NpVector E(objE);
    const NpVector h0(objH0), h1(objH1);
    const NpMatrix J(objJ);
    NpConstScalar c(objC);
    const NpBitVector q0(objQ0), q1(objQ1);
    if (c.err)
        return NULL;
    /* do the native job */
    pyObjToBGFormulas<real>(objExt)->calculate_E(E.vec.data, h0, h1, J, c,
                                  sq::cast<real>(q0.vec), sq::cast<real>(q1.vec));
    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *bipartite_graph_calculate_E_from_spin(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objH0, *objH1, *objJ, *objC, *objQ0, *objQ1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOOO",
                          &objExt, &objE, &objH0, &objH1, &objJ, &objC,
                          &objQ0, &objQ1, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            return internal_bipartite_graph_calculate_E_from_spin<double>
                    (objExt, objE, objH0, objH1, objJ, objC, objQ0, objQ1);
        else // if (isFloat32(dtype))
            return internal_bipartite_graph_calculate_E_from_spin<float>
                    (objExt, objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    } CATCH_ERROR_AND_RETURN;
    return NULL;
}


template<class real> PyObject *
internal_bipartite_graph_batch_calculate_E_from_spin(PyObject *objExt, PyObject *objE,
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

    if (c.err)
        return NULL;
    
    /* do the native job */
    pyObjToBGFormulas<real>(objExt)->calculate_E(&E, h0, h1, J, c,
                                  sq::cast<real>(q0.mat), sq::cast<real>(q1.mat));
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *bipartite_graph_batch_calculate_E_from_spin(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *objH0, *objH1, *objJ, *objC, *objQ0, *objQ1;
    PyObject *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOOOOO",
                          &objExt, &objE, &objH0, &objH1, &objJ, &objC,
                          &objQ0, &objQ1, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            return internal_bipartite_graph_batch_calculate_E_from_spin<double>
                    (objExt, objE, objH0, objH1, objJ, objC, objQ0, objQ1);
        else // if (isFloat32(dtype))
            return internal_bipartite_graph_batch_calculate_E_from_spin<float>
                    (objExt, objE, objH0, objH1, objJ, objC, objQ0, objQ1);
    } CATCH_ERROR_AND_RETURN;
    return NULL;
}
    
}



static
PyMethodDef formulas_methods[] = {
    {"dg_formulas_new", dg_formulas_new, METH_VARARGS},
    {"dg_formulas_delete", dg_formulas_delete, METH_VARARGS},
#ifdef CUDA_FORMULAS
    {"dg_formulas_assign_device", dg_formulas_assign_device, METH_VARARGS},
#endif
    {"dense_graph_calculate_E", dense_graph_calculate_E, METH_VARARGS},
    {"dense_graph_batch_calculate_E", dense_graph_batch_calculate_E, METH_VARARGS},
    {"dense_graph_calculate_hamiltonian", dense_graph_calculate_hamiltonian, METH_VARARGS},
    {"dense_graph_calculate_E_from_spin", dense_graph_calculate_E_from_spin, METH_VARARGS},
    {"dense_graph_batch_calculate_E_from_spin", dense_graph_batch_calculate_E_from_spin, METH_VARARGS},
    {"bg_formulas_new", bg_formulas_new, METH_VARARGS},
    {"bg_formulas_delete", bg_formulas_delete, METH_VARARGS},
#ifdef CUDA_FORMULAS
    {"bg_formulas_assign_device", bg_formulas_assign_device, METH_VARARGS},
#endif
    {"bipartite_graph_calculate_E", bipartite_graph_calculate_E, METH_VARARGS},
    {"bipartite_graph_batch_calculate_E", bipartite_graph_batch_calculate_E, METH_VARARGS},
    {"bipartite_graph_batch_calculate_E_2d", bipartite_graph_batch_calculate_E_2d, METH_VARARGS},
    {"bipartite_graph_calculate_hamiltonian", bipartite_graph_calculate_hamiltonian, METH_VARARGS},
    {"bipartite_graph_calculate_E_from_spin", bipartite_graph_calculate_E_from_spin, METH_VARARGS},
    {"bipartite_graph_batch_calculate_E_from_spin", bipartite_graph_batch_calculate_E_from_spin, METH_VARARGS},
    {NULL},
};



#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        modname,
        NULL, 0,
        formulas_methods,
        NULL, NULL, NULL, NULL
};

extern "C"
PyMODINIT_FUNC INIT_MODULE(void) {
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;
    import_array();
    return module;
}

#else

PyMODINIT_FUNC INIT_MODULE(void) {
    PyObject *module = Py_InitModule(modname, formulas_methods);
    if (module == NULL)
        return;
    import_array();
}

#endif
