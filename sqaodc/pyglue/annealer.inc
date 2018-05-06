/* -*- c++ -*- */


namespace {

template<class real>
Annealer<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<Annealer<real> *>(val);
}


extern "C"
PyObject *annealer_new(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    if (isFloat64(dtype))
        ext = (void*)newAnnealer<double>();
    else
        ext = (void*)newAnnealer<float>();
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    return obj;
}

extern "C"
PyObject *annealer_delete(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    if (isFloat64(dtype))
        deleteInstance(pyobjToCppObj<double>(objExt));
    else if (isFloat32(dtype))
        deleteInstance(pyobjToCppObj<float>(objExt));
    
    Py_INCREF(Py_None);
    return Py_None;    
}

#ifdef CUDA_SOLVER

namespace sqcu = sqaod::cuda;

extern "C"
PyObject *annealer_assign_device(PyObject *module, PyObject *args) {
    PyObject *objExt, *objDevice, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objDevice, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    sqcu::Device *device = (sqcu::Device*)PyArrayScalar_VAL(objDevice, UInt64);
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->assignDevice(*device);
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->assignDevice(*device);
    } CATCH_ERROR_AND_RETURN;
    
    Py_INCREF(Py_None);
    return Py_None;    
}

#endif


extern "C"
PyObject *annealer_seed(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    unsigned long long seed;
    if (!PyArg_ParseTuple(args, "OKO", &objExt, &seed, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->seed(seed);
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->seed(seed);
    } CATCH_ERROR_AND_RETURN;
    
    Py_INCREF(Py_None);
    return Py_None;    
}


#ifdef DENSE_GRAPH

template<class real> static
void internal_set_qubo(PyObject *objExt, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    NpMatrix W(objW);
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setQUBO(W, om);
}
    
extern "C"
PyObject *annealer_set_qubo(PyObject *module, PyObject *args) {
    PyObject *objExt, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOiO", &objExt, &objW, &opt, &dtype))

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            internal_set_qubo<double>(objExt, objW, opt);
        else if (isFloat32(dtype))
            internal_set_qubo<float>(objExt, objW, opt);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
PyObject *internal_set_hamiltonian(PyObject *objExt,
                                   PyObject *objH, PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpConstScalarType<real> NpConstScalar;

    NpVector h(objH);
    NpMatrix J(objJ);
    NpConstScalar c(objC);
    if (c.err)
        return NULL;
    pyobjToCppObj<real>(objExt)->setHamiltonian(h, J, c);
    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *annealer_set_hamiltonian(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objH, &objJ, &objC, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            return internal_set_hamiltonian<double>(objExt, objH, objJ, objC);
        else // if (isFloat32(dtype))
            return internal_set_hamiltonian<float>(objExt, objH, objJ, objC);
    } CATCH_ERROR_AND_RETURN;

    return NULL;
}


extern "C"
PyObject *annealer_get_problem_size(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    sqaod::SizeType N;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->getProblemSize(&N);
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->getProblemSize(&N);
    } CATCH_ERROR_AND_RETURN;

    return Py_BuildValue("I", N);
}

#endif

#ifdef BIPARTITE_GRAPH


template<class real>
void internal_set_qubo(PyObject *objExt,
                       PyObject *objB0, PyObject *objB1, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    const NpVector b0(objB0), b1(objB1);
    const NpMatrix W(objW);
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setQUBO(b0, b1, W, om);
}


extern "C"
PyObject *annealer_set_qubo(PyObject *module, PyObject *args) {
    PyObject *objExt, *objB0, *objB1, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOOOiO", &objExt, &objB0, &objB1, &objW, &opt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_set_qubo<double>(objExt, objB0, objB1, objW, opt);
        else // if (isFloat32(dtype))
            internal_set_qubo<float>(objExt, objB0, objB1, objW, opt);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real>
PyObject *internal_set_hamiltonian(PyObject *objExt, PyObject *objH0, PyObject *objH1,
                              PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpConstScalarType<real> NpConstScalar;

    NpVector h0(objH0);
    NpVector h1(objH1);
    NpMatrix J(objJ);
    NpConstScalar c(objC);
    if (c.err)
        return NULL;
    pyobjToCppObj<real>(objExt)->setHamiltonian(h0, h1, J, c);
    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *annealer_set_hamiltonian(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH0, *objH1, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objExt, &objH0, &objH1, &objJ, &objC, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            return internal_set_hamiltonian<double>(objExt, objH0, objH1, objJ, objC);
        else // if (isFloat32(dtype))
            return internal_set_hamiltonian<float>(objExt, objH0, objH1, objJ, objC);
    } CATCH_ERROR_AND_RETURN;
    return NULL;
}


extern "C"
PyObject *annealer_get_problem_size(PyObject *module, PyObject *args) {

    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    sqaod::SizeType N0, N1;
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->getProblemSize(&N0, &N1);
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->getProblemSize(&N0, &N1);
    } CATCH_ERROR_AND_RETURN;

    return Py_BuildValue("II", N0, N1);
}

#endif


extern "C"
PyObject *annealer_set_preferences(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype, *objPrefs;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objPrefs, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    sq::Preferences prefs;
    if (parsePreferences(objPrefs, &prefs) == -1)
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->setPreferences(prefs);
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->setPreferences(prefs);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *annealer_get_preferences(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    sq::Preferences prefs;
    TRY {
        if (isFloat64(dtype))
            prefs = pyobjToCppObj<double>(objExt)->getPreferences();
        else // if (isFloat32(dtype))
            prefs = pyobjToCppObj<float>(objExt)->getPreferences();
    } CATCH_ERROR_AND_RETURN;

    return createPreferences(prefs);    
}


template<class real>
PyObject *internal_get_E(PyObject *objExt, int typenum) {
    typedef NpVectorType<real> NpVector;
    const sqaod::VectorType<real> &E = pyobjToCppObj<real>(objExt)->get_E();
    NpVector npE(E.size, typenum); /* allocate PyObject */
    npE.vec = E;
    return npE.obj;
}

    
extern "C"
PyObject *annealer_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            return internal_get_E<double>(objExt, NPY_FLOAT64);
        else // if (isFloat32(dtype))
            return internal_get_E<float>(objExt, NPY_FLOAT32);
    } CATCH_ERROR_AND_RETURN;
}


#ifdef DENSE_GRAPH

template<class real>
PyObject *internal_get_x(PyObject *objExt) {
    Annealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N;
    ann->getProblemSize(&N);
    const sqaod::BitSetArray &xList = ann->get_x();
    PyObject *list = PyList_New(xList.size());
    for (sq::IdxType idx = 0; idx < xList.size(); ++idx) {
        NpBitVector x(N, NPY_INT8);
        x.vec = xList[idx];
        PyList_SET_ITEM(list, idx, x.obj);
    }
    return list;
}
    
extern "C"
PyObject *annealer_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            return internal_get_x<double>(objExt);
        else // if (isFloat32(dtype))
            return internal_get_x<float>(objExt);
    } CATCH_ERROR_AND_RETURN;
}


template<class real>
void internal_set_q(PyObject *objExt, PyObject *objQ) {
    if (PyArray_Check(objQ)) {
        NpBitVector q(objQ);
        pyobjToCppObj<real>(objExt)->set_q(q);
    }
    else {
        throwError("q is not an Array object.");
    }
}

extern "C"
PyObject *annealer_set_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQ, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objQ, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_set_q<double>(objExt, objQ);
        else // if (isFloat32(dtype))
            internal_set_q<float>(objExt, objQ);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
void internal_set_qset(PyObject *objExt, PyObject *objQ) {
    if (PyList_Check(objQ)) {
        PyListObject *list = (PyListObject*)objQ;
        int nItems = (int)PyList_GET_SIZE(list);
        sq::BitSetArray qList;
        for (int idx = 0; idx < nItems; ++idx) {
            PyObject *item = PyList_GET_ITEM(list, idx);
            if (!PyArray_Check(item))
                throwError("q is not an Array object.");
            NpBitVector q(item);
            qList.pushBack(q);
        }
        pyobjToCppObj<real>(objExt)->set_qset(qList);
    }
    else {
        throwError("q is not a List.");
    }
}

extern "C"
PyObject *annealer_set_qset(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQ, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objQ, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_set_qset<double>(objExt, objQ);
        else // if (isFloat32(dtype))
            internal_set_qset<float>(objExt, objQ);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_get_hamiltonian(PyObject *objExt,
                                          PyObject *objH, PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpScalarRefType<real> NpScalarRef;

    NpMatrix J(objJ);
    NpVector h(objH);
    NpScalarRef c(objC);
    
    Annealer<real> *ann = pyobjToCppObj<real>(objExt);
    ann->getHamiltonian(&h, &J, &c);
}
    
    
extern "C"
PyObject *annealer_get_hamiltonian(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objH, &objJ, &objC, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            internal_get_hamiltonian<double>(objExt, objH, objJ, objC);
        else // if (isFloat32(dtype))
            internal_get_hamiltonian<float>(objExt, objH, objJ, objC);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}

    
template<class real>
PyObject *internal_get_q(PyObject *objExt) {
    Annealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N;
    ann->getProblemSize(&N);
    const sqaod::BitSetArray &qList = ann->get_q();
    PyObject *list = PyList_New(qList.size());
    for (sq::IdxType idx = 0; idx < qList.size(); ++idx) {
        NpBitVector q(N, NPY_INT8);
        q.vec = qList[idx];
        PyList_SET_ITEM(list, idx, q.obj);
    }
    return list;
}
    
extern "C"
PyObject *annealer_get_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            return internal_get_q<double>(objExt);
        else // if (isFloat32(dtype))
            return internal_get_q<float>(objExt);
    } CATCH_ERROR_AND_RETURN;
}

#endif

#ifdef BIPARTITE_GRAPH

template<class real>
PyObject *internal_get_x(PyObject *objExt) {
    Annealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N0, N1;
    ann->getProblemSize(&N0, &N1);
    const sq::BitSetPairArray &xPairList = ann->get_x();

    PyObject *list = PyList_New(xPairList.size());
    for (sq::IdxType idx = 0; idx < xPairList.size(); ++idx) {
        const sq::BitSetPairArray::ValueType &pair = xPairList[idx];

        NpBitVector x0(N0, NPY_INT8), x1(N1, NPY_INT8);
        x0.vec = pair.bits0;
        x1.vec = pair.bits1;

        PyObject *tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(tuple, 0, x0.obj);
        PyTuple_SET_ITEM(tuple, 1, x1.obj);
        PyList_SET_ITEM(list, idx, tuple);
    }
    return list;
}


extern "C"
PyObject *annealer_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            return internal_get_x<double>(objExt);
        else // if (isFloat32(dtype))
            return internal_get_x<float>(objExt);
    } CATCH_ERROR_AND_RETURN;
}


template<class real>
void internal_set_q(PyObject *objExt, PyObject *objQ) {

    if (PyTuple_Check(objQ)) {
        if (PyTuple_GET_SIZE(objQ) != 2)
            throwError("Invalid size of tuple.");
        NpBitVector q0(PyTuple_GET_ITEM(objQ, 0));
        NpBitVector q1(PyTuple_GET_ITEM(objQ, 1));
        pyobjToCppObj<real>(objExt)->set_q(sq::BitSetPair(q0, q1));
    }
    else {
        throwError("q is not a tuple object.");
    }
}

extern "C"
PyObject *annealer_set_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQPair, *dtype;
    
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objQPair, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_set_q<double>(objExt, objQPair);
        else // if (isFloat32(dtype))
            internal_set_q<float>(objExt, objQPair);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real>
void internal_set_qset(PyObject *objExt, PyObject *objQ) {

    if (PyList_Check(objQ)) {
        PyListObject *list = (PyListObject*)objQ;
        int nItems = (int)PyList_GET_SIZE(list);
        /* FIXME: smarter memory management. */
        sq::BitSetPairArray qList;
        for (int idx = 0; idx < nItems; ++idx) {
            PyObject *item = PyList_GET_ITEM(list, idx);
            /* tuple or list */
            if (PyTuple_Check(item)) {
                if (PyTuple_GET_SIZE(item) != 2)
                    throwError("Invalid size of tuple.");
                NpBitVector q0(PyTuple_GET_ITEM(item, 0));
                NpBitVector q1(PyTuple_GET_ITEM(item, 1));
                qList.pushBack(sq::BitSetPair(q0, q1));
            }
            else {
                throwError("q is not a tuple.");
            }
        }
        pyobjToCppObj<real>(objExt)->set_qset(qList);
    }
    else {
        throwError("q is not a list object.");
    }
}

extern "C"
PyObject *annealer_set_qset(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQPair, *dtype;
    
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objQPair, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_set_qset<double>(objExt, objQPair);
        else // if (isFloat32(dtype))
            internal_set_qset<float>(objExt, objQPair);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
PyObject *internal_bg_annealer_get_q(PyObject *objExt) {
    Annealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N0, N1;
    ann->getProblemSize(&N0, &N1);
    const sq::BitSetPairArray &xPairList = ann->get_q();

    PyObject *list = PyList_New(xPairList.size());
    for (sq::IdxType idx = 0; idx < xPairList.size(); ++idx) {
        const sq::BitSetPairArray::ValueType &pair = xPairList[idx];

        NpBitVector q0(N0, NPY_INT8), q1(N1, NPY_INT8);
        q0.vec = pair.bits0;
        q1.vec = pair.bits1;

        PyObject *tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(tuple, 0, q0.obj);
        PyTuple_SET_ITEM(tuple, 1, q1.obj);
        PyList_SET_ITEM(list, idx, tuple);
    }
    return list;
}
    
extern "C"
PyObject *annealer_get_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_bg_annealer_get_q<double>(objExt);
        else // if (isFloat32(dtype))
            return internal_bg_annealer_get_q<float>(objExt);
    } CATCH_ERROR_AND_RETURN;
}
    
extern "C"
PyObject *bg_annealer_radomize_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->randomizeSpin();
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->randomizeSpin();
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_bg_annealer_get_hamiltonian(PyObject *objExt,
                                          PyObject *objH0, PyObject *objH1,
                                          PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpScalarRefType<real> NpScalarRef;
    
    Annealer<real> *ann = pyobjToCppObj<real>(objExt);
    NpVector h0(objH0), h1(objH1);
    NpScalarRef c(objC);
    NpMatrix J(objJ);
    ann->getHamiltonian(&h0, &h1, &J, &c);
}
    
    
extern "C"
PyObject *annealer_get_hamiltonian(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH0, *objH1, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objExt, &objH0, &objH1, &objJ, &objC, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            internal_bg_annealer_get_hamiltonian<double>(objExt, objH0, objH1, objJ, objC);
        else // if (isFloat32(dtype))
            internal_bg_annealer_get_hamiltonian<float>(objExt, objH0, objH1, objJ, objC);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}


#endif

extern "C"
PyObject *annealer_radomize_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->randomizeSpin();
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->randomizeSpin();
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *annealer_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->calculate_E();
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->calculate_E();
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}

        
extern "C"
PyObject *annealer_prepare(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->prepare();
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->prepare();
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *annealer_make_solution(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    
    ASSERT_DTYPE(dtype);
    
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->makeSolution();
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->makeSolution();
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
PyObject *internal_anneal_one_step(PyObject *objExt, PyObject *objG, PyObject *objBeta) {
    typedef NpConstScalarType<real> NpConstScalar;
    NpConstScalar G(objG), beta(objBeta);
    if (G.err || beta.err)
        return NULL;
    pyobjToCppObj<real>(objExt)->annealOneStep(G, beta);
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *annealer_anneal_one_step(PyObject *module, PyObject *args) {
    PyObject *objExt, *objG, *objKT, *dtype;
    if (!PyArg_ParseTuple(args, "OOOO", &objExt, &objG, &objKT, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            return internal_anneal_one_step<double>(objExt, objG, objKT);
        else // if (isFloat32(dtype))
            return internal_anneal_one_step<float>(objExt, objG, objKT);
    } CATCH_ERROR_AND_RETURN;
    return NULL;
}

}



static
PyMethodDef annealer_methods[] = {
	{"new", annealer_new, METH_VARARGS},
	{"delete", annealer_delete, METH_VARARGS},
#ifdef CUDA_SOLVER
        {"assign_device", annealer_assign_device, METH_VARARGS},
#endif
	{"seed", annealer_seed, METH_VARARGS},
	{"set_qubo", annealer_set_qubo, METH_VARARGS},
	{"set_hamiltonian", annealer_set_hamiltonian, METH_VARARGS},
	{"get_problem_size", annealer_get_problem_size, METH_VARARGS},
	{"set_preferences", annealer_set_preferences, METH_VARARGS},
	{"get_preferences", annealer_get_preferences, METH_VARARGS},
	{"get_E", annealer_get_E, METH_VARARGS},
	{"get_x", annealer_get_x, METH_VARARGS},
	{"get_hamiltonian", annealer_get_hamiltonian, METH_VARARGS},
	{"get_q", annealer_get_q, METH_VARARGS},
	{"set_q", annealer_set_q, METH_VARARGS},
	{"set_qset", annealer_set_qset, METH_VARARGS},
	{"randomize_spin", annealer_radomize_q, METH_VARARGS},
	{"calculate_E", annealer_calculate_E, METH_VARARGS},
	{"prepare", annealer_prepare, METH_VARARGS},
	{"make_solution", annealer_make_solution, METH_VARARGS},
	{"anneal_one_step", annealer_anneal_one_step, METH_VARARGS},
	{NULL},
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        modname,
        NULL, 0,
        annealer_methods,
        NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC INIT_MODULE(void) {
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;
    import_array();
    return module;
}

#else

PyMODINIT_FUNC INIT_MODULE(void) {
    PyObject *module = Py_InitModule(modname, annealer_methods);
    if (module == NULL)
        return;
    import_array();
}

#endif
