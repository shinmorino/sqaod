/* -*- c++ -*- */


namespace {

template<class real>
BFSearcher<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<BFSearcher<real>*>(val);
}

extern "C"
PyObject *bf_searcher_new(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    if (isFloat64(dtype))
        ext = (void*)new BFSearcher<double>();
    else // if (isFloat32(dtype))
        ext = (void*)new BFSearcher<float>();
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    return obj;
}

extern "C"
PyObject *bf_searcher_delete(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    if (isFloat64(dtype))
        delete pyobjToCppObj<double>(objExt);
    else // if (isFloat32(dtype))
        delete pyobjToCppObj<float>(objExt);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

#ifdef CUDA_SOLVER

namespace sqcu = sqaod::cuda;

extern "C"
PyObject *bf_searcher_assign_device(PyObject *module, PyObject *args) {
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


#ifdef DENSE_GRAPH

template<class real>
void internal_set_qubo(PyObject *objExt, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    const NpMatrix W(objW);
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setQUBO(W, om);
}
    
extern "C"
PyObject *bf_searcher_set_qubo(PyObject *module, PyObject *args) {
    PyObject *objExt, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOiO", &objExt, &objW, &opt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            internal_set_qubo<double>(objExt, objW, opt);
        else // if (isFloat32(dtype))
            internal_set_qubo<float>(objExt, objW, opt);
    } CATCH_ERROR_AND_RETURN;
    
    Py_INCREF(Py_None);
    return Py_None;    
}

#endif

#ifdef BIPARTITE_GRAPH


template<class real>
void internal_set_qubo(PyObject *objExt,
                       PyObject *objB0, PyObject *objB1, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpVector b0(objB0), b1(objB1);
    NpMatrix W(objW);
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setQUBO(b0, b1, W, om);
}
    
extern "C"
PyObject *bf_searcher_set_qubo(PyObject *module, PyObject *args) {
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

#endif


extern "C"
PyObject *bf_searcher_set_preferences(PyObject *module, PyObject *args) {
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
PyObject *bf_searcher_get_preferences(PyObject *module, PyObject *args) {
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


#ifdef DENSE_GRAPH

template<class real>
PyObject *internal_get_x(PyObject *objExt) {
    sqaod::SizeType N;
    BFSearcher<real> *sol = pyobjToCppObj<real>(objExt);
    const sq::BitSetArray &xList = sol->get_x();
    sol->getProblemSize(&N);

    PyObject *list = PyList_New(xList.size());
    for (sq::IdxType idx = 0; idx < xList.size(); ++idx) {
        const sq::BitSet &bits = xList[idx];
        NpBitVector x(N, NPY_INT8);
        x.vec = bits;
        PyList_SET_ITEM(list, idx, x.obj);
    }
    return list;
}
    
    
extern "C"
PyObject *bf_searcher_get_x(PyObject *module, PyObject *args) {
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

#endif

#ifdef BIPARTITE_GRAPH

template<class real>
PyObject *internal_get_x(PyObject *objExt) {
    BFSearcher<real> *sol = pyobjToCppObj<real>(objExt);
    const sq::BitSetPairArray &xList = sol->get_x();

    sqaod::SizeType N0, N1;
    sol->getProblemSize(&N0, &N1);
    
    PyObject *list = PyList_New(xList.size());
    for (sq::IdxType idx = 0; idx < xList.size(); ++idx) {
        const sq::BitSetPairArray::ValueType &pair = xList[idx];
        NpBitVector x0(N0, NPY_INT8), x1(N1, NPY_INT8);
        x0.vec = pair.first;
        x1.vec = pair.second;

        PyObject *tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(tuple, 0, x0.obj);
        PyTuple_SET_ITEM(tuple, 1, x1.obj);
        PyList_SET_ITEM(list, idx, tuple);
    }
    return list;
}
    
    
extern "C"
PyObject *bf_searcher_get_x(PyObject *module, PyObject *args) {
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

#endif


template<class real>
PyObject *internal_get_E(PyObject *objExt, int typenum) {
    typedef NpVectorType<real> NpVector;
    const sqaod::VectorType<real> &E = pyobjToCppObj<real>(objExt)->get_E();
    NpVector npE(E.size, typenum); /* allocate PyObject */
    npE.vec = E;
    return npE.obj;
}
    
extern "C"
PyObject *bf_searcher_get_E(PyObject *module, PyObject *args) {
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


extern "C"
PyObject *bf_searcher_prepare(PyObject *module, PyObject *args) {
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
PyObject *bf_searcher_calculate_E(PyObject *module, PyObject *args) {
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
PyObject *bf_searcher_make_solution(PyObject *module, PyObject *args) {
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

#ifdef DENSE_GRAPH

extern "C"
PyObject *bf_searcher_search_range(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    sq::PackedBitSet curX;
    bool res;
    TRY {
        if (isFloat64(dtype))
            res = pyobjToCppObj<double>(objExt)->searchRange(&curX);
        else // if (isFloat32(dtype))
            res = pyobjToCppObj<float>(objExt)->searchRange(&curX);
    } CATCH_ERROR_AND_RETURN;

    return Py_BuildValue("OK", res ? Py_True : Py_False, curX);
}

#endif

#ifdef BIPARTITE_GRAPH
    
extern "C"
PyObject *bf_searcher_search_range(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);
    
    sq::PackedBitSet curX0, curX1;
    bool res;
    TRY {
        if (isFloat64(dtype))
            res = pyobjToCppObj<double>(objExt)->searchRange(&curX0, &curX1);
        else // if (isFloat32(dtype))
            res = pyobjToCppObj<float>(objExt)->searchRange(&curX0, &curX1);
    } CATCH_ERROR_AND_RETURN;

    return Py_BuildValue("OKK", res ? Py_True : Py_False, curX0, curX1);
}

#endif


extern "C"
PyObject *bf_searcher_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    ASSERT_DTYPE(dtype);

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->search();
        else // if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->search();
    } CATCH_ERROR_AND_RETURN;
    
    Py_INCREF(Py_None);
    return Py_None;    
}

}



static
PyMethodDef bf_searcher_methods[] = {
	{"new", bf_searcher_new, METH_VARARGS},
	{"delete", bf_searcher_delete, METH_VARARGS},
#ifdef CUDA_SOLVER
        {"assign_device", bf_searcher_assign_device, METH_VARARGS},
#endif
	{"set_qubo", bf_searcher_set_qubo, METH_VARARGS},
	{"set_preferences", bf_searcher_set_preferences, METH_VARARGS},
	{"get_preferences", bf_searcher_get_preferences, METH_VARARGS},
	{"get_x", bf_searcher_get_x, METH_VARARGS},
	{"get_E", bf_searcher_get_E, METH_VARARGS},
	{"prepare", bf_searcher_prepare, METH_VARARGS},
	{"calculate_E", bf_searcher_calculate_E, METH_VARARGS},
	{"make_solution", bf_searcher_make_solution, METH_VARARGS},
	{"search_range", bf_searcher_search_range, METH_VARARGS},
	{"search", bf_searcher_search, METH_VARARGS},
	{NULL},
};


#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        modname,
        NULL, 0,
        bf_searcher_methods,
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
    PyObject *module = Py_InitModule(modname, bf_searcher_methods);
    if (module == NULL)
        return;
    import_array();
}

#endif
