#include <sqaodc/pyglue/pyglue.h>
#include <sqaodc/sqaodc.h>
#include <string.h>


static PyObject *Cuda_BgBfSearcherError;
namespace sq = sqaod;
namespace sqcu = sqaod_cuda;

template<class real>
using BipartiteGraphBFSearcher = sq::cuda::BipartiteGraphBFSearcher<real>;



namespace {
    
template<class real>
BipartiteGraphBFSearcher<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<BipartiteGraphBFSearcher<real>*>(val);
}

extern "C"
PyObject *bg_bf_searcher_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new BipartiteGraphBFSearcher<double>();
    else if (isFloat32(dtype))
        ext = (void*)new BipartiteGraphBFSearcher<float>();
    else
        RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    Py_INCREF(obj);
    return obj;
}

extern "C"
PyObject *bg_bf_searcher_delete(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        delete pyobjToCppObj<double>(objExt);
    else if (isFloat32(dtype))
        delete pyobjToCppObj<float>(objExt);
    else
        RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    
    Py_INCREF(Py_None);
    return Py_None;
}

extern "C"
PyObject *bg_bf_searcher_assign_device(PyObject *module, PyObject *args) {
    PyObject *objExt, *objDevice, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objDevice, &dtype))
        return NULL;

    sqcu::Device *device = (sqcu::Device*)PyArrayScalar_VAL(objDevice, UInt64);
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->assignDevice(*device);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->assignDevice(*device);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;
}
    

template<class real>
void internal_bg_bf_searcher_set_problem(PyObject *objExt,
                                       PyObject *objB0, PyObject *objB1, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpVector b0(objB0), b1(objB1);
    NpMatrix W(objW);
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(b0, b1, W, om);
}
    
extern "C"
PyObject *bg_bf_searcher_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objB0, *objB1, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOOOiO", &objExt, &objB0, &objB1, &objW, &opt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_bg_bf_searcher_set_problem<double>(objExt, objB0, objB1, objW, opt);
        else if (isFloat32(dtype))
            internal_bg_bf_searcher_set_problem<float>(objExt, objB0, objB1, objW, opt);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *bg_bf_searcher_set_preferences(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype, *objPrefs;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objPrefs, &dtype))
        return NULL;

    sq::Preferences prefs;
    if (parsePreferences(objPrefs, &prefs, Cuda_BgBfSearcherError) == -1)
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->setPreferences(prefs);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->setPreferences(prefs);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *bg_bf_searcher_get_preferences(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    sq::Preferences prefs;

    TRY {
        if (isFloat64(dtype))
            prefs = pyobjToCppObj<double>(objExt)->getPreferences();
        else if (isFloat32(dtype))
            prefs = pyobjToCppObj<float>(objExt)->getPreferences();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    return createPreferences(prefs);    
}


template<class real>
PyObject *internal_bg_bf_searcher_get_x(PyObject *objExt) {
    BipartiteGraphBFSearcher<real> *sol = pyobjToCppObj<real>(objExt);
    const sq::BitsPairArray &xList = sol->get_x();

    sq::SizeType N0, N1;
    sol->getProblemSize(&N0, &N1);
    
    PyObject *list = PyList_New(xList.size());
    for (size_t idx = 0; idx < xList.size(); ++idx) {
        const sq::BitsPairArray::ValueType &pair = xList[idx];
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
PyObject *bg_bf_searcher_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_bg_bf_searcher_get_x<double>(objExt);
        else if (isFloat32(dtype))
            return internal_bg_bf_searcher_get_x<float>(objExt);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
}


template<class real>
PyObject *internal_bg_bf_searcher_get_E(PyObject *objExt, int typenum) {
    typedef NpVectorType<real> NpVector;
    const sqaod::VectorType<real> &E = pyobjToCppObj<real>(objExt)->get_E();
    NpVector npE(E.size, typenum); /* allocate PyObject */
    npE.vec = E;
    return npE.obj;
}

    
extern "C"
PyObject *bg_bf_searcher_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_bg_bf_searcher_get_E<double>(objExt, NPY_FLOAT64);
        else if (isFloat32(dtype))
            return internal_bg_bf_searcher_get_E<float>(objExt, NPY_FLOAT32);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
}
    

extern "C"
PyObject *bg_bf_searcher_init_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->initSearch();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->initSearch();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;    
}


extern "C"
PyObject *bg_bf_searcher_fin_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->finSearch();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->finSearch();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *bg_bf_searcher_search_range(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    unsigned long long iBegin0, iEnd0, iBegin1, iEnd1;
    if (!PyArg_ParseTuple(args, "OKKKKO", &objExt, &iBegin0, &iEnd0, &iBegin1, &iEnd1, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->searchRange(iBegin0, iEnd0, iBegin1, iEnd1);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->searchRange(iBegin0, iEnd0, iBegin1, iEnd1);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *bg_bf_searcher_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->search();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->search();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_BgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_BgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;    
}

}


static
PyMethodDef cuda_bg_bf_searcher_methods[] = {
	{"new_searcher", bg_bf_searcher_create, METH_VARARGS},
	{"delete_searcher", bg_bf_searcher_delete, METH_VARARGS},
	{"assign_device", bg_bf_searcher_assign_device, METH_VARARGS},
	{"set_problem", bg_bf_searcher_set_problem, METH_VARARGS},
	{"set_preferences", bg_bf_searcher_set_preferences, METH_VARARGS},
	{"get_preferences", bg_bf_searcher_get_preferences, METH_VARARGS},
	{"get_x", bg_bf_searcher_get_x, METH_VARARGS},
	{"get_E", bg_bf_searcher_get_E, METH_VARARGS},
	{"init_search", bg_bf_searcher_init_search, METH_VARARGS},
	{"fin_search", bg_bf_searcher_fin_search, METH_VARARGS},
	{"search_range", bg_bf_searcher_search_range, METH_VARARGS},
	{"search", bg_bf_searcher_search, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcuda_bg_bf_searcher(void) {
    PyObject *m;
    
    m = Py_InitModule("cuda_bg_bf_searcher", cuda_bg_bf_searcher_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cuda_bg_searcher.error";
    Cuda_BgBfSearcherError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cuda_BgBfSearcherError);
    PyModule_AddObject(m, "error", Cuda_BgBfSearcherError);
}
