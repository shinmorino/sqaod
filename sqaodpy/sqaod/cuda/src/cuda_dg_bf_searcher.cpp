#include <sqaodc/pyglue/pyglue.h>
#include <sqaodc/sqaodc.h>


static PyObject *Cuda_DgBfSearcherError;
namespace sq = sqaod;
namespace sqcu = sqaod::cuda;

template<class real>
using DenseGraphBFSearcher = sq::cuda::DenseGraphBFSearcher<real>;


namespace {


template<class real>
DenseGraphBFSearcher<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<DenseGraphBFSearcher<real>*>(val);
}

extern "C"
PyObject *dg_bf_searcher_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new DenseGraphBFSearcher<double>();
    else if (isFloat32(dtype))
        ext = (void*)new DenseGraphBFSearcher<float>();
    else
        RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    return obj;
}

extern "C"
PyObject *dg_bf_searcher_delete(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        delete pyobjToCppObj<double>(objExt);
    else if (isFloat32(dtype))
        delete pyobjToCppObj<float>(objExt);
    else
        RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_bf_searcher_assign_device(PyObject *module, PyObject *args) {
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
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
void internal_dg_bf_searcher_set_problem(PyObject *objExt, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    const NpMatrix W(objW);
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(W, om);
}
    
extern "C"
PyObject *dg_bf_searcher_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOiO", &objExt, &objW, &opt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_bf_searcher_set_problem<double>(objExt, objW, opt);
        else if (isFloat32(dtype))
            internal_dg_bf_searcher_set_problem<float>(objExt, objW, opt);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_bf_searcher_set_preferences(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype, *objPrefs;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objPrefs, &dtype))
        return NULL;

    sq::Preferences prefs;
    if (parsePreferences(objPrefs, &prefs, Cuda_DgBfSearcherError) == -1)
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->setPreferences(prefs);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->setPreferences(prefs);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_bf_searcher_get_preferences(PyObject *module, PyObject *args) {
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
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);

    return createPreferences(prefs);    
}

template<class real>
PyObject *internal_dg_bf_searcher_get_x(PyObject *objExt) {
    sqaod::SizeType N;
    DenseGraphBFSearcher<real> *sol = pyobjToCppObj<real>(objExt);
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
PyObject *dg_bf_searcher_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_dg_bf_searcher_get_x<double>(objExt);
        else if (isFloat32(dtype))
            return internal_dg_bf_searcher_get_x<float>(objExt);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);

    RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
}


template<class real>
PyObject *internal_dg_bf_searcher_get_E(PyObject *objExt, int typenum) {
    typedef NpVectorType<real> NpVector;
    const sqaod::VectorType<real> &E = pyobjToCppObj<real>(objExt)->get_E();
    NpVector npE(E.size, typenum); /* allocate PyObject */
    npE.vec = E;
    return npE.obj;
}
    
extern "C"
PyObject *dg_bf_searcher_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_dg_bf_searcher_get_E<double>(objExt, NPY_FLOAT64);
        else if (isFloat32(dtype))
            return internal_dg_bf_searcher_get_E<float>(objExt, NPY_FLOAT32);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);
        
    RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
}


extern "C"
PyObject *dg_bf_searcher_prepare(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->prepare();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->prepare();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_bf_searcher_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->calculate_E();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->calculate_E();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_bf_searcher_make_solution(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->makeSolution();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->makeSolution();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
extern "C"
PyObject *dg_bf_searcher_search_range(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OKKO", &objExt, &dtype))
        return NULL;

    sq::PackedBitSet curX;
    bool res;
    TRY {
        if (isFloat64(dtype))
            res = pyobjToCppObj<double>(objExt)->searchRange(&curX);
        else if (isFloat32(dtype))
            res = pyobjToCppObj<float>(objExt)->searchRange(&curX);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);

    return Py_BuildValue("OK", res ? Py_True : Py_False, curX);
}

extern "C"
PyObject *dg_bf_searcher_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->search();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->search();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSearcherError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSearcherError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

}



static
PyMethodDef cuda_dg_bf_searcher_methods[] = {
	{"new_bf_searcher", dg_bf_searcher_create, METH_VARARGS},
	{"delete_bf_searcher", dg_bf_searcher_delete, METH_VARARGS},
	{"assign_device", dg_bf_searcher_assign_device, METH_VARARGS},
	{"set_problem", dg_bf_searcher_set_problem, METH_VARARGS},
	{"set_preferences", dg_bf_searcher_set_preferences, METH_VARARGS},
	{"get_preferences", dg_bf_searcher_get_preferences, METH_VARARGS},
	{"get_x", dg_bf_searcher_get_x, METH_VARARGS},
	{"get_E", dg_bf_searcher_get_E, METH_VARARGS},
	{"prepare", dg_bf_searcher_prepare, METH_VARARGS},
	{"calculate_E", dg_bf_searcher_calculate_E, METH_VARARGS},
	{"make_solution", dg_bf_searcher_make_solution, METH_VARARGS},
	{"search_range", dg_bf_searcher_search_range, METH_VARARGS},
	{"search", dg_bf_searcher_search, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcuda_dg_bf_searcher(void) {
    PyObject *m;
    
    m = Py_InitModule("cuda_dg_bf_searcher", cuda_dg_bf_searcher_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cuda_dg_searcher.error";
    Cuda_DgBfSearcherError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cuda_DgBfSearcherError);
    PyModule_AddObject(m, "error", Cuda_DgBfSearcherError);
}
