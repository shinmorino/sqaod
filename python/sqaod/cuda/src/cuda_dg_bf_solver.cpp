#include <pyglue.h>
#include <cuda/CUDAFormulas.h>
#include <cuda/CUDADenseGraphBFSolver.h>
#include <string.h>


static PyObject *Cuda_DgBfSolverError;
namespace sq = sqaod;
namespace sqcu = sqaod_cuda;

namespace {


template<class real>
sqcu::CUDADenseGraphBFSolver<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<sqcu::CUDADenseGraphBFSolver<real>*>(val);
}

extern "C"
PyObject *dg_bf_solver_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new sqcu::CUDADenseGraphBFSolver<double>();
    else if (isFloat32(dtype))
        ext = (void*)new sqcu::CUDADenseGraphBFSolver<float>();
    else
        RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    return obj;
}

extern "C"
PyObject *dg_bf_solver_delete(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        delete pyobjToCppObj<double>(objExt);
    else if (isFloat32(dtype))
        delete pyobjToCppObj<float>(objExt);
    else
        RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dg_bf_solver_set_problem(PyObject *objExt, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    const NpMatrix W(objW);
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(W, om);
}
    
extern "C"
PyObject *dg_bf_solver_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOiO", &objExt, &objW, &opt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_bf_solver_set_problem<double>(objExt, objW, opt);
        else if (isFloat32(dtype))
            internal_dg_bf_solver_set_problem<float>(objExt, objW, opt);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSolverError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_bf_solver_set_solver_preference(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    sqaod::SizeType tileSize;
    if (!PyArg_ParseTuple(args, "OIO", &objExt, &tileSize, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->setTileSize(tileSize);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->setTileSize(tileSize);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSolverError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
PyObject *internal_dg_bf_solver_get_x(PyObject *objExt) {
    sqaod::SizeType N;
    sqcu::CUDADenseGraphBFSolver<real> *sol = pyobjToCppObj<real>(objExt);
    const sq::BitsArray &xList = sol->get_x();
    sol->getProblemSize(&N);

    PyObject *list = PyList_New(xList.size());
    for (size_t idx = 0; idx < xList.size(); ++idx) {
        const sq::Bits &bits = xList[idx];
        NpBitVector x(N, NPY_INT8);
        x.vec = bits;
        PyList_SET_ITEM(list, idx, x.obj);
    }
    return list;
}
    
    
extern "C"
PyObject *dg_bf_solver_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_dg_bf_solver_get_x<double>(objExt);
        else if (isFloat32(dtype))
            return internal_dg_bf_solver_get_x<float>(objExt);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSolverError);

    RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
}


template<class real>
PyObject *internal_dg_bf_solver_get_E(PyObject *objExt, int typenum) {
    typedef NpVectorType<real> NpVector;
    const sqaod::VectorType<real> &E = pyobjToCppObj<real>(objExt)->get_E();
    NpVector npE(E.size, typenum); /* allocate PyObject */
    npE.vec = E;
    return npE.obj;
}
    
extern "C"
PyObject *dg_bf_solver_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_dg_bf_solver_get_E<double>(objExt, NPY_FLOAT64);
        else if (isFloat32(dtype))
            return internal_dg_bf_solver_get_E<float>(objExt, NPY_FLOAT32);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSolverError);
        
    RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
}


extern "C"
PyObject *dg_bf_solver_init_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->initSearch();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->initSearch();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_bf_solver_fin_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->finSearch();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->finSearch();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
extern "C"
PyObject *dg_bf_solver_search_range(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    unsigned long long iBegin, iEnd;
    if (!PyArg_ParseTuple(args, "OKKO", &objExt, &iBegin, &iEnd, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->searchRange(iBegin, iEnd);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->searchRange(iBegin, iEnd);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_bf_solver_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->search();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->search();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgBfSolverError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgBfSolverError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

}


static
PyMethodDef cuda_dg_bf_solver_methods[] = {
	{"new_bf_solver", dg_bf_solver_create, METH_VARARGS},
	{"delete_bf_solver", dg_bf_solver_delete, METH_VARARGS},
	{"set_problem", dg_bf_solver_set_problem, METH_VARARGS},
	{"set_solver_preference", dg_bf_solver_set_solver_preference, METH_VARARGS},
	{"get_x", dg_bf_solver_get_x, METH_VARARGS},
	{"get_E", dg_bf_solver_get_E, METH_VARARGS},
	{"init_search", dg_bf_solver_init_search, METH_VARARGS},
	{"fin_search", dg_bf_solver_fin_search, METH_VARARGS},
	{"search_range", dg_bf_solver_search_range, METH_VARARGS},
	{"search", dg_bf_solver_search, METH_VARARGS},
	{NULL},
};

extern "C"
PyMODINIT_FUNC
initcuda_dg_bf_solver(void) {
    PyObject *m;
    
    m = Py_InitModule("cuda_dg_bf_solver", cuda_dg_bf_solver_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cuda_dg_solver.error";
    Cuda_DgBfSolverError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cuda_DgBfSolverError);
    PyModule_AddObject(m, "error", Cuda_DgBfSolverError);
}
