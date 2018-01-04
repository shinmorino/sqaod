#include <pyglue.h>
#include <common/Common.h>
#include <cpu/CPUBipartiteGraphBFSolver.h>
#include <string.h>


/* FIXME : remove DONT_REACH_HERE macro */


// http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
// http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

/* NOTE: Value type checks for python objs have been already done in python glue, 
 * Here we only get entities needed. */


static PyObject *Cpu_BgBfSolverError;
namespace sqd = sqaod;


namespace {



void setErrInvalidDtype(PyObject *dtype) {
    PyErr_SetString(Cpu_BgBfSolverError, "dtype must be numpy.float64 or numpy.float32.");
}

#define RAISE_INVALID_DTYPE(dtype) {setErrInvalidDtype(dtype); return NULL; }

    
template<class real>
sqd::
CPUBipartiteGraphBFSolver<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<sqd::CPUBipartiteGraphBFSolver<real>*>(val);
}

extern "C"
PyObject *bg_bf_solver_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new sqd::CPUBipartiteGraphBFSolver<double>();
    else if (isFloat32(dtype))
        ext = (void*)new sqd::CPUBipartiteGraphBFSolver<float>();
    else
        RAISE_INVALID_DTYPE(dtype);
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    Py_INCREF(obj);
    return obj;
}

extern "C"
PyObject *bg_bf_solver_delete(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        delete pyobjToCppObj<double>(objExt);
    else if (isFloat32(dtype))
        delete pyobjToCppObj<float>(objExt);
    else
        RAISE_INVALID_DTYPE(dtype);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *bg_bf_solver_rand_seed(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    unsigned long long seed;
    if (!PyArg_ParseTuple(args, "OKO", &objExt, &seed, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->seed(seed);
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->seed(seed);
    else
        RAISE_INVALID_DTYPE(dtype);
    
    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real>
void internal_bg_bf_solver_set_problem(PyObject *objExt,
                                       PyObject *objB0, PyObject *objB1, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    NpVector b0(objB0), b1(objB1);
    NpMatrix W(objW);
    sqd::OptimizeMethod om = (opt == 0) ? sqd::optMinimize : sqd::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(b0, b1, W, om);
}
    
extern "C"
PyObject *bg_bf_solver_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objB0, *objB1, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOOOiO", &objExt, &objB0, &objB1, &objW, &opt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_bg_bf_solver_set_problem<double>(objExt, objB0, objB1, objW, opt);
    else if (isFloat32(dtype))
        internal_bg_bf_solver_set_problem<float>(objExt, objB0, objB1, objW, opt);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *bg_bf_solver_set_solver_preference(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    int tileSize0, tileSize1;
    if (!PyArg_ParseTuple(args, "OiiO", &objExt, &tileSize0, &tileSize1, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->setTileSize(tileSize0, tileSize1);
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->setTileSize(tileSize0, tileSize1);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
PyObject *internal_bg_bf_solver_get_x(PyObject *objExt) {
    sqd::CPUBipartiteGraphBFSolver<real> *sol = pyobjToCppObj<real>(objExt);
    const sqd::BitsPairArray &xList = sol->get_x();

    int N0, N1;
    sol->getProblemSize(&N0, &N1);
    
    PyObject *list = PyList_New(xList.size());
    for (size_t idx = 0; idx < xList.size(); ++idx) {
        const sqd::BitsPairArray::ValueType &pair = xList[idx];
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
PyObject *bg_bf_solver_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        return internal_bg_bf_solver_get_x<double>(objExt);
    else if (isFloat32(dtype))
        return internal_bg_bf_solver_get_x<float>(objExt);
    RAISE_INVALID_DTYPE(dtype);
}


template<class real>
PyObject *internal_bg_bf_solver_get_E(PyObject *objExt, int typenum) {
    typedef NpVectorType<real> NpVector;
    const sqaod::VectorType<real> &E = pyobjToCppObj<real>(objExt)->get_E();
    NpVector npE(E.size, typenum); /* allocate PyObject */
    npE.vec = E;
    return npE.obj;
}

    
extern "C"
PyObject *bg_bf_solver_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        return internal_bg_bf_solver_get_E<double>(objExt, NPY_FLOAT64);
    else if (isFloat32(dtype))
        return internal_bg_bf_solver_get_E<float>(objExt, NPY_FLOAT32);

    RAISE_INVALID_DTYPE(dtype);
}
    

extern "C"
PyObject *bg_bf_solver_init_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->initSearch();
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->initSearch();
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


extern "C"
PyObject *bg_bf_solver_fin_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->finSearch();
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->finSearch();
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *bg_bf_solver_search_range(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    unsigned long long iBegin0, iEnd0, iBegin1, iEnd1;
    if (!PyArg_ParseTuple(args, "OKKKKO", &objExt, &iBegin0, &iEnd0, &iBegin1, &iEnd1, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->searchRange(iBegin0, iEnd0, iBegin1, iEnd1);
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->searchRange(iBegin0, iEnd0, iBegin1, iEnd1);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *bg_bf_solver_search(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->search();
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->search();
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

    

}




static
PyMethodDef cpu_bg_bf_solver_methods[] = {
	{"new_solver", bg_bf_solver_create, METH_VARARGS},
	{"delete_solver", bg_bf_solver_delete, METH_VARARGS},
	{"rand_seed", bg_bf_solver_rand_seed, METH_VARARGS},
	{"set_problem", bg_bf_solver_set_problem, METH_VARARGS},
	{"set_solver_preference", bg_bf_solver_set_solver_preference, METH_VARARGS},
	{"get_x", bg_bf_solver_get_x, METH_VARARGS},
	{"get_E", bg_bf_solver_get_E, METH_VARARGS},
	{"init_search", bg_bf_solver_init_search, METH_VARARGS},
	{"fin_search", bg_bf_solver_fin_search, METH_VARARGS},
	{"search_range", bg_bf_solver_search_range, METH_VARARGS},
	{"search", bg_bf_solver_search, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcpu_bg_bf_solver(void) {
    PyObject *m;
    
    m = Py_InitModule("cpu_bg_bf_solver", cpu_bg_bf_solver_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cpu_bg_solver.error";
    Cpu_BgBfSolverError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cpu_BgBfSolverError);
    PyModule_AddObject(m, "error", Cpu_BgBfSolverError);
}
