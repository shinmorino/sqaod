#include <pyglue.h>
#include <cpu/Traits.h>
#include <cpu/CPUDenseGraphBFSolver.h>
#include <string.h>


/* FIXME : remove DONT_REACH_HERE macro */


// http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
// http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

/* NOTE: Value type checks for python objs have been already done in python glue, 
 * Here we only get entities needed. */


static PyObject *Cpu_DgBfSolverError;
namespace sqd = sqaod;


namespace {



void setErrInvalidDtype(PyObject *dtype) {
    PyErr_SetString(Cpu_DgBfSolverError, "dtype must be numpy.float64 or numpy.float32.");
}

#define RAISE_INVALID_DTYPE(dtype) {setErrInvalidDtype(dtype); return NULL; }

    
template<class real>
sqd::CPUDenseGraphBFSolver<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<sqd::CPUDenseGraphBFSolver<real>*>(val);
}

extern "C"
PyObject *dg_bf_solver_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new sqd::CPUDenseGraphBFSolver<double>();
    else if (isFloat32(dtype))
        ext = (void*)new sqd::CPUDenseGraphBFSolver<float>();
    else
        RAISE_INVALID_DTYPE(dtype);
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    Py_INCREF(obj);
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
        RAISE_INVALID_DTYPE(dtype);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_bf_solver_rand_seed(PyObject *module, PyObject *args) {
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
void internal_dg_bf_solver_set_problem(PyObject *objExt, PyObject *objW, int opt) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix W(objW);
    sqd::OptimizeMethod om = (opt == 0) ? sqd::optMinimize : sqd::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(W, W.dims[0], om);
}
    
extern "C"
PyObject *dg_bf_solver_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOiO", &objExt, &objW, &opt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_dg_bf_solver_set_problem<double>(objExt, objW, opt);
    else if (isFloat32(dtype))
        internal_dg_bf_solver_set_problem<float>(objExt, objW, opt);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_bf_solver_set_solver_preference(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    int tileSize;
    if (!PyArg_ParseTuple(args, "OiO", &objExt, &tileSize, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->setTileSize(tileSize);
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->setTileSize(tileSize);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
PyObject *internal_dg_bf_solver_get_x(PyObject *objExt) {
    int N;
    sqd::CPUDenseGraphBFSolver<real> *sol = pyobjToCppObj<real>(objExt);
    const sqd::BitMatrix &xList = sol->get_x();
    sol->getProblemSize(&N);

    NpBitMatrix bit;
    bit.allocate((int)xList.rows(), xList.cols());
    memcpy(bit.data, xList.data(), sizeof(char) * xList.size());
    return bit.obj;
}
    
    
extern "C"
PyObject *dg_bf_solver_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *objX, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        objX = internal_dg_bf_solver_get_x<double>(objExt);
    else if (isFloat32(dtype))
        objX = internal_dg_bf_solver_get_x<float>(objExt);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(objX);
    return objX;    
}


template<class real>
PyObject *internal_dg_bf_solver_get_E(PyObject *objExt) {
    sqd::CPUDenseGraphBFSolver<real> *ext = pyobjToCppObj<real>(objExt);
    real E = ext->get_E();
    return newScalarObj(E);
}

    
extern "C"
PyObject *dg_bf_solver_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype, *objE;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        objE = internal_dg_bf_solver_get_E<double>(objExt);
    else if (isFloat32(dtype))
        objE = internal_dg_bf_solver_get_E<float>(objExt);
    else
        RAISE_INVALID_DTYPE(dtype);
    Py_INCREF(objE);
    return objE;
}


extern "C"
PyObject *dg_bf_solver_init_search(PyObject *module, PyObject *args) {
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
PyObject *dg_bf_solver_search_range(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    unsigned long long iBegin, iEnd;
    if (!PyArg_ParseTuple(args, "OKKO", &objExt, &iBegin, &iEnd, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->searchRange(iBegin, iEnd);
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->searchRange(iBegin, iEnd);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_bf_solver_search(PyObject *module, PyObject *args) {
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
PyMethodDef cpu_dg_bf_solver_methods[] = {
	{"new_bf_solver", dg_bf_solver_create, METH_VARARGS},
	{"delete_bf_solver", dg_bf_solver_delete, METH_VARARGS},
	{"rand_seed", dg_bf_solver_rand_seed, METH_VARARGS},
	{"set_problem", dg_bf_solver_set_problem, METH_VARARGS},
	{"set_solver_preference", dg_bf_solver_set_solver_preference, METH_VARARGS},
	{"get_x", dg_bf_solver_get_x, METH_VARARGS},
	{"get_E", dg_bf_solver_get_E, METH_VARARGS},
	{"init_search", dg_bf_solver_init_search, METH_VARARGS},
	{"search_range", dg_bf_solver_search_range, METH_VARARGS},
	{"search", dg_bf_solver_search, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcpu_dg_bf_solver(void) {
    PyObject *m;
    
    m = Py_InitModule("cpu_dg_bf_solver", cpu_dg_bf_solver_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cpu_dg_solver.error";
    Cpu_DgBfSolverError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cpu_DgBfSolverError);
    PyModule_AddObject(m, "error", Cpu_DgBfSolverError);
}
