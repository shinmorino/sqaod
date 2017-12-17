#include <pyglue.h>
#include <cpu/Traits.h>
#include <cpu/CPUDenseGraphAnnealer.h>
#include <string.h>


/* FIXME : remove DONT_REACH_HERE macro */


// http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
// http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

/* NOTE: Value type checks for python objs have been already done in python glue, 
 * Here we only get entities needed. */


static PyObject *Cpu_DgSolverError;
namespace qd = quantd_cpu;


namespace {



void setErrInvalidDtype(PyObject *dtype) {
    PyErr_SetString(Cpu_DgSolverError, "dtype must be numpy.float64 or numpy.float32.");
}

#define RAISE_INVALID_DTYPE(dtype) {setErrInvalidDtype(dtype); return NULL; }

    
template<class real>
qd::CPUDenseGraphAnnealer<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<qd::CPUDenseGraphAnnealer<real> *>(val);
}

extern "C"
PyObject *dg_annealer_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new qd::CPUDenseGraphAnnealer<double>();
    else if (isFloat32(dtype))
        ext = (void*)new qd::CPUDenseGraphAnnealer<float>();
    else
        RAISE_INVALID_DTYPE(dtype);
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    Py_INCREF(obj);
    return obj;
}

extern "C"
PyObject *dg_annealer_delete(PyObject *module, PyObject *args) {
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
PyObject *dg_annealer_rand_seed(PyObject *module, PyObject *args) {
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
    
    
    
extern "C"
PyObject *dg_annealer_set_problem_size(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    int N = 0, m = 0;
    if (!PyArg_ParseTuple(args, "OiiO", &objExt, &N, &m, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->setProblemSize(N, m);
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->setProblemSize(N, m);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
void internal_dg_annealer_set_problem(PyObject *objExt, PyObject *objW, int opt) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix W(objW);
    qd::OptimizeMethod om = (opt == 0) ? qd::optMinimize : qd::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(W, om);
}
    
extern "C"
PyObject *dg_annealer_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOiO", &objExt, &objW, &opt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_dg_annealer_set_problem<double>(objExt, objW, opt);
    else if (isFloat32(dtype))
        internal_dg_annealer_set_problem<float>(objExt, objW, opt);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real>
void internal_dg_annealer_get_q(PyObject *objExt, PyObject *objQ) {
    NpBitMatrix Q(objQ);
    int N, m;
    qd::CPUDenseGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);
    const char *q = ann->get_q();
    ann->getProblemSize(&N, &m);
    memcpy(Q.data, q, N * m);
}
    
    
extern "C"
PyObject *dg_annealer_get_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *objQ, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objQ, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_dg_annealer_get_q<double>(objExt, objQ);
    else if (isFloat32(dtype))
        internal_dg_annealer_get_q<float>(objExt, objQ);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_annealer_radomize_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->randomize_q();
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->randomize_q();
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dg_annealer_get_hJc(PyObject *objExt,
                                  PyObject *objH, PyObject *objJ, PyObject *objC) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix h(objH), J(objJ), c(objC);
    const real *nh, *nJ;
    int N, m;
    
    qd::CPUDenseGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);
    ann->getProblemSize(&N, &m);
    ann->get_hJc(&nh, &nJ, c);
    memcpy(h, nh, sizeof(real) * N);
    memcpy(J, nJ, sizeof(real) * N * N);
}
    
    
extern "C"
PyObject *dg_annealer_get_hJc(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objH, &objJ, &objC, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_dg_annealer_get_hJc<double>(objExt, objH, objJ, objC);
    else if (isFloat32(dtype))
        internal_dg_annealer_get_hJc<float>(objExt, objH, objJ, objC);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dg_annealer_get_E(PyObject *objExt, PyObject *objE) {
    typedef NpMatrixT<real> NpMatrix;
    NpMatrix E(objE);

    int N, m;
    qd::CPUDenseGraphAnnealer<real> *ext = pyobjToCppObj<real>(objExt);
    ext->getProblemSize(&N, &m);
    const real *nE = ext->get_E();
    memcpy(E, nE, sizeof(real) * m);
}

    
extern "C"
PyObject *dg_annealer_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objE, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_dg_annealer_get_E<double>(objExt, objE);
    else if (isFloat32(dtype))
        internal_dg_annealer_get_q<float>(objExt, objE);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
extern "C"
PyObject *dg_annealer_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->calculate_E();
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->calculate_E();
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real>
void internal_dg_annealer_anneal_one_step(PyObject *objExt, PyObject *objG, PyObject *objKT) {
    typedef NpConstScalarT<real> NpConstScalar;
    NpConstScalar G(objG), kT(objKT);
    pyobjToCppObj<real>(objExt)->annealOneStep(G, kT);
}


extern "C"
PyObject *dg_annealer_anneal_one_step(PyObject *module, PyObject *args) {
    PyObject *objExt, *objG, *objKT, *dtype;
    if (!PyArg_ParseTuple(args, "OOOO", &objExt, &objG, &objKT, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_dg_annealer_anneal_one_step<double>(objExt, objG, objKT);
    else if (isFloat32(dtype))
        internal_dg_annealer_anneal_one_step<float>(objExt, objG, objKT);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

    

}




static
PyMethodDef cpu_dg_annealer_methods[] = {
	{"new_annealer", dg_annealer_create, METH_VARARGS},
	{"delete_annealer", dg_annealer_delete, METH_VARARGS},
	{"rand_seed", dg_annealer_rand_seed, METH_VARARGS},
	{"set_problem_size", dg_annealer_set_problem_size, METH_VARARGS},
	{"set_problem", dg_annealer_set_problem, METH_VARARGS},
	{"get_q", dg_annealer_get_q, METH_VARARGS},
	{"randomize_q", dg_annealer_radomize_q, METH_VARARGS},
	{"get_hJc", dg_annealer_get_hJc, METH_VARARGS},
	{"get_E", dg_annealer_get_E, METH_VARARGS},
	{"calculate_E", dg_annealer_calculate_E, METH_VARARGS},
	{"anneal_one_step", dg_annealer_anneal_one_step, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcpu_dg_annealer(void) {
    PyObject *m;
    
    m = Py_InitModule("cpu_dg_annealer", cpu_dg_annealer_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cpu_dg_solver.error";
    Cpu_DgSolverError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cpu_DgSolverError);
    PyModule_AddObject(m, "error", Cpu_DgSolverError);
}
