#include <pyglue.h>
#include <cpu/CPUFormulas.h>
#include <cpu/CPUBipartiteGraphAnnealer.h>
#include <string.h>


/* FIXME : remove DONT_REACH_HERE macro */


// http://owa.as.wakwak.ne.jp/zope/docs/Python/BindingC/
// http://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

/* NOTE: Value type checks for python objs have been already done in python glue, 
 * Here we only get entities needed. */


static PyObject *Cpu_BgSolverError;
namespace sqd = sqaod;


namespace {



void setErrInvalidDtype(PyObject *dtype) {
    PyErr_SetString(Cpu_BgSolverError, "dtype must be numpy.float64 or numpy.float32.");
}

#define RAISE_INVALID_DTYPE(dtype) {setErrInvalidDtype(dtype); return NULL; }

    
template<class real>
sqd::CPUBipartiteGraphAnnealer<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<sqd::CPUBipartiteGraphAnnealer<real> *>(val);
}

extern "C"
PyObject *bg_annealer_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new sqd::CPUBipartiteGraphAnnealer<double>();
    else if (isFloat32(dtype))
        ext = (void*)new sqd::CPUBipartiteGraphAnnealer<float>();
    else
        RAISE_INVALID_DTYPE(dtype);
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
    return obj;
}

extern "C"
PyObject *bg_annealer_delete(PyObject *module, PyObject *args) {
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
PyObject *bg_annealer_rand_seed(PyObject *module, PyObject *args) {
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
void internal_bg_annealer_set_problem(PyObject *objExt,
                                      PyObject *objB0, PyObject *objB1, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    const NpVector b0(objB0), b1(objB1);
    const NpMatrix W(objW);
    sqd::OptimizeMethod om = (opt == 0) ? sqd::optMinimize : sqd::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(b0, b1, W, om);
}
    
extern "C"
PyObject *bg_annealer_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objB0, *objB1, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOOOiO", &objExt, &objB0, &objB1, &objW, &opt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_bg_annealer_set_problem<double>(objExt, objB0, objB1, objW, opt);
    else if (isFloat32(dtype))
        internal_bg_annealer_set_problem<float>(objExt, objB0, objB1, objW, opt);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
    
extern "C"
PyObject *bg_annealer_set_solver_preference(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    int m = 0;
    if (!PyArg_ParseTuple(args, "OiO", &objExt, &m, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->setNumTrotters(m);
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->setNumTrotters(m);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
PyObject *internal_bg_annealer_get_x(PyObject *objExt) {
    sqd::CPUBipartiteGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);

    int N0, N1, m;
    ann->getProblemSize(&N0, &N1, &m);
    const sqd::BitsPairArray &xPairList = ann->get_x();

    PyObject *list = PyList_New(xPairList.size());
    for (size_t idx = 0; idx < xPairList.size(); ++idx) {
        const sqd::BitsPairArray::value_type &pair = xPairList[idx];

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
PyObject *bg_annealer_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        return internal_bg_annealer_get_x<double>(objExt);
    else if (isFloat32(dtype))
        return internal_bg_annealer_get_x<float>(objExt);
    RAISE_INVALID_DTYPE(dtype);
}

    

template<class real>
PyObject *internal_bg_annealer_get_q(PyObject *objExt) {
    sqd::CPUBipartiteGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);

    int N0, N1, m;
    ann->getProblemSize(&N0, &N1, &m);
    const sqd::BitsPairArray &xPairList = ann->get_q();

    PyObject *list = PyList_New(xPairList.size());
    for (size_t idx = 0; idx < xPairList.size(); ++idx) {
        const sqd::BitsPairArray::value_type &pair = xPairList[idx];

        NpBitVector q0(N0, NPY_INT8), q1(N1, NPY_INT8);
        q0.vec = pair.first;
        q1.vec = pair.second;

        PyObject *tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(tuple, 0, q0.obj);
        PyTuple_SET_ITEM(tuple, 1, q1.obj);
        PyList_SET_ITEM(list, idx, tuple);
    }
    return list;
}
    
extern "C"
PyObject *bg_annealer_get_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        return internal_bg_annealer_get_q<double>(objExt);
    else if (isFloat32(dtype))
        return internal_bg_annealer_get_q<float>(objExt);
    RAISE_INVALID_DTYPE(dtype);
}
    
extern "C"
PyObject *bg_annealer_radomize_q(PyObject *module, PyObject *args) {
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
void internal_bg_annealer_get_hJc(PyObject *objExt,
                                  PyObject *objH0, PyObject *objH1,
                                  PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    
    sqd::CPUBipartiteGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);
    NpVector h0(objH0), h1(objH1), c(objC);
    NpMatrix J(objJ);
    ann->get_hJc(&h0, &h0, &J, c.vec.data);
}
    
    
extern "C"
PyObject *bg_annealer_get_hJc(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH0, *objH1, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objH0, &objH1, &objJ, &objC, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_bg_annealer_get_hJc<double>(objExt, objH0, objH1, objJ, objC);
    else if (isFloat32(dtype))
        internal_bg_annealer_get_hJc<float>(objExt, objH0, objH1, objJ, objC);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_bg_annealer_get_E(PyObject *objExt, PyObject *objE) {
    typedef NpVectorType<real> NpVector;
    NpVector E(objE);
    sqd::CPUBipartiteGraphAnnealer<real> *ext = pyobjToCppObj<real>(objExt);
    E.vec = ext->get_E();
}

    
extern "C"
PyObject *bg_annealer_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objE, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_bg_annealer_get_E<double>(objExt, objE);
    else if (isFloat32(dtype))
        internal_bg_annealer_get_E<float>(objExt, objE);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
extern "C"
PyObject *bg_annealer_calculate_E(PyObject *module, PyObject *args) {
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
    
extern "C"
PyObject *bg_annealer_init_anneal(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->initAnneal();
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->initAnneal();
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
template<class real>
void internal_bg_annealer_anneal_one_step(PyObject *objExt, PyObject *objG, PyObject *objKT) {
    typedef NpConstScalarT<real> NpConstScalar;
    NpConstScalar G(objG), kT(objKT);
    pyobjToCppObj<real>(objExt)->annealOneStep(G, kT);
}


extern "C"
PyObject *bg_annealer_anneal_one_step(PyObject *module, PyObject *args) {
    PyObject *objExt, *objG, *objKT, *dtype;
    if (!PyArg_ParseTuple(args, "OOOO", &objExt, &objG, &objKT, &dtype))
        return NULL;
    if (isFloat64(dtype))
        internal_bg_annealer_anneal_one_step<double>(objExt, objG, objKT);
    else if (isFloat32(dtype))
        internal_bg_annealer_anneal_one_step<float>(objExt, objG, objKT);
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *bg_annealer_fin_anneal(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->finAnneal();
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->finAnneal();
    else
        RAISE_INVALID_DTYPE(dtype);

    Py_INCREF(Py_None);
    return Py_None;    
}
    

}




static
PyMethodDef cpu_bg_annealer_methods[] = {
	{"new_annealer", bg_annealer_create, METH_VARARGS},
	{"delete_annealer", bg_annealer_delete, METH_VARARGS},
	{"rand_seed", bg_annealer_rand_seed, METH_VARARGS},
	{"set_problem", bg_annealer_set_problem, METH_VARARGS},
	{"set_solver_preference", bg_annealer_set_solver_preference, METH_VARARGS},
	{"get_E", bg_annealer_get_E, METH_VARARGS},
	{"get_x", bg_annealer_get_x, METH_VARARGS},
	{"get_hJc", bg_annealer_get_hJc, METH_VARARGS},
	{"get_q", bg_annealer_get_q, METH_VARARGS},
	{"randomize_q", bg_annealer_radomize_q, METH_VARARGS},
	{"calculate_E", bg_annealer_calculate_E, METH_VARARGS},
	{"init_anneal", bg_annealer_init_anneal, METH_VARARGS},
	{"fin_anneal", bg_annealer_fin_anneal, METH_VARARGS},
	{"anneal_one_step", bg_annealer_anneal_one_step, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcpu_bg_annealer(void) {
    PyObject *m;
    
    m = Py_InitModule("cpu_bg_annealer", cpu_bg_annealer_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cpu_dg_solver.error";
    Cpu_BgSolverError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cpu_BgSolverError);
    PyModule_AddObject(m, "error", Cpu_BgSolverError);
}
