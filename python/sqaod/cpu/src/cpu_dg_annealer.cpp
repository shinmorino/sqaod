#include <pyglue.h>
#include <cpu/CPUFormulas.h>
#include <cpu/CPUDenseGraphAnnealer.h>
#include <string.h>


static PyObject *Cpu_DgSolverError;
namespace sqd = sqaod;


namespace {
    
template<class real>
sqd::CPUDenseGraphAnnealer<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<sqd::CPUDenseGraphAnnealer<real> *>(val);
}

extern "C"
PyObject *dg_annealer_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new sqd::CPUDenseGraphAnnealer<double>();
    else if (isFloat32(dtype))
        ext = (void*)new sqd::CPUDenseGraphAnnealer<float>();
    else
        RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    
    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)ext);
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
        RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    
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
        RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
void internal_dg_annealer_set_problem(PyObject *objExt, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    NpMatrix W(objW);
    sqd::OptimizeMethod om = (opt == 0) ? sqd::optMinimize : sqd::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(W, om);
}
    
extern "C"
PyObject *dg_annealer_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOiO", &objExt, &objW, &opt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_set_problem<double>(objExt, objW, opt);
        else if (isFloat32(dtype))
            internal_dg_annealer_set_problem<float>(objExt, objW, opt);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_annealer_get_problem_size(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    sqaod::SizeType N, m;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->getProblemSize(&N, &m);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->getProblemSize(&N, &m);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    return Py_BuildValue("II", N, m);
}
    
extern "C"
PyObject *dg_annealer_set_solver_preference(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    sqaod::SizeType m = 0;
    if (!PyArg_ParseTuple(args, "OIO", &objExt, &m, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->setNumTrotters(m);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->setNumTrotters(m);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dg_annealer_get_E(PyObject *objExt, PyObject *objE) {
    typedef NpVectorType<real> NpVector;
    NpVector E(objE);
    sqd::CPUDenseGraphAnnealer<real> *ext = pyobjToCppObj<real>(objExt);
    E.vec = ext->get_E();
}

    
extern "C"
PyObject *dg_annealer_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *objE, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objE, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_get_E<double>(objExt, objE);
        else if (isFloat32(dtype))
            internal_dg_annealer_get_E<float>(objExt, objE);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);
        
    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
PyObject *internal_dg_annealer_get_x(PyObject *objExt) {
    sqd::CPUDenseGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N, m;
    ann->getProblemSize(&N, &m);
    const sqaod::BitsArray &xList = ann->get_x();
    PyObject *list = PyList_New(xList.size());
    for (size_t idx = 0; idx < xList.size(); ++idx) {
        NpBitVector x(N, NPY_INT8);
        x.vec = xList[idx];
        PyList_SET_ITEM(list, idx, x.obj);
    }
    return list;
}
    
extern "C"
PyObject *dg_annealer_get_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_dg_annealer_get_x<double>(objExt);
        else if (isFloat32(dtype))
            return internal_dg_annealer_get_x<float>(objExt);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
}


template<class real>
void internal_dg_annealer_set_x(PyObject *objExt, PyObject *objX) {
    NpBitVector x(objX);
    pyobjToCppObj<real>(objExt)->set_x(x);
}

extern "C"
PyObject *dg_annealer_set_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *objX, *dtype;
    if (!PyArg_ParseTuple(args, "OOOO", &objExt, &objX, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_set_x<double>(objExt, objX);
        else if (isFloat32(dtype))
            internal_dg_annealer_set_x<float>(objExt, objX);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dg_annealer_get_hJc(PyObject *objExt,
                                  PyObject *objH, PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpScalarRefType<real> NpScalarRef;

    NpMatrix J(objJ);
    NpVector h(objH);
    NpScalarRef c(objC);
    
    sqd::CPUDenseGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);
    ann->get_hJc(&h, &J, &c);
}
    
    
extern "C"
PyObject *dg_annealer_get_hJc(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objH, &objJ, &objC, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_get_hJc<double>(objExt, objH, objJ, objC);
        else if (isFloat32(dtype))
            internal_dg_annealer_get_hJc<float>(objExt, objH, objJ, objC);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
template<class real>
PyObject *internal_dg_annealer_get_q(PyObject *objExt) {
    sqd::CPUDenseGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N, m;
    ann->getProblemSize(&N, &m);
    const sqaod::BitsArray &qList = ann->get_q();
    PyObject *list = PyList_New(qList.size());
    for (size_t idx = 0; idx < qList.size(); ++idx) {
        NpBitVector q(N, NPY_INT8);
        q.vec = qList[idx];
        PyList_SET_ITEM(list, idx, q.obj);
    }
    return list;
}
    
extern "C"
PyObject *dg_annealer_get_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_dg_annealer_get_q<double>(objExt);
        else if (isFloat32(dtype))
            return internal_dg_annealer_get_q<float>(objExt);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
}
    
extern "C"
PyObject *dg_annealer_radomize_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->randomize_q();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->randomize_q();
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_annealer_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->calculate_E();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->calculate_E();
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}

        
extern "C"
PyObject *dg_annealer_init_anneal(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->initAnneal();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->initAnneal();
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_annealer_fin_anneal(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->finAnneal();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->finAnneal();
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dg_annealer_anneal_one_step(PyObject *objExt, PyObject *objG, PyObject *objKT) {
    typedef NpConstScalarType<real> NpConstScalar;
    NpConstScalar G(objG), kT(objKT);
    pyobjToCppObj<real>(objExt)->annealOneStep(G, kT);
}

extern "C"
PyObject *dg_annealer_anneal_one_step(PyObject *module, PyObject *args) {
    PyObject *objExt, *objG, *objKT, *dtype;
    if (!PyArg_ParseTuple(args, "OOOO", &objExt, &objG, &objKT, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_anneal_one_step<double>(objExt, objG, objKT);
        else if (isFloat32(dtype))
            internal_dg_annealer_anneal_one_step<float>(objExt, objG, objKT);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_DgSolverError);
    } CATCH_ERROR_AND_RETURN(Cpu_DgSolverError);

    Py_INCREF(Py_None);
    return Py_None;    
}

}




static
PyMethodDef cpu_dg_annealer_methods[] = {
	{"new_annealer", dg_annealer_create, METH_VARARGS},
	{"delete_annealer", dg_annealer_delete, METH_VARARGS},
	{"rand_seed", dg_annealer_rand_seed, METH_VARARGS},
	{"set_problem", dg_annealer_set_problem, METH_VARARGS},
	{"get_problem_size", dg_annealer_get_problem_size, METH_VARARGS},
	{"set_solver_preference", dg_annealer_set_solver_preference, METH_VARARGS},
	{"get_E", dg_annealer_get_E, METH_VARARGS},
	{"get_x", dg_annealer_get_x, METH_VARARGS},
	{"set_x", dg_annealer_set_x, METH_VARARGS},
	{"get_hJc", dg_annealer_get_hJc, METH_VARARGS},
	{"get_q", dg_annealer_get_q, METH_VARARGS},
	{"randomize_q", dg_annealer_radomize_q, METH_VARARGS},
	{"calculate_E", dg_annealer_calculate_E, METH_VARARGS},
	{"init_anneal", dg_annealer_init_anneal, METH_VARARGS},
	{"fin_anneal", dg_annealer_fin_anneal, METH_VARARGS},
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
