#include <sqaodc/pyglue/pyglue.h>
#include <sqaodc/sqaodc.h>


static PyObject *Cpu_BgAnnealerError;
namespace sq = sqaod;

template<class real>
using BipartiteGraphAnnealer = sq::cpu::BipartiteGraphAnnealer<real>;


namespace {
    
template<class real>
BipartiteGraphAnnealer<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<BipartiteGraphAnnealer<real> *>(val);
}

extern "C"
PyObject *bg_annealer_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new BipartiteGraphAnnealer<double>();
    else if (isFloat32(dtype))
        ext = (void*)new BipartiteGraphAnnealer<float>();
    else
        RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    
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
        RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *bg_annealer_seed(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    unsigned long long seed;
    if (!PyArg_ParseTuple(args, "OKO", &objExt, &seed, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->seed(seed);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->seed(seed);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);
    
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
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setProblem(b0, b1, W, om);
}
    
extern "C"
PyObject *bg_annealer_set_problem(PyObject *module, PyObject *args) {
    PyObject *objExt, *objB0, *objB1, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOOOiO", &objExt, &objB0, &objB1, &objW, &opt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_bg_annealer_set_problem<double>(objExt, objB0, objB1, objW, opt);
        else if (isFloat32(dtype))
            internal_bg_annealer_set_problem<float>(objExt, objB0, objB1, objW, opt);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
    
extern "C"
PyObject *bg_annealer_get_problem_size(PyObject *module, PyObject *args) {

    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    sqaod::SizeType N0, N1;
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->getProblemSize(&N0, &N1);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->getProblemSize(&N0, &N1);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    return Py_BuildValue("II", N0, N1);
}
    
extern "C"
PyObject *bg_annealer_set_preferences(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype, *objPrefs;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objPrefs, &dtype))
        return NULL;

    sq::Preferences prefs;
    if (parsePreferences(objPrefs, &prefs, Cpu_BgAnnealerError) == -1)
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->setPreferences(prefs);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->setPreferences(prefs);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *bg_annealer_get_preferences(PyObject *module, PyObject *args) {
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
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    return createPreferences(prefs);    
}

template<class real>
PyObject *internal_bg_annealer_get_x(PyObject *objExt) {
    BipartiteGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N0, N1;
    ann->getProblemSize(&N0, &N1);
    const sq::BitsPairArray &xPairList = ann->get_x();

    PyObject *list = PyList_New(xPairList.size());
    for (size_t idx = 0; idx < xPairList.size(); ++idx) {
        const sq::BitsPairArray::ValueType &pair = xPairList[idx];

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

    TRY {
        if (isFloat64(dtype))
            return internal_bg_annealer_get_x<double>(objExt);
        else if (isFloat32(dtype))
            return internal_bg_annealer_get_x<float>(objExt);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
}


template<class real>
void internal_bg_annealer_set_x(PyObject *objExt, PyObject *objX0, PyObject *objX1) {
    NpBitVector x0(objX0), x1(objX1);
    pyobjToCppObj<real>(objExt)->set_x(x0, x1);
}

extern "C"
PyObject *bg_annealer_set_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *objX0, *objX1, *dtype;
    
    if (!PyArg_ParseTuple(args, "OOOO", &objExt, &objX0, &objX1, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_bg_annealer_set_x<double>(objExt, objX0, objX1);
        else if (isFloat32(dtype))
            internal_bg_annealer_set_x<float>(objExt, objX0, objX1);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    

template<class real>
PyObject *internal_bg_annealer_get_q(PyObject *objExt) {
    BipartiteGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N0, N1;
    ann->getProblemSize(&N0, &N1);
    const sq::BitsPairArray &xPairList = ann->get_q();

    PyObject *list = PyList_New(xPairList.size());
    for (size_t idx = 0; idx < xPairList.size(); ++idx) {
        const sq::BitsPairArray::ValueType &pair = xPairList[idx];

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

    TRY {
        if (isFloat64(dtype))
            return internal_bg_annealer_get_q<double>(objExt);
        else if (isFloat32(dtype))
            return internal_bg_annealer_get_q<float>(objExt);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
}
    
extern "C"
PyObject *bg_annealer_radomize_q(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->randomize_q();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->randomize_q();
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_bg_annealer_get_hJc(PyObject *objExt,
                                  PyObject *objH0, PyObject *objH1,
                                  PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpScalarRefType<real> NpScalarRef;
    
    BipartiteGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);
    NpVector h0(objH0), h1(objH1);
    NpScalarRef c(objC);
    NpMatrix J(objJ);
    ann->get_hJc(&h0, &h0, &J, &c);
}
    
    
extern "C"
PyObject *bg_annealer_get_hJc(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH0, *objH1, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOOO", &objExt, &objH0, &objH1, &objJ, &objC, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_bg_annealer_get_hJc<double>(objExt, objH0, objH1, objJ, objC);
        else if (isFloat32(dtype))
            internal_bg_annealer_get_hJc<float>(objExt, objH0, objH1, objJ, objC);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
PyObject *internal_bg_annealer_get_E(PyObject *objExt, int typenum) {
    typedef NpVectorType<real> NpVector;
    const sqaod::VectorType<real> &E = pyobjToCppObj<real>(objExt)->get_E();
    NpVector npE(E.size, typenum); /* allocate PyObject */
    npE.vec = E;
    return npE.obj;
}

    
extern "C"
PyObject *bg_annealer_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_bg_annealer_get_E<double>(objExt, NPY_FLOAT64);
        else if (isFloat32(dtype))
            return internal_bg_annealer_get_E<float>(objExt, NPY_FLOAT32);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
}

    
extern "C"
PyObject *bg_annealer_calculate_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->calculate_E();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->calculate_E();
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *bg_annealer_prepare(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->prepare();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->prepare();
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
template<class real>
void internal_bg_annealer_anneal_one_step(PyObject *objExt, PyObject *objG, PyObject *objKT) {
    typedef NpConstScalarType<real> NpConstScalar;
    NpConstScalar G(objG), kT(objKT);
    pyobjToCppObj<real>(objExt)->annealOneStep(G, kT);
}


extern "C"
PyObject *bg_annealer_anneal_one_step(PyObject *module, PyObject *args) {
    PyObject *objExt, *objG, *objKT, *dtype;
    if (!PyArg_ParseTuple(args, "OOOO", &objExt, &objG, &objKT, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_bg_annealer_anneal_one_step<double>(objExt, objG, objKT);
        else if (isFloat32(dtype))
            internal_bg_annealer_anneal_one_step<float>(objExt, objG, objKT);
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *bg_annealer_make_solution(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->makeSolution();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->makeSolution();
        else
            RAISE_INVALID_DTYPE(dtype, Cpu_BgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cpu_BgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}

}


static
PyMethodDef cpu_bg_annealer_methods[] = {
	{"new_annealer", bg_annealer_create, METH_VARARGS},
	{"delete_annealer", bg_annealer_delete, METH_VARARGS},
	{"seed", bg_annealer_seed, METH_VARARGS},
	{"set_problem", bg_annealer_set_problem, METH_VARARGS},
	{"get_problem_size", bg_annealer_get_problem_size, METH_VARARGS},
	{"set_preferences", bg_annealer_set_preferences, METH_VARARGS},
	{"get_preferences", bg_annealer_get_preferences, METH_VARARGS},
	{"get_E", bg_annealer_get_E, METH_VARARGS},
	{"get_x", bg_annealer_get_x, METH_VARARGS},
	{"set_x", bg_annealer_set_x, METH_VARARGS},
	{"get_hJc", bg_annealer_get_hJc, METH_VARARGS},
	{"get_q", bg_annealer_get_q, METH_VARARGS},
	{"randomize_q", bg_annealer_radomize_q, METH_VARARGS},
	{"calculate_E", bg_annealer_calculate_E, METH_VARARGS},
	{"prepare", bg_annealer_prepare, METH_VARARGS},
	{"make_solution", bg_annealer_make_solution, METH_VARARGS},
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
    
    char name[] = "cpu_dg_annealer.error";
    Cpu_BgAnnealerError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cpu_BgAnnealerError);
    PyModule_AddObject(m, "error", Cpu_BgAnnealerError);
}
