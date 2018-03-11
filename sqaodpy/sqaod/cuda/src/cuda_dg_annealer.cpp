#include <sqaodc/pyglue/pyglue.h>
#include <sqaodc/sqaodc.h>


static PyObject *Cuda_DgAnnealerError;
namespace sq = sqaod;
namespace sqcu = sqaod::cuda;

template<class real>
using DenseGraphAnnealer = sq::cuda::DenseGraphAnnealer<real>;


namespace {
    
template<class real>
DenseGraphAnnealer<real> *pyobjToCppObj(PyObject *obj) {
    npy_uint64 val = PyArrayScalar_VAL(obj, UInt64);
    return reinterpret_cast<DenseGraphAnnealer<real> *>(val);
}

extern "C"
PyObject *dg_annealer_create(PyObject *module, PyObject *args) {
    PyObject *dtype;
    void *ext;
    if (!PyArg_ParseTuple(args, "O", &dtype))
        return NULL;
    if (isFloat64(dtype))
        ext = (void*)new DenseGraphAnnealer<double>();
    else if (isFloat32(dtype))
        ext = (void*)new DenseGraphAnnealer<float>();
    else
        RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    
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
        RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_annealer_assign_device(PyObject *module, PyObject *args) {
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
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

extern "C"
PyObject *dg_annealer_seed(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    unsigned int seed;
    if (!PyArg_ParseTuple(args, "OIO", &objExt, &seed, &dtype))
        return NULL;
    if (isFloat64(dtype))
        pyobjToCppObj<double>(objExt)->seed(seed);
    else if (isFloat32(dtype))
        pyobjToCppObj<float>(objExt)->seed(seed);
    else
        RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    
    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
void internal_dg_annealer_set_qubo(PyObject *objExt, PyObject *objW, int opt) {
    typedef NpMatrixType<real> NpMatrix;
    NpMatrix W(objW);
    sq::OptimizeMethod om = (opt == 0) ? sq::optMinimize : sq::optMaximize;
    pyobjToCppObj<real>(objExt)->setQUBO(W, om);
}
    
extern "C"
PyObject *dg_annealer_set_qubo(PyObject *module, PyObject *args) {
    PyObject *objExt, *objW, *dtype;
    int opt;
    if (!PyArg_ParseTuple(args, "OOiO", &objExt, &objW, &opt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_set_qubo<double>(objExt, objW, opt);
        else if (isFloat32(dtype))
            internal_dg_annealer_set_qubo<float>(objExt, objW, opt);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}

template<class real>
void internal_dg_annealer_set_hamiltonian(PyObject *objExt,
                                          PyObject *objH, PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpConstScalarType<real> NpConstScalar;

    NpVector h(objH);
    NpMatrix J(objJ);
    NpConstScalar c(objC);
    pyobjToCppObj<real>(objExt)->setHamiltonian(h, J, c);
}
    
extern "C"
PyObject *dg_annealer_set_hamiltonian(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objH, &objJ, &objC, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_set_hamiltonian<double>(objExt, objH, objJ, objC);
        else if (isFloat32(dtype))
            internal_dg_annealer_set_hamiltonian<float>(objExt, objH, objJ, objC);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_annealer_get_problem_size(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;
    sqaod::SizeType N;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->getProblemSize(&N);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->getProblemSize(&N);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    return Py_BuildValue("I", N);
}
    
extern "C"
PyObject *dg_annealer_set_preferences(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype, *objPrefs;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objPrefs, &dtype))
        return NULL;

    sq::Preferences prefs;
    if (parsePreferences(objPrefs, &prefs, Cuda_DgAnnealerError) == -1)
        return NULL;
    
    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->setPreferences(prefs);
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->setPreferences(prefs);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_annealer_get_preferences(PyObject *module, PyObject *args) {
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
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    return createPreferences(prefs);    
}


template<class real>
PyObject *internal_dg_annealer_get_E(PyObject *objExt, int typenum) {
    typedef NpVectorType<real> NpVector;
    const sqaod::VectorType<real> &E = pyobjToCppObj<real>(objExt)->get_E();
    NpVector npE(E.size, typenum); /* allocate PyObject */
    npE.vec = E;
    return npE.obj;
}

extern "C"
PyObject *dg_annealer_get_E(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            return internal_dg_annealer_get_E<double>(objExt, NPY_FLOAT64);
        else if (isFloat32(dtype))
            return internal_dg_annealer_get_E<float>(objExt, NPY_FLOAT32);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
}

template<class real>
PyObject *internal_dg_annealer_get_x(PyObject *objExt) {
    DenseGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N;
    ann->getProblemSize(&N);
    const sqaod::BitSetArray &xList = ann->get_x();
    PyObject *list = PyList_New(xList.size());
    for (sq::IdxType idx = 0; idx < xList.size(); ++idx) {
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
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
}


template<class real>
void internal_dg_annealer_set_x(PyObject *objExt, PyObject *objX) {
    NpBitVector x(objX);
    pyobjToCppObj<real>(objExt)->set_x(x);
}

extern "C"
PyObject *dg_annealer_set_x(PyObject *module, PyObject *args) {
    PyObject *objExt, *objX, *dtype;
    if (!PyArg_ParseTuple(args, "OOO", &objExt, &objX, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_set_x<double>(objExt, objX);
        else if (isFloat32(dtype))
            internal_dg_annealer_set_x<float>(objExt, objX);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}


template<class real>
void internal_dg_annealer_get_hamiltonian(PyObject *objExt,
                                          PyObject *objH, PyObject *objJ, PyObject *objC) {
    typedef NpMatrixType<real> NpMatrix;
    typedef NpVectorType<real> NpVector;
    typedef NpScalarRefType<real> NpScalarRef;

    NpMatrix J(objJ);
    NpVector h(objH);
    NpScalarRef c(objC);
    
    DenseGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);
    ann->getHamiltonian(&h, &J, &c);
}
    
    
extern "C"
PyObject *dg_annealer_get_hamiltonian(PyObject *module, PyObject *args) {
    PyObject *objExt, *objH, *objJ, *objC, *dtype;
    if (!PyArg_ParseTuple(args, "OOOOO", &objExt, &objH, &objJ, &objC, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            internal_dg_annealer_get_hamiltonian<double>(objExt, objH, objJ, objC);
        else if (isFloat32(dtype))
            internal_dg_annealer_get_hamiltonian<float>(objExt, objH, objJ, objC);
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}

    
template<class real>
PyObject *internal_dg_annealer_get_q(PyObject *objExt) {
    DenseGraphAnnealer<real> *ann = pyobjToCppObj<real>(objExt);

    sqaod::SizeType N;
    ann->getProblemSize(&N);
    const sqaod::BitSetArray &qList = ann->get_q();
    PyObject *list = PyList_New(qList.size());
    for (sq::IdxType idx = 0; idx < qList.size(); ++idx) {
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
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
}
    
extern "C"
PyObject *dg_annealer_randomize_spin(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->randomizeSpin();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->randomizeSpin();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

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
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}

        
extern "C"
PyObject *dg_annealer_prepare(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->prepare();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->prepare();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}
    
extern "C"
PyObject *dg_annealer_make_solution(PyObject *module, PyObject *args) {
    PyObject *objExt, *dtype;
    if (!PyArg_ParseTuple(args, "OO", &objExt, &dtype))
        return NULL;

    TRY {
        if (isFloat64(dtype))
            pyobjToCppObj<double>(objExt)->makeSolution();
        else if (isFloat32(dtype))
            pyobjToCppObj<float>(objExt)->makeSolution();
        else
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

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
            RAISE_INVALID_DTYPE(dtype, Cuda_DgAnnealerError);
    } CATCH_ERROR_AND_RETURN(Cuda_DgAnnealerError);

    Py_INCREF(Py_None);
    return Py_None;    
}

}




static
PyMethodDef cuda_dg_annealer_methods[] = {
	{"new_annealer", dg_annealer_create, METH_VARARGS},
	{"delete_annealer", dg_annealer_delete, METH_VARARGS},
	{"assign_device", dg_annealer_assign_device, METH_VARARGS},
	{"seed", dg_annealer_seed, METH_VARARGS},
	{"set_qubo", dg_annealer_set_qubo, METH_VARARGS},
	{"set_hamiltonian", dg_annealer_set_hamiltonian, METH_VARARGS},
	{"get_problem_size", dg_annealer_get_problem_size, METH_VARARGS},
	{"set_preferences", dg_annealer_set_preferences, METH_VARARGS},
	{"get_preferences", dg_annealer_get_preferences, METH_VARARGS},
	{"get_E", dg_annealer_get_E, METH_VARARGS},
	{"get_x", dg_annealer_get_x, METH_VARARGS},
	{"set_x", dg_annealer_set_x, METH_VARARGS},
	{"get_hamiltonian", dg_annealer_get_hamiltonian, METH_VARARGS},
	{"get_q", dg_annealer_get_q, METH_VARARGS},
	{"randomize_spin", dg_annealer_randomize_spin, METH_VARARGS},
	{"calculate_E", dg_annealer_calculate_E, METH_VARARGS},
	{"prepare", dg_annealer_prepare, METH_VARARGS},
	{"make_solution", dg_annealer_make_solution, METH_VARARGS},
	{"anneal_one_step", dg_annealer_anneal_one_step, METH_VARARGS},
	{NULL},
};



extern "C"
PyMODINIT_FUNC
initcuda_dg_annealer(void) {
    PyObject *m;
    
    m = Py_InitModule("cuda_dg_annealer", cuda_dg_annealer_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cuda_dg_annealer.error";
    Cuda_DgAnnealerError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cuda_DgAnnealerError);
    PyModule_AddObject(m, "error", Cuda_DgAnnealerError);
}
