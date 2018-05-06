#include <sqaodc/pyglue/pyglue.h>

namespace {

extern "C"
PyObject *cuda_device_new(PyObject *module, PyObject *args) {
    sq::cuda::Device *device;
    TRY {
        device = sq::cuda::newDevice();
    } CATCH_ERROR_AND_RETURN;

    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)device);
    return obj;
}


extern "C"
PyObject *cuda_device_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    sq::cuda::Device *device = (sq::cuda::Device*)PyArrayScalar_VAL(objExt, UInt64);
    deleteInstance(device);
    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *cuda_device_initialize(PyObject *module, PyObject *args) {
    PyObject *objExt;
    int devNo;
    if (!PyArg_ParseTuple(args, "Oi", &objExt, &devNo))
        return NULL;
    sq::cuda::Device *device = (sq::cuda::Device*)PyArrayScalar_VAL(objExt, UInt64);
    TRY {
        device->initialize(devNo);
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;
}


extern "C"
PyObject *cuda_device_finalize(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    sq::cuda::Device *device = (sq::cuda::Device*)PyArrayScalar_VAL(objExt, UInt64);
    TRY {
        device->finalize();
    } CATCH_ERROR_AND_RETURN;

    Py_INCREF(Py_None);
    return Py_None;
}

}




static
PyMethodDef cuda_device_methods[] = {
    {"new", cuda_device_new, METH_VARARGS},
    {"delete", cuda_device_delete, METH_VARARGS},
    {"initialize", cuda_device_initialize, METH_VARARGS},
    {"finalize", cuda_device_finalize, METH_VARARGS},
    {NULL},
};



#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "cuda_device",
        NULL, 0,
        cuda_device_methods,
        NULL, NULL, NULL, NULL
};

extern "C"
PyMODINIT_FUNC PyInit_cuda_device(void) {
    PyObject *module = PyModule_Create(&moduledef);
    if (module == NULL)
        return NULL;
    import_array();
    return module;
}

#else

PyMODINIT_FUNC initcuda_device(void) {
    PyObject *module = Py_InitModule("cuda_device", cuda_device_methods);
    if (module == NULL)
        return;
    import_array();
}

#endif
