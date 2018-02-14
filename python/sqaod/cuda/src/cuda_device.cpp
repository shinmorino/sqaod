#include <pyglue.h>
#include <cuda/Device.h>
#include <string.h>

static PyObject *Cuda_DeviceError;
namespace sq = sqaod;
namespace sqcu = sqaod_cuda;


namespace {

extern "C"
PyObject *cuda_device_new(PyObject *module, PyObject *args) {
    int devNo = -1;
    if (!PyArg_ParseTuple(args, "i", &devNo))
        return NULL;
    sqcu::Device *device;
    TRY {
        device = new sqcu::Device(devNo);
    } CATCH_ERROR_AND_RETURN(Cuda_DeviceError);

    PyObject *obj = PyArrayScalar_New(UInt64);
    PyArrayScalar_ASSIGN(obj, UInt64, (npy_uint64)device);
    return obj;
}


extern "C"
PyObject *cuda_device_delete(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    sqcu::Device *device = (sqcu::Device*)PyArrayScalar_VAL(objExt, UInt64);
    delete device;
    Py_INCREF(Py_None);
    return Py_None;    
}


extern "C"
PyObject *cuda_device_initialize(PyObject *module, PyObject *args) {
    PyObject *objExt;
    int devNo;
    if (!PyArg_ParseTuple(args, "Oi", &objExt, &devNo))
        return NULL;
    sqcu::Device *device = (sqcu::Device*)PyArrayScalar_VAL(objExt, UInt64);
    TRY {
        device->initialize(devNo);
    } CATCH_ERROR_AND_RETURN(Cuda_DeviceError);

    Py_INCREF(Py_None);
    return Py_None;    
}


extern "C"
PyObject *cuda_device_finalize(PyObject *module, PyObject *args) {
    PyObject *objExt;
    if (!PyArg_ParseTuple(args, "O", &objExt))
        return NULL;
    sqcu::Device *device = (sqcu::Device*)PyArrayScalar_VAL(objExt, UInt64);
    TRY {
        device->finalize();
        delete device;
    } CATCH_ERROR_AND_RETURN(Cuda_DeviceError);

    Py_INCREF(Py_None);
    return Py_None;    
}

}




static
PyMethodDef cuda_device_methods[] = {
    {"device_new", cuda_device_new, METH_VARARGS},
    {"delete_delete", cuda_device_delete, METH_VARARGS},
    {"device_initialize", cuda_device_initialize, METH_VARARGS},
    {"device_finalize", cuda_device_finalize, METH_VARARGS},
    {NULL},
};

extern "C"
PyMODINIT_FUNC
initcuda_device(void) {
    PyObject *m;
    
    m = Py_InitModule("cuda_device", cuda_device_methods);
    import_array();
    if (m == NULL)
        return;
    
    char name[] = "cuda_device.error";
    Cuda_DeviceError = PyErr_NewException(name, NULL, NULL);
    Py_INCREF(Cuda_DeviceError);
    PyModule_AddObject(m, "error", Cuda_DeviceError);
}
