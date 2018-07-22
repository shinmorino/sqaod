from __future__ import print_function
from __future__ import absolute_import
from .cuda_probe import is_cuda_available, cuda_failure_reason 
from ctypes.util import find_library
from ctypes import CDLL, c_int, c_char_p, POINTER, byref

    
def _check_shared_object(libname, version_getter) :
    try :
        so = CDLL(libname)
    except :
        return (False, libname, None)
    ver = version_getter(so)
    return (True, libname, ver)


def _sqaodc_version_getter(so) :
    funcname = 'sqaodc_version'
    if not hasattr(so, funcname) :
        return None
    func = getattr(so, funcname)
    func.restype = None
    func.argtype = [POINTER(c_int), POINTER(c_char_p)]
    ver = c_int(0)
    simd = c_char_p(0)
    func(byref(ver), byref(simd))
    return (ver.value, simd.value)

def check_libsqaodc() :
    return _check_shared_object('libsqaodc.so.0', _sqaodc_version_getter)

def _sqaodc_cuda_version_getter(so) :
    funcname = 'sqaodc_cuda_version'
    if not hasattr(so, funcname) :
        return None
    func = getattr(so, funcname)
    func.restype = None
    func.argtype = [POINTER(c_int), POINTER(c_int)]
    ver = c_int(0)
    cuda_ver = c_int(0)
    func(byref(ver), byref(cuda_ver))
    return (ver.value, cuda_ver.value)

def check_libsqaodc_cuda() :
    return _check_shared_object('libsqaodc_cuda.so.0', _sqaodc_cuda_version_getter)


def get_string_version(numver) :
    return '{}.{}.{}'.format(numver / 10000, (numver / 100) % 100, numver % 100)

def check_version(version, pkgver) :
    if version is None :
        return False
    strver = get_string_version(version[0])
    return strver == pkgver

def show_libraries() :
    libsqaodc = check_libsqaodc()
    libsqaodc_cuda = check_libsqaodc_cuda()
    print(libsqaodc)
    print(libsqaodc_cuda)


    

class EnvChecker :
    def __init__(self, pkgver) :
        self.pkgver = pkgver
    
    def check(self) :
        self.sqaodc = check_libsqaodc()
        self.sqaodc_cuda = check_libsqaodc_cuda()
        if not self.sqaodc[0] or not self.sqaodc_cuda[0] :
            return False
        return check_version(self.sqaodc[2], self.pkgver) and check_version(self.sqaodc_cuda[2], self.pkgver)
    
    def _format_message(self, lib) :
        success, libname, version = lib
        verok = False
        
        if not success :
            if libname is not None :
                err_reason = 'load failed'
            else :
                err_reason = 'library not found'
            verstr = 'N/A'
        else :
            verok = check_version(version, self.pkgver)
            verstr = get_string_version(version[0]) if verok else '< 0.3.1'
            if not verok :
                err_reason = 'version mismatch'

        if success and verok :
            return 'OK {} ({})'.format(libname, verstr)
        else :
            return 'NG {} ({}) {}'.format(libname, verstr, err_reason)

    def show(self) :
        msg = self._format_message(self.sqaodc)
        print(msg)
        msg = self._format_message(self.sqaodc_cuda)
        print(msg)
    
        #print('CUDA                : {}'.format('Available' if cuda_available else 'not available'))
        #cuda_available = is_cuda_available()
    
if __name__ == '__main__' :
    show_libraries()
    checker = EnvChecker('0.3.1')
    checker.check()
    checker.show()
