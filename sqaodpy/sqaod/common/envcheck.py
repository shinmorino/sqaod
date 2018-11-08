from __future__ import print_function
from __future__ import absolute_import
from .cuda_probe import is_cuda_available, cuda_failure_reason 
from ctypes.util import find_library
from ctypes import CDLL, c_int, c_char_p, POINTER, byref
from .version import sqaodc_version, libsqaodc_name, libsqaodc_cuda_name

def load_shared_library(libname) :
    try :
        so = CDLL(libname)
        return so
    except :
        return None

def sqaodc_version_getter(so) :
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

def sqaodc_cuda_version_getter(so) :
    funcname = 'sqaodc_cuda_version'
    if not hasattr(so, funcname) :
        return (None, None)
    func = getattr(so, funcname)
    func.restype = None
    func.argtype = [POINTER(c_int), POINTER(c_int)]
    ver = c_int(0)
    cuda_ver = c_int(0)
    func(byref(ver), byref(cuda_ver))
    return (ver.value, cuda_ver.value)


def check_libsqaodc() :
    results = {}
    libsqaodc = load_shared_library(libsqaodc_name)
    if libsqaodc is None :
        return results
    
    results['installed'] = True
    ver = sqaodc_version_getter(libsqaodc)
    results['ver'] = ver[0]
    results['simd'] = ver[1]
    return results

def check_libsqaodc_cuda() :
    results = {}
    libsqaodc_cuda = load_shared_library(libsqaodc_cuda_name)
    if libsqaodc_cuda is None :
        return results
    
    results['installed'] = True
    ver = sqaodc_cuda_version_getter(libsqaodc_cuda)
    results['ver'] = ver[0]
    results['cuda'] = ver[1]

    return results

def show_libraries() :
    libsqaodc = check_libsqaodc()
    libsqaodc_cuda = check_libsqaodc_cuda()
    print(libsqaodc)
    print(libsqaodc_cuda)

def get_string_version(numver) :
    return '{}.{}.{}'.format(numver // 10000, (numver // 100) % 100, numver % 100)

def get_cuda_string_version(numver) :
    return '{}.{}.{}'.format(numver // 1000, (numver // 10) % 100, numver % 10)

class EnvChecker :
    def is_installed(self, results) :
        return 'installed' in results.keys()
    
    def check_ver(self, expected_ver, results) :
        ver = results['ver']
        return expected_ver <= ver
    
    def check(self) :
        self.sqaodc = check_libsqaodc()
        self.sqaodc_cuda = check_libsqaodc_cuda()

        if not is_cuda_available() :
            # CUDA is not avilable, check CPU lib.
            if not self.is_installed(self.sqaodc) :
                return False
            return self.check_ver(sqaodc_version, self.sqaodc)

        # CUDA available, check both CPU and CUDA libs.
        if not self.is_installed(self.sqaodc) or not self.is_installed(self.sqaodc_cuda) :
            return False
        return self.check_ver(sqaodc_version, self.sqaodc) and \
            self.check_ver(sqaodc_version, self.sqaodc_cuda)

    def show_cpu(self) :
        if not self.is_installed(self.sqaodc) :
            # libsqaodc not found.
            print(' [NG] {} : not found'.format(libsqaodc_name))
            return
        ver = self.sqaodc['ver']
        strver = get_string_version(ver)
        if not self.check_ver(sqaodc_version, self.sqaodc) :
            print(' [NG] {}, {} : too old, update to the latest version.'.format(libsqaodc_name, strver))
            return
        simd = self.sqaodc['simd'].decode('ASCII')
        print(' [OK] {}, {}, {}'.format(libsqaodc_name, strver, simd))

    def show_cuda(self) :
        if not self.is_installed(self.sqaodc_cuda) :
            # libsqaodc_cuda not found.
            print(' [NG] {} : not found'.format(libsqaodc_cuda_name))
            return
        ver = self.sqaodc_cuda['ver']
        strver = get_string_version(ver)
        if not self.check_ver(sqaodc_version, self.sqaodc_cuda) :
            print(' [NG] {}, {} : too old, update to the latest version.'.format(libsqaodc_cuda_name, strver))
            return
        cudaver = self.sqaodc_cuda['cuda']
        strcudaver = get_cuda_string_version(cudaver)
        print(' [OK] {}, {}, CUDA {}'.format(libsqaodc_cuda_name, strver, strcudaver))
        
    def show(self) :
        # libsqaodc
        print('CPU backend')
        self.show_cpu()
        print('CUDA backend')
        self.show_cuda()
        if not is_cuda_available() :
            print(" GPU not available.")
            return
        
if __name__ == '__main__' :
    show_libraries()
    checker = EnvChecker()
    checker.check()
    checker.show()
