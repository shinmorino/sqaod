from __future__ import print_function
import sys

control='''
Source: sqaod
Priority: optional
Maintainer: Shinya Morino <shin.morino@gmail.com>
Build-Depends: debhelper (>=9),autotools-dev
Standards-Version: 3.9.6
Section: libs
Homepage: https://github.com/shinmorino/sqaod
#Vcs-Git: git://anonscm.debian.org/collab-maint/sqaod.git
#Vcs-Browser: https://anonscm.debian.org/cgit/collab-maint/sqaod.git
'''

control_libsqaodc='''
Package: libsqaodc{package_postfix}
Provides: libsqaodc
Section: libs
Architecture: amd64
Depends:  libgomp1:amd64, libblas3:amd64, ${{shlibs:Depends}}, ${{misc:Depends}}
Description: sqaodc library
'''

control_libsqaodc_cuda='''
Package: libsqaodc-cuda-{cudaver}
Provides: libsqaodc-cuda
Section: libs
Architecture: amd64
Depends: libsqaodc:amd64, cuda-cublas-{cudaver}:amd64, cuda-cudart-{cudaver}:amd64, cuda-curand-{cudaver}:amd64, ${{shlibs:Depends}}, ${{misc:Depends}}
Description: sqaodc CUDA library
'''

libsqaodc_install='usr/lib/*/libsqaodc.so.*'
libsqaodc_cuda_install='usr/lib/*/libsqaodc_cuda.so.*'


simd = sys.argv[1]
package_postfix = '' if simd == 'sse2' else simd
if len(sys.argv) >= 3:
    cudaver = sys.argv[2]
    
if simd == 'sse2' :
    with open('control', 'w') as file:
        file.write(control)
        file.write(control_libsqaodc.format(simd=simd, package_postfix=package_postfix))
        file.write(control_libsqaodc_cuda.format(simd=simd, package_postfix=package_postfix,
                                                cudaver=cudaver))
    with open('libsqaodc.install', 'w') as file:
        file.write(libsqaodc_install)
    with open('libsqaodc-cuda.install', 'w') as file:
        file.write(libsqaodc_cuda_install)
else :
    with open('control', 'w') as file:
        file.write(control)
        file.write(control_libsqaodc.format(simd=simd, package_postfix=package_postfix))
    with open('libsqaodc_{simd}.install'.format(simd=simd), 'w') as file:
        file.write(libsqaodc_install)
    
