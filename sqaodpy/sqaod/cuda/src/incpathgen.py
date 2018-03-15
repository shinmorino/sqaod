from __future__ import print_function
import sysconfig
import numpy

if __name__ == '__main__' :
    include = sysconfig.get_config_var('INCLUDEPY')
    npinclude = numpy.get_include()

    print(''.join(( 'INCLUDE=-I', include, ' -I', npinclude)))
