from __future__ import print_function
from __future__ import absolute_import

import sqaod
import unittest

class TestWrongType(unittest.TestCase) :

    def test_exception(self) :
        pkg_version = sqaod.__version__
        import pkg_resources
        rsc_version =  pkg_resources.get_distribution("sqaod").version
        self.assertEqual(pkg_version, rsc_version)
        

if __name__ == '__main__':
    unittest.main()
