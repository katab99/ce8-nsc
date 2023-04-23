import unittest
import mandulakenyer as mk
import numpy as np
# https://www.blog.pythonlibrary.org/2016/07/20/an-intro-to-coverage-py/
class TestMandelbrot(unittest.TestCase):

    # TODO : need at least 3 test func

    # Test that the output values are symmetric about the y-axis
    def test_func1(self):
        re = [-2, 0.5, 600]
        im = [-1, 1, 400]
        itr = 100
        thresh = 2

        M = mk.naive(re, im, itr, thresh)
        
        self.assertTrue(np.all(np.fliplr(M) == M))
        self.assertEqual(M.shape, (600, 400))

    def test_func2(self):
        re = [-2, 0.5, 1000]
        im = [-1, 1, 1000]
        itr = 100
        thresh = 2

        M = mk.vectorized(re, im, itr, thresh)
        self.assertTrue(np.all(np.fliplr(M) == M))
    #def test_func3(self):
    
    
    # Test that the output values are symmetric about the x-axis
    # Test that the output values are symmetric about the y-axis
    # Test that the output values are within the expected range

if __name__ == '__main__':
    unittest.main()