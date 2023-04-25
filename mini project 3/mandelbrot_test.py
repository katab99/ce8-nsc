import unittest
import numpy as np
import time 
import mandulakenyer as mk

class TestMandelbrot(unittest.TestCase):
    # Set up test data
    def setUp(self):
        self.re_scale = 500
        self.im_scale = 500
        
        self.re = np.linspace(-2, 0.5, self.re_scale, dtype=np.float32)
        self.im = np.linspace(-1, 1, self.im_scale, dtype=np.float32)
        
        self.max_iter = 100
        self.thresh = 2
    
    # Test if output shape is the same as the input - naive
    def test_naive_shape(self):
        M = mk.naive(self.re, self.im, self.max_iter, self.thresh)
        self.assertEqual(M.shape, (self.re_scale, self.im_scale))

    # Test if output shape is the same as the input - vectorized
    def test_vectorized_shape(self):
        M = mk.vectorized(self.re, self.im, self.max_iter, self.thresh)
        self.assertEqual(M.shape, (self.re_scale, self.im_scale))
    
    # Test runtime - naive
    def test_naive_run_time(self):
        t0 = time.time()
        M = mk.naive(self.re, self.im, self.max_iter, self.thresh)
        t1 = time.time() - t0
        self.assertLess(t1, 30)

    # Test runtime - vectorized
    def test_vectorized_rum_time(self):
        t0 = time.time()
        M = mk.vectorized(self.re, self.im, self.max_iter, self.thresh)
        t1 = time.time() - t0
        self.assertLess(t1, 30)

    # Test if solution is in range [0, max_iter] - naive
    def test_naive_range(self):
        M = mk.naive(self.re, self.im, self.max_iter, self.thresh)
        self.assertGreaterEqual(M.min(), 0)
        self.assertLessEqual(M.max(), self.max_iter)

    # Test if solution is in range [0, max_iter] - vectorized
    def test_vectorized_range(self):
        M = mk.vectorized(self.re, self.im, self.max_iter, self.thresh)
        self.assertGreaterEqual(M.min(), 0)
        self.assertLessEqual(M.max(), self.max_iter)

if __name__ == '__main__':
    unittest.main()