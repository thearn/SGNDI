from numpy.testing import run_module_suite, assert_allclose
import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline
from sgndi import SeparableGridNDInterpolator

class TestSGNDIbase(object):

    def check_output_and_gradient(self, x):
        interp = SeparableGridNDInterpolator(self.points, self.values, 
                                    interpolator = self.interpolator)
        f, dfdx = interp(x)

        assert_allclose(self.F(*x), f, rtol = self.tol)
        assert_allclose(self.dF(*x), dfdx, rtol=self.tol)

    def test_values(self):
        if not hasattr(self, 'setup'):
            return

        self.setup()
    
        samples = np.empty((self.m, len(self.points)))
        for i, pt in enumerate(self.points):
            np.random.seed(42)
            samples[:, i] = np.random.uniform(pt[1], pt[-2], self.m)

        for x in samples:
            yield self.check_output_and_gradient, x

class TestSGNDIparabola(TestSGNDIbase):
    tol = 1e-5
    m = 4
    interpolator = CubicSpline

    def F(self, u,v,z,w):
        return (u-5)**2 + (v-2)**2 + (z-5)**2 + (w-0.5)**2

    def dF(self, u,v,z,w):
        return 2*(u-5), 2*(v-2), 2*(z-5), 2*(w-0.5)

    def setup(self):
        U = np.linspace(0, 10, 10)
        V = np.linspace(0, 4, 6)
        Z = np.linspace(0, 10, 7) 
        W = np.linspace(0, 1, 8)

        self.points = [U, V, Z, W]

        u, v, z, w = np.meshgrid(*self.points, indexing='ij')

        self.values = self.F(u, v, z, w)

class TestSGNDItrig(TestSGNDIbase):
    tol = 5e-2
    m = 6
    interpolator = Akima1DInterpolator

    def F(self, u, v):
        return u*np.cos(u*v) + v*np.sin(u*v)

    def dF(self, u, v):
        return -u*v*np.sin(u*v) + v**2*np.cos(u*v) + np.cos(u*v), -u**2*np.sin(u*v) + u*v*np.cos(u*v) + np.sin(u*v)

    def setup(self):
        U = np.linspace(0, 2, 50)
        V = np.linspace(0, 2, 50)

        self.points = [U, V]

        u, v = np.meshgrid(*self.points, indexing='ij')

        self.values = self.F(u, v)

if __name__ == '__main__':
    run_module_suite()