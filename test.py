from numpy.testing import run_module_suite, assert_allclose
import numpy as np
from sgdni import SeparableGridNDInterpolator

class TestSGNDIparabola(object):

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

    def check_value(self, x):
        interp = SeparableGridNDInterpolator(self.points, self.values)
        f, dfdx = interp(x)

        assert_allclose(self.F(*x), f, rtol = 1e-5)
        assert_allclose(self.dF(*x), dfdx, rtol=1e-5)

    def test_values(self):
        for x in [[5.26434, 2.121235, 2.7352, 0.5213345]]:
            yield self.check_value, x

if __name__ == '__main__':
    run_module_suite()