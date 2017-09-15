from nose.tools import raises

import numpy as np

from numpy.testing import assert_allclose, run_module_suite

from scipy.interpolate import (Akima1DInterpolator,
                               BarycentricInterpolator,
                               CubicSpline,
                               UnivariateSpline)

from sgndi import SeparableGridNDInterpolator


class TestSGNDIoutput(object):
    interpolator = CubicSpline
    interp_kwargs = {}
    tol = 1e-5
    m = 10

    def check_output_and_cached_gradient(self, x):
        interp = SeparableGridNDInterpolator(self.points, self.values,
                                             interpolator=self.interpolator,
                                             interp_kwargs=self.interp_kwargs)
        f = interp(x)
        dfdx = interp.derivative(x)

        assert_allclose(self.F(*x), f, rtol=self.tol)
        assert_allclose(self.dF(*x), dfdx, rtol=self.tol)

    def check_recomputed_gradient(self, x, y):
        interp = SeparableGridNDInterpolator(self.points, self.values,
                                             interpolator=self.interpolator,
                                             interp_kwargs=self.interp_kwargs)
        interp.derivative(y)
        dfdx = interp.derivative(x)
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
            yield self.check_output_and_cached_gradient, x

        y = samples[0]
        for x in samples[1:2]:
            yield self.check_recomputed_gradient, x, y


class TestSGNDIparabola(TestSGNDIoutput):
    m = 4

    def F(self, u, v, z, w):
        return (u - 5)**2 + (v - 2)**2 + (z - 5)**2 + (w - 0.5)**2

    def dF(self, u, v, z, w):
        return 2 * (u - 5), 2 * (v - 2), 2 * (z - 5), 2 * (w - 0.5)

    def setup(self):
        U = np.linspace(0, 10, 10)
        V = np.linspace(0, 4, 6)
        Z = np.linspace(0, 10, 7)
        W = np.linspace(0, 1, 8)
        self.points = [U, V, Z, W]
        u, v, z, w = np.meshgrid(*self.points, indexing='ij')
        self.values = self.F(u, v, z, w)


class TestSGNDItrig(TestSGNDIoutput):
    tol = 5e-2
    m = 6
    interpolator = Akima1DInterpolator

    def F(self, u, v):
        return u * np.cos(u * v) + v * np.sin(u * v)

    def dF(self, u, v):
        return (-u * v * np.sin(u * v) + v**2 * np.cos(u * v) + np.cos(u * v),
                -u**2 * np.sin(u * v) + u * v * np.cos(u * v) + np.sin(u * v))

    def setup(self):
        U = np.linspace(0, 2, 50)
        V = np.linspace(0, 2, 50)
        self.points = [U, V]
        u, v = np.meshgrid(*self.points, indexing='ij')
        self.values = self.F(u, v)


class TestLinearInterp(TestSGNDIoutput):
    tol = 1e-12
    interpolator = UnivariateSpline
    m = 5
    interp_kwargs = {'k': 1}

    def F(self, x, y, z):
        return 2 * x - y + z

    def dF(self, x, y, z):
        return 2, -1, 1

    def setup(self):
        X = np.linspace(-1, 2, 50)
        Y = np.linspace(-10, 0, 10)
        Z = np.linspace(3, 9, 20)
        self.points = [X, Y, Z]
        x, y, z = np.meshgrid(*self.points, indexing='ij')
        self.values = self.F(x, y, z)


class TestSGDNIExceptions(object):

    @raises(ValueError)
    def test_dim_mismatch(self):
        a = np.array([0, 1, 6])
        b = np.array([0, 1, 2])
        c = np.array([[0, 1], [1, 0], [5, 4]])
        cs = SeparableGridNDInterpolator([a, b], c)

    @raises(ValueError)
    def test_max_deriv_order(self):
        np.random.seed(0)

        a = np.random.uniform(0, 1, 10)
        b = np.random.uniform(-10, 10, 20)
        a.sort(), b.sort()
        A, B = np.meshgrid(a, b, indexing='ij')
        c = A + B**2
        cs = SeparableGridNDInterpolator([a, b], c)
        cs([2, 4], 2)

    @raises(ValueError)
    def test_deriv_available(self):
        np.random.seed(2)
        a = np.random.uniform(-1, 4, 10)
        b = np.random.uniform(-1, 15, 20)
        a.sort(), b.sort()
        A, B = np.meshgrid(a, b, indexing='ij')
        c = np.sin(A - B)
        cs = SeparableGridNDInterpolator([a, b], c,
                                         interpolator=BarycentricInterpolator)
        cs([2, 4])

if __name__ == '__main__':
    run_module_suite()
