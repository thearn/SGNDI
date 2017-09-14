import numpy as np
from separable_grid_nd_interpolator import SeparableGridNDInterpolator
from scipy.interpolate import CubicSpline, UnivariateSpline, Akima1DInterpolator


def F(u,v,z,w):
    return (u-5)**2 + (v-2)**2 + (z-5)**2 + (w-0.5)**2

def dF(u,v,z,w):
    return 2*(u-5), 2*(v-2), 2*(z-5), 2*(w-0.5)

U = np.linspace(0, 10, 10)
V = np.linspace(0, 4, 6)
Z = np.linspace(0, 10, 7) 
W = np.linspace(0, 1, 8)

points = [U, V, Z, W]

u, v, z, w = np.meshgrid(*points, indexing='ij')

values = F(u, v, z, w)

x = [5.26434, 2.121235, 2.7352, 0.5213345]

interp = SeparableGridNDInterpolator(points, values)
f, dfdx = interp(x)

print("actual value", F(*x))
print("computed value", f)

print("actual gradient:", dF(*x))
print("computed gradient:", dfdx)