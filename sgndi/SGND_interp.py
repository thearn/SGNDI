import numpy as np

from scipy.interpolate import CubicSpline


class SeparableGridNDInterpolator(object):
    """
    Provides interpolation on a regular grid in arbitrary dimensions, by
    applying a selected 1D interpolation class on each grid axis sequentially.

    If derivatives are provided by the chosen 1D interpolation method, then
    a gradient vector of the multidimensional interpolation may be computed
    and returned when the interpolation is performed. At the moment, only
    first-order derivatives are supported.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.

    interpolator : A 1D interpolation class such as CubicSpline,
        UnivariateSpline, or Akima1DInterpolator. Defaults to CubicSpline.

    interp_args : An optional tuple of positional arguments to pass to the
        interpolator class when called

    interp_kwargs : An optional dictionary of keyword arguments to pass to the
        interpolator class when called

    Methods
    -------
    __call__
    derivative

    See Also
    --------
    interpn
    RegularGridInterpolator
    RectBivariateSpline

    Examples
    --------

    Let's define a function and its gradient:

    >>> def F(u,v,z,w):
    ...    return (u-5)**2 + (v-2)**2 + (z-5)**2 + (w-0.5)**2
    >>> def dF(u,v,z,w):
    ...    return 2*(u-5), 2*(v-2), 2*(z-5), 2*(w-0.5)

    Now create 1D arrays for each of the function parameters for sampling.
    These are fairly course.

    >>> U = np.linspace(0, 10, 10)
    >>> V = np.linspace(0, 4, 6)
    >>> Z = np.linspace(0, 10, 7)
    >>> W = np.linspace(0, 1, 8)
    >>> points = [U, V, Z, W]

    Create coordinate meshes

    >>> u, v, z, w = np.meshgrid(*points, indexing='ij')

    Now create the 4D value array

    >>> values = F(u, v, z, w)

    Define a random point to interpolate at

    >>> x = [5.26434, 2.121235, 2.7352, 0.5213345]

    Create the interpolation class instance

    >>> interp = SeparableGridNDInterpolator(points, values)

    Call the interpolation at the point above, which by default also
    computes the gradient of the interpolant at this point

    >>> f, dfdx = interp(x)
    >>> print("actual value", F(*x))
    >>> print("computed value", f)
    >>> print("actual gradient:", dF(*x))
    >>> print("computed gradient:", dfdx)
    actual value 5.21434776171525
    computed value 5.214347761715252
    actual gradient: (0.5286799999999996, 0.24246999999999996, -4.5296,
    0.04266900000000007)
    computed gradient: [ 0.52868   0.24247  -4.5296    0.042669]

    """

    def __init__(self, points, values, interpolator=CubicSpline,
                 interp_args=(), interp_kwargs={}):

        dim_valid = [len(points[i]) == values.shape[i]
                     for i in range(len(points))]
        if not np.all(dim_valid):
            msg = "Dimension mismatch between the points" +\
                  " and the data values arrays"
            raise ValueError(msg)

        self.points = points
        self.values = values
        self.interpolator = interpolator
        self.interp_args = interp_args
        self.interp_kwargs = interp_kwargs
        self._x = None
        self._gradient = None

    def __call__(self, x, nu=1):
        """
        Evaluate the interpolation at given positions. If mu=1, the gradient is
        computed together with the interpolation, but is cached and returned
        by the derivative method.

        Parameters
        ----------
        x : array_like
            Input coordinates to evaluate each parameter at.

        nu : int
            Determines whether a gradient is computed or not.
            if nu = 0, no gradient is computed or returned.
            if nu = 1, the first order gradient is computed and returned.

        Returns
        -------
        y : float
            Interpolated value.
        """

        if nu > 0 and 'derivative' not in dir(self.interpolator):
            msg = "Selected 1D Interpolant class must support derivative" +\
                " computations if derivative order nu > 0"
            raise ValueError(msg)

        if nu > 1:
            msg = "Only first derivatives (nu=1) are currently supported"
            raise ValueError(msg)

        self._x = x
        gradient = []
        axis_derivs = []
        values = self.values.copy()

        for i in reversed(range(1, len(self.points))):
            values_reduced = np.zeros(values.size // values.shape[-1])
            newshape = values.shape[: -1]
            local_derivs = []
            values = values.reshape(values.size // values.shape[-1],
                                    values.shape[-1])
            for k, row in enumerate(values):
                local_interp = self.interpolator(self.points[i],
                                                 row,
                                                 *self.interp_args,
                                                 **self.interp_kwargs)
                values_reduced[k] = local_interp(x[i])
                if nu > 0:
                    local_derivs.append(local_interp(x[i], 1))
            values = values_reduced.reshape(newshape)
            if nu > 0:
                local_derivs = np.array(local_derivs).reshape(newshape)
            axis_derivs.append(local_derivs)

        final_interp = self.interpolator(self.points[0], values,
                                         *self.interp_args,
                                         **self.interp_kwargs)
        y = final_interp(x[0])

        if nu < 1:
            return y

        for i in range(len(self.points) - 1):
            deriv_interp = self.__class__(self.points[: -i - 1],
                                          axis_derivs[i],
                                          interpolator=self.interpolator,
                                          interp_args=self.interp_args,
                                          interp_kwargs=self.interp_kwargs)
            g = deriv_interp(x[: -i - 1], nu=nu - 1)
            gradient.insert(0, g)

        gradient.insert(0, final_interp(x[0], 1))
        self._gradient = np.array(gradient)

        return y

    def derivative(self, pt):
        """
        Returns the computed gradients at the specified point.
        The gradients are computed with the interpolation is performed, but
        are cached and returned separately by this method.

        If the point for evaluation differs from the point used to produce
        the currently cached gradient, the interpolation is re-performed in
        order to return the correct gradient.

        Parameters
        ----------
        x : array_like
            Input coordinates to evaluate each parameter at.

        Returns
        -------
        gradient : array_like
            if nu = 1, the gradient vector of the interpolated values with r
            respect to each parameter is computed and returned.
        """
        if not (self._x is None) and np.array_equal(pt, self._x):
            return self._gradient
        else:
            self(pt, nu=1)
            return self._gradient
