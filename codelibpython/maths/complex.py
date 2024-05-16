#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: G. Howell
"""

def cplxpair(x, tol=None, dim=None):
    """
    Sorts values into complex pairs a la Matlab.

    The function takes a vector or multidimensional array of of complex
    conjugate pairs or real numbers and rearranges it so that the complex
    numbers are collected into matched pairs of complex conjugates. The pairs
    are ordered by increasing real part, with purely real elements placed
    after all the complex pairs.

    In the search for complex conjugate pairs a relative tolerance equal to
    ``tol`` is used for comparison purposes. The default tolerance is
    100 times the system floating point accuracy.

    If the input vector is a multidimensional array, the rearrangement is done
    working along the axis specifid by the parameter ``dim`` or along the
    first axis with non-unitary length if ``dim`` is not provided.

    Parameters
    ----------
    x : array_like of complex
        x is an array of complex values, with the assumption that it contains
        either real values or complex values in conjugate pairs.
    tol: real, optional
        relative tolerance for the recognition of pairs.
        Defaults to 100 times the system floating point accuracy for the
        specific number type.
    dim: integer, optional
        The axis to operate upon.

    Returns
    -------
    y : ndarray
        y is an array of complex values, with the same values in x, yet now
        sorted as complex pairs by increasing real part. Real elements in x
        are place after the complex pairs, sorted in increasing order.

    Raises
    ------
    ValueError
        'Complex numbers cannot be paired' if there are unpaired complex
        entries in x.

    Examples
    --------
    >>> a = np.exp(2j*np.pi*np.arange(0, 5)/5)
    >>> b1 = cplxpair(a)
    >>> b2 = np.asarray([-0.80901699-0.58778525j, -0.80901699+0.58778525j,
    ...                   0.30901699-0.95105652j,  0.30901699+0.95105652j,
    ...                   1.00000000+0.j])
    >>> np.allclose(b1, b2)
    True

    >>> cplxpair(1)
    array([1])

    >>> cplxpair([[5, 6, 4], [3, 2, 1]])
    array([[3, 2, 1],
           [5, 6, 4]])

    >>> cplxpair([[5, 6, 4], [3, 2, 1]], dim=1)
    array([[4, 5, 6],
           [1, 2, 3]])

    See also
    --------
    eps : the system floating point accuracy
    """

    def cplxpair_vec(x, tol):
        real_mask = np.abs(x.imag) <= tol*np.abs(x)
        x_real = np.sort(np.real(x[real_mask]))
        x_cplx = np.sort(x[np.logical_not(real_mask)])
        if x_cplx.size == 0:
            return x_real
        if (x_cplx.size % 2) != 0:
            raise ValueError('Complex numbers cannot be paired')
        if np.any(np.real(x_cplx[1::2])-np.real(x_cplx[0::2]) >
                  tol*np.abs(x_cplx[0::2])):
            raise ValueError('Complex numbers cannot be paired')
        start = 0
        while start < x_cplx.size:
            sim_len = next((i for i, v in enumerate(x_cplx[start+1:]) if
                           (np.abs(np.real(v)-np.real(x_cplx[start])) >
                            tol*np.abs(v))), x_cplx.size-start-1)+1
            if (sim_len % 2) != 0:
                sim_len -= 1
            # At this point, sim_len elements with identical real part
            # have been identified.
            sub_x = x_cplx[start:start+sim_len]
            srt = np.argsort(np.imag(sub_x))
            sub_x = sub_x[srt]
            if np.any(np.abs(np.imag(sub_x)+np.imag(sub_x[::-1])) >
                      tol*np.abs(sub_x)):
                raise ValueError('Complex numbers cannot be paired')
            # Output should contain "perfect" pairs. Hence, keep entries
            # with positive imaginary parts amd use conjugate for pair
            x_cplx[start:start+sim_len] = np.concatenate(
                (np.conj(sub_x[:sim_len//2-1:-1]),
                 sub_x[:sim_len//2-1:-1]))
            start += sim_len
        return np.concatenate((x_cplx, x_real))

    x = np.atleast_1d(x)
    if x.size == 0:
        return x
    if dim is None:
        dim = next((i for i, v in enumerate(x.shape) if v > 1), 0)
    if tol is None:
        try:
            tol = 100*eps(x.dtype)
        except:
            tol = 100*eps(np.float)
    return np.apply_along_axis(cplxpair_vec, dim, x, tol)