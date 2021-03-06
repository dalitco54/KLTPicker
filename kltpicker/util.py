import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter
from scipy.fftpack import fftshift


def fftcorrelate(image, filt):
    filt = np.rot90(filt, 2)
    pad_shift = 1 - np.mod(np.array(filt.shape), 2)
    filt_center = np.floor((np.array(filt.shape) + 1) / 2).astype("int")
    pad = np.array(filt.shape) - filt_center
    padded_image = np.zeros((image.shape[0] + 2 * pad[0], image.shape[1] + 2 * pad[1]))
    padded_image[pad[0]: pad[0] + image.shape[0], pad[1]: pad[1] + image.shape[1]] = image
    if np.any(pad_shift == 1):
        padded_image = padded_image[pad_shift[0] - 1: -1, pad_shift[1] - 1: -1]
    result = signal.fftconvolve(padded_image, filt, 'valid')
    return result


def f_trans_2(b):
    """
    2-D FIR filter using frequency transformation.

    Produces the 2-D FIR filter h that corresponds to the 1-D FIR
    filter b using the McClellan transform.
    :param b: 1-D FIR filter.
    :return h: 2-D FIR filter.
    """
    # McClellan transformation:
    t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]]) / 8
    n = int((b.size - 1) / 2)
    b = np.flip(b, 0)
    b = fftshift(b)
    b = np.flip(b, 0)
    a = 2 * b[0:n + 1]
    a[0] = a[0] / 2
    # Use Chebyshev polynomials to compute h:
    p0 = 1
    p1 = t
    h = a[1] * p1
    rows = 1
    cols = 1
    h[rows, cols] = h[rows, cols] + a[0] * p0
    p2 = 2 * signal.convolve2d(t, p1)
    p2[2, 2] = p2[2, 2] - p0
    for i in range(2, n + 1):
        rows = p1.shape[0] + 1
        cols = p1.shape[1] + 1
        hh = h
        h = a[i] * p2
        h[1:rows, 1:cols] = h[1:rows, 1:cols] + hh
        p0 = p1
        p1 = p2
        rows += 1
        cols += 1
        p2 = 2 * signal.convolve2d(t, p1)
        p2[2:rows, 2:cols] = p2[2:rows, 2:cols] - p0
    h = np.rot90(h, k=2)
    return h


def radial_avg(z, m):
    """
    Radially average 2-D square matrix z into m bins.

    Computes the average along the radius of a unit circle
    inscribed in the square matrix z. The average is computed in m bins. The radial average is not computed beyond
    the unit circle, in the corners of the matrix z. The radial average is returned in zr and the mid-points of the
    m bins are returned in vector R.
    :param z: 2-D square matrix.
    :param m: Number of bins.
    :return zr: Radial average of z.
    :return R: Mid-points of the bins.
    """
    N = z.shape[1]
    X, Y = np.meshgrid(np.arange(N) * 2 / (N - 1) - 1, np.arange(N) * 2 / (N - 1) - 1)
    r = np.sqrt(np.square(X) + np.square(Y))
    dr = 1 / (m - 1)
    rbins = np.linspace(-dr / 2, 1 + dr / 2, m + 1, endpoint=True)
    R = (rbins[0:-1] + rbins[1:]) / 2
    zr = np.zeros(m)
    for j in range(m - 1):
        bins = np.where(np.logical_and(r >= rbins[j], r < rbins[j + 1]))
        n = np.count_nonzero(np.logical_and(r >= rbins[j], r < rbins[j + 1]))
        if n != 0:
            zr[j] = sum(z[bins]) / n
        else:
            zr[j] = np.nan
    bins = np.where(np.logical_and(r >= rbins[m - 1], r <= 1))
    n = np.count_nonzero(np.logical_and(r >= rbins[m - 1], r <= 1))
    if n != 0:
        zr[m - 1] = sum(z[bins]) / n
    else:
        zr[m - 1] = np.nan
    return zr, R


def stdfilter(a, nhood):
    """Local standard deviation of image."""
    c1 = uniform_filter(a, nhood, mode='reflect')
    c2 = uniform_filter(a * a, nhood, mode='reflect')
    return np.sqrt(c2 - c1 * c1) * np.sqrt(nhood ** 2. / (nhood ** 2 - 1))



def trig_interpolation(x, y, xq):
    n = x.size
    h = 2 / n
    scale = (x[1] - x[0]) / h
    xs = x / scale
    xi = xq / scale
    p = np.zeros(xi.size)
    if n % 2 == 1:
        for k in range(n):
            a = np.sin(n * np.pi * (xi - xs[k]) / 2) / (n * np.sin(np.pi * (xi - xs[k]) / 2))
            a[(xi - xs[k]) == 0] = 1
            p = p + y[k] * a
    else:
        for k in range(n):
            a = np.sin(n * np.pi * (xi - xs[k]) / 2) / (n * np.tan(np.pi * (xi - xs[k]) / 2))
            a[(xi - xs[k]) == 0] = 1
            p = p + y[k] * a
    return p