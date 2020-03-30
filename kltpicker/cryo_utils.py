import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw import FFTW


def crop(x, out_shape):
    """

    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i is the size we want to cut from the
        center of x in dimension i. If the value is <= 0 then the dimension is left as is
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if s > 0 else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out


def downsample(stack, n, mask=None, stack_in_fourier=False):
    """ Use Fourier methods to change the sample interval and/or aspect ratio
        of any dimensions of the input image 'img'. If the optional argument
        stack is set to True, then the *first* dimension of 'img' is interpreted as the index of
        each image in the stack. The size argument side is an integer, the size of the
        output images.  Let the size of a stack
        of 2D images 'img' be n1 x n1 x k.  The size of the output will be side x side x k.

        If the optional mask argument is given, this is used as the
        zero-centered Fourier mask for the re-sampling. The size of mask should
        be the same as the output image size. For example for downsampling an
        n0 x n0 image with a 0.9 x nyquist filter, do the following:
        msk = fuzzymask(n,2,.45*n,.05*n)
        out = downsample(img, n, 0, msk)
        The size of the mask must be the size of output. The optional fx output
        argument is the padded or cropped, masked, FT of in, with zero
        frequency at the origin.
    """

    size_in = np.square(stack.shape[1])
    size_out = np.square(n)
    mask = 1 if mask is None else mask
    num_images = stack.shape[0]
    output = np.zeros((num_images, n, n), dtype='float64')
    images_batches = np.array_split(np.arange(num_images), 500)
    for batch in images_batches:
        if batch.size:
            curr_batch = np.array(stack[batch])
            curr_batch = curr_batch if stack_in_fourier else fft2(curr_batch)
            fx = crop(np.fft.fftshift(curr_batch, axes=(-2, -1)), (-1, n, n)) * mask
            output[batch] = ifft2(np.fft.ifftshift(fx, axes=(-2, -1))) * (size_out / size_in)
            print('finished {}/{}'.format(batch[-1] + 1, num_images))
    return output

def cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = np.fft.ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = np.fft.ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft(x, axis=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axis), axis=axis), axis)


def fast_cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return fftshift(np.transpose(fft2(np.transpose(ifftshift(x)))))
    elif len(x.shape) == 3:
        y = ifftshift(x, axes=axes)
        y = fft2(y, axes=axes)
        y = fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def fast_icfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return fftshift(np.transpose(ifft2(np.transpose(ifftshift(x)))))

    elif len(x.shape) == 3:
        y = ifftshift(x, axes=axes)
        y = ifft2(y, axes=axes)
        y = fftshift(y, axes=axes)
        return y

    else:
        raise ValueError("x must be 2D or 3D")


def lgwt(n, a, b):
    """
    Get n leggauss points in interval [a, b]

    :param n: number of points
    :param a: interval starting point
    :param b: interval end point
    :returns SamplePoints(x, w): sample points, weight
    """

    x1, w = leggauss(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return create_struct({'x': x, 'w': w})

def fill_struct(obj=None, att_vals=None, overwrite=None):
    """
    Fill object with attributes in a dictionary.
    If a struct is not given a new object will be created and filled.
    If the given struct has a field in att_vals, the original field will stay, unless specified otherwise in overwrite.
    att_vals is a dictionary with string keys, and for each key:
    if hasattr(s, key) and key in overwrite:
        pass
    else:
        setattr(s, key, att_vals[key])
    :param obj:
    :param att_vals:
    :param overwrite
    :return:
    """
    if obj is None:
        class DisposableObject:
            pass

        obj = DisposableObject()

    if att_vals is None:
        return obj

    if overwrite is None or not overwrite:
        overwrite = []
    if overwrite is True:
        overwrite = list(att_vals.keys())

    for key in att_vals.keys():
        if hasattr(obj, key) and key not in overwrite:
            continue
        else:
            setattr(obj, key, att_vals[key])

    return obj


def create_struct(att_vals=None):
    """
    Creates object
    :param att_vals:
    :return:
    """
    return fill_struct(att_vals=att_vals)

def cryo_epsds(imstack, samples_idx, max_d):
    p = imstack.shape[0]
    if max_d >= p:
        max_d = p-1
        print('max_d too large. Setting max_d to {}'.format(max_d))

    r, x, _ = cryo_epsdr(imstack, samples_idx, max_d)

    r2 = np.zeros((2 * p - 1, 2 * p - 1))
    dsquare = np.square(x)
    for i in range(-max_d, max_d + 1):
        for j in range(-max_d, max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d*(1-1e-13), d*(1+1e-13))
                r2[i+p-1, j+p-1] = r[idx-1]

    w = gwindow(p, max_d)
    p2 = fast_cfft2(r2 * w)

    p2 = p2.real

    e = 0
    for i in range(imstack.shape[2]):
        im = imstack[:, :, i]
        e += np.sum(np.square(im[samples_idx] - np.mean(im[samples_idx])))

    mean_e = e / (len(samples_idx[0]) * imstack.shape[2])
    p2 = (p2 / p2.sum()) * mean_e * p2.size
    neg_idx = np.where(p2 < 0)
    p2[neg_idx] = 0
    return p2, r, r2, x


def cryo_epsdr(vol, samples_idx, max_d):
    p = vol.shape[0]
    k = vol.shape[2]
    i, j = np.meshgrid(np.arange(max_d + 1), np.arange(max_d + 1))
    dists = np.square(i) + np.square(j)
    dsquare = np.sort(np.unique(dists[np.where(dists <= max_d ** 2)]))

    corrs = np.zeros(len(dsquare))
    corr_count = np.zeros(len(dsquare))
    x = np.sqrt(dsquare)

    dist_map = np.zeros(dists.shape)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d - 1e-13, d + 1e-13)
                dist_map[i, j] = idx

    dist_map = dist_map.astype('int') - 1
    valid_dists = np.where(dist_map != -1)

    mask = np.zeros((p, p))
    mask[samples_idx] = 1
    tmp = np.zeros((2 * p + 1, 2 * p + 1))
    tmp[:p, :p] = mask
    ftmp = np.fft.fft2(tmp)
    c = np.fft.ifft2(ftmp * np.conj(ftmp))
    c = c[:max_d+1, :max_d+1]
    c = np.round(c.real).astype('int')

    r = np.zeros(len(corrs))

    # optimized version
    vol = vol.transpose((2, 0, 1)).copy()
    input_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    input_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    flags = ('FFTW_MEASURE', 'FFTW_UNALIGNED')
    fft2 = FFTW(input_fft2, output_fft2, axes=(0, 1), direction='FFTW_FORWARD', flags=flags)
    ifft2 = FFTW(input_ifft2, output_ifft2, axes=(0, 1), direction='FFTW_BACKWARD', flags=flags)
    sum_s = np.zeros(output_ifft2.shape, output_ifft2.dtype)
    sum_c = c * vol.shape[0]
    for i in range(k):
        proj = vol[i]

        input_fft2[samples_idx] = proj[samples_idx]
        fft2()
        np.multiply(output_fft2, np.conj(output_fft2), out=input_ifft2)
        ifft2()
        sum_s += output_ifft2

    for curr_dist in zip(valid_dists[0], valid_dists[1]):
        dmidx = dist_map[curr_dist]
        corrs[dmidx] += sum_s[curr_dist].real
        corr_count[dmidx] += sum_c[curr_dist]

    idx = np.where(corr_count != 0)[0]
    r[idx] += corrs[idx] / corr_count[idx]
    cnt = corr_count[idx]

    idx = np.where(corr_count == 0)[0]
    r[idx] = 0
    x[idx] = 0
    return r, x, cnt


def gwindow(p, max_d):
    x, y = np.meshgrid(np.arange(-(p-1), p), np.arange(-(p-1), p))
    alpha = 3.0
    w = np.exp(-alpha * (np.square(x) + np.square(y)) / (2 * max_d ** 2))
    return w


def bsearch(x, lower_bound, upper_bound):
    if lower_bound > x[-1] or upper_bound < x[0] or upper_bound < lower_bound:
        return None, None
    lower_idx_a = 1
    lower_idx_b = len(x)
    upper_idx_a = 1
    upper_idx_b = len(x)

    while lower_idx_a + 1 < lower_idx_b or upper_idx_a + 1 < upper_idx_b:
        lw = int(np.floor((lower_idx_a + lower_idx_b) / 2))
        if x[lw-1] >= lower_bound:
            lower_idx_b = lw
        else:
            lower_idx_a = lw
            if upper_idx_a < lw < upper_idx_b:
                upper_idx_a = lw

        up = int(np.ceil((upper_idx_a + upper_idx_b) / 2))
        if x[up-1] <= upper_bound:
            upper_idx_a = up
        else:
            upper_idx_b = up
            if lower_idx_a < up < lower_idx_b:
                lower_idx_b = up

    if x[lower_idx_a-1] >= lower_bound:
        lower_idx = lower_idx_a
    else:
        lower_idx = lower_idx_b
    if x[upper_idx_b-1] <= upper_bound:
        upper_idx = upper_idx_b
    else:
        upper_idx = upper_idx_a

    if upper_idx < lower_idx:
        return None, None

    return lower_idx, upper_idx
