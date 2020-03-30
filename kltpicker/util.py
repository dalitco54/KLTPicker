import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter
import operator as op
import os
from scipy.fftpack import fftshift


def f_trans_2(b):
    """
    2-D FIR filter using frequency transformation.

    Produces the 2-D FIR filter h that corresponds to the 1-D FIR
    filter b using the McClellan transform.
    :param b: 1-D FIR filter.
    :return h: 2-D FIR filter.
    """
    # McClellan transformation:
    t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]])/8
    n = int((b.size - 1)/2)
    b = np.flip(b, 0)
    b = fftshift(b)
    b = np.flip(b, 0)
    a = 2*b[0:n+1]
    a[0] = a[0]/2
    # Use Chebyshev polynomials to compute h:
    p0 = 1
    p1 = t
    h = a[1]*p1
    rows = 1
    cols = 1
    h[rows, cols] = h[rows, cols] + a[0]*p0
    p2 = 2 * signal.convolve2d(t, p1)
    p2[2, 2] = p2[2, 2] - p0
    for i in range(2, n+1):
        rows = p1.shape[0]+1
        cols = p1.shape[1]+1
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
    X, Y = np.meshgrid(np.arange(N) * 2 / (N-1) - 1, np.arange(N) * 2 / (N-1) - 1)
    r = np.sqrt(np.square(X) + np.square(Y))
    dr = 1 / (m - 1)
    rbins = np.linspace(-dr / 2, 1 + dr / 2, m + 1, endpoint=True)
    R = (rbins[0:-1] + rbins[1:]) / 2
    zr = np.zeros(m)
    for j in range(m - 1):
        bins = np.where(np.logical_and(r >= rbins[j], r < rbins[j+1]))
        n = np.count_nonzero(np.logical_and(r >= rbins[j], r < rbins[j+1]))
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
    return np.sqrt(c2 - c1 * c1)*np.sqrt(nhood**2./(nhood**2-1))


def als_find_min(sreal, eps, max_iter):
    """
    ALS method for RPSD factorization.

    Approximate Clean and Noise PSD and the particle location vector alpha.
    :param sreal: PSD matrix to be factorized
    :param eps: Convergence term
    :param max_iter: Maximum iterations
    :return approx_clean_psd: Approximated clean PSD
    :return approx_noise_psd: Approximated noise PSD
    :return alpha_approx: Particle location vector alpha.
    :return stop_par: Stop algorithm if an error occurred.
    """
    sz = sreal.shape
    patch_num = sz[1]
    One = np.ones(patch_num)
    s_norm_inf = np.apply_along_axis(lambda x: max(np.abs(x)), 0, sreal)
    max_col = np.argmax(s_norm_inf)
    min_col = np.argmin(s_norm_inf)
    clean_sig_tmp = np.abs(sreal[:, max_col] - sreal[:, min_col])
    s_norm_1 = np.apply_along_axis(lambda x: sum(np.abs(x)), 0, sreal)
    min_col = np.argmin(s_norm_1)
    noise_sig_tmp = np.abs(sreal[:, min_col])
    s = sreal - np.outer(noise_sig_tmp, One)
    alpha_tmp = (clean_sig_tmp@s)/np.sum(clean_sig_tmp**2)
    alpha_tmp = alpha_tmp.clip(min=0, max=1)
    stop_par = 0
    cnt = 1
    while stop_par == 0:
        if np.linalg.norm(alpha_tmp, 1) == 0:
            alpha_tmp = np.random.random(alpha_tmp.size)
        approx_clean_psd = (s @ alpha_tmp)/sum(alpha_tmp ** 2)
        approx_clean_psd = approx_clean_psd.clip(min=0, max=None)
        s = sreal - np.outer(approx_clean_psd, alpha_tmp)
        approx_noise_psd = (s@np.ones(patch_num))/patch_num
        approx_noise_psd = approx_noise_psd.clip(min=0, max=None)
        s = sreal - np.outer(approx_noise_psd, One)
        if np.linalg.norm(approx_clean_psd, 1) == 0:
            approx_clean_psd = np.random.random(approx_clean_psd.size)
        alpha_approx = (approx_clean_psd@s)/sum(approx_clean_psd**2)
        alpha_approx = alpha_approx.clip(min=0, max=1)
        if np.linalg.norm(noise_sig_tmp-approx_noise_psd) / np.linalg.norm(approx_noise_psd) < eps:
            if np.linalg.norm(clean_sig_tmp-approx_clean_psd) / np.linalg.norm(approx_clean_psd) < eps:
                if np.linalg.norm(alpha_approx-alpha_tmp) / np.linalg.norm(alpha_approx) < eps:
                    break
        noise_sig_tmp = approx_noise_psd
        alpha_tmp = alpha_approx
        clean_sig_tmp = approx_clean_psd
        cnt += 1
        if cnt > max_iter:
            stop_par = 1
            break
    return approx_clean_psd, approx_noise_psd, alpha_approx, stop_par


def trig_interpolation(x, y, xq):
    n = x.size
    h = 2 / n
    scale = (x[1] - x[0]) / h
    xs = x / scale
    xi = xq / scale
    p = np.zeros(xi.size)
    for k in range(n):
        if n % 2 == 1:
            a = np.sin(n * np.pi * (xi - xs[k]) / 2) / (n * np.sin(np.pi * (xi - xs[k]) / 2))
        else:
            a = np.sin(n * np.pi * (xi - xs[k]) / 2) / (n * np.tan(np.pi * (xi - xs[k]) / 2))
        a[(xi - xs[k]) == 0] = 1
        p = p + y[k] * a
    return p


def picking_from_scoring_mat(log_test_n, mrc_name, kltpicker, mg_big_size):
    idx_row = np.arange(log_test_n.shape[0])
    idx_col = np.arange(log_test_n.shape[1])
    [col_idx, row_idx] = np.meshgrid(idx_col, idx_row)
    r_del = np.floor(kltpicker.patch_size_pick_box)
    shape = log_test_n.shape
    scoring_mat = log_test_n
    if kltpicker.num_of_particles == -1:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, np.iinfo(np.int32(10)).max, op.gt,
                                                  kltpicker.threshold+1, kltpicker.threshold, kltpicker.patch_size_func,
                                                  row_idx, col_idx, kltpicker.output_particles, mrc_name,
                                                  kltpicker.mgscale, mg_big_size, -np.inf, kltpicker.patch_size_pick_box)
    else:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_particles, op.gt,
                                                  kltpicker.threshold+1, kltpicker.threshold, kltpicker.patch_size_func,
                                                  row_idx, col_idx, kltpicker.output_particles, mrc_name,
                                                  kltpicker.mgscale, mg_big_size, -np.inf, kltpicker.patch_size_pick_box)
    if kltpicker.num_of_noise_images != 0:
        num_picked_noise = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_noise_images, op.lt,
                                              kltpicker.threshold-1, kltpicker.threshold, kltpicker.patch_size_func,
                                              row_idx, col_idx, kltpicker. output_noise, mrc_name,
                                              kltpicker.mgscale, mg_big_size, np.inf, kltpicker.patch_size_pick_box)
    else:
        num_picked_noise = 0
    return num_picked_particles, num_picked_noise


def write_output_files(scoring_mat, shape, r_del, max_iter, oper, oper_param, threshold, patch_size_func, row_idx,
                       col_idx, output_path, mrc_name, mgscale, mg_big_size, replace_param, patch_size_pick_box):
    num_picked = 0
    box_path = output_path+'/box'
    star_path = output_path+'/star'
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    if not os.path.isdir(box_path):
        os.mkdir(box_path)
    if not os.path.isdir(star_path):
        os.mkdir(star_path)
    box_file = open("%s/%s.box" % (box_path, mrc_name), 'w')
    star_file = open("%s/%s.star" % (star_path, mrc_name), 'w')
    star_file.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
    iter_pick = 0
    log_max = np.max(scoring_mat)
    while iter_pick <= max_iter and oper(oper_param, threshold):
        max_index = np.argmax(scoring_mat.transpose().flatten())
        oper_param = scoring_mat.transpose().flatten()[max_index]
        if not oper(oper_param, threshold):
            break
        else:
            [index_col, index_row] = np.unravel_index(max_index, shape)
            ind_row_patch = (index_row - 1) + patch_size_func
            ind_col_patch = (index_col - 1) + patch_size_func
            row_idx_b = row_idx - index_row
            col_idx_b = col_idx - index_col
            rsquare = row_idx_b**2 + col_idx_b**2
            scoring_mat[rsquare <= (r_del**2)] = replace_param
            box_file.write('%i\t%i\t%i\t%i\n' % ((1 / mgscale) * (ind_col_patch + 1 - np.floor(patch_size_pick_box / 2)),
                                            (mg_big_size[0] + 1) - (1 / mgscale) * (ind_row_patch + 1 + np.floor(patch_size_pick_box / 2)),
                                            (1 / mgscale) * patch_size_pick_box, (1 / mgscale) * patch_size_pick_box))
            star_file.write('%i\t%i\t%f\n' % ((1 / mgscale) * (ind_col_patch + 1), (mg_big_size[0] + 1) - ((1 / mgscale) * (ind_row_patch + 1)), oper_param/log_max))
            iter_pick += 1
            num_picked += 1
            print(num_picked)
    star_file.close()
    box_file.close()
    return num_picked
