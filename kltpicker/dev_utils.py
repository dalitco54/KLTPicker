
def get_matlab_files_preprocess(kltpicker):
    # PREPROCESS:
    kltpicker.rsamp_length = 1521
    kltpicker.j_samp = mat_to_npy('Jsamp' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.cosine = mat_to_npy('cosine' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.sine = mat_to_npy('sine' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.j_r_rho = mat_to_npy('JrRho' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.quad_ker = mat_to_npy_vec('quadKer' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.quad_nys = mat_to_npy_vec('quadNys' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.rho = mat_to_npy_vec('rho' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.rsamp_r = mat_to_npy('rSampr' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.r_r = mat_to_npy('rr' ,'/home/dalitcohen/Documents/kltdata/matlab64')
    kltpicker.rad_mat = mat_to_npy('radMat' ,'/home/dalitcohen/Documents/kltdata/matlab64')

def get_matlab_files(micrograph):
    micrograph.r = mat_to_npy_vec('R', '/home/dalitcohen/Documents/kltdata/matlab64')
    micrograph.approx_clean_psd = mat_to_npy_vec('apprxCleanPsd', '/home/dalitcohen/Documents/kltdata/matlab64')
    micrograph.approx_noise_psd = mat_to_npy_vec('apprxNoisePsd', '/home/dalitcohen/Documents/kltdata/matlab64')
    micrograph.micrograph = mat_to_npy('mg', '/home/dalitcohen/Documents/kltdata/matlab64')
    # micrograph.noise_mc = mat_to_npy('noiseMc','/home/dalitcohen/Documents/kltdata/matlab')
    # BEFORE DETECT PARTICLES:
    # micrograph.eig_func = mat_to_npy('eigFun', '/home/dalitcohen/Documents/kltdata/matlab')
    # micrograph.eig_val = mat_to_npy_vec('eigVal', '/home/dalitcohen/Documents/kltdata/matlab')
    micrograph.noise_mc = mat_to_npy('noiseMc', '/home/dalitcohen/Documents/kltdata/matlab64')
    micrograph.num_of_func = 400
    micrograph.approx_noise_var = mat_to_npy('noiseVar', '/home/dalitcohen/Documents/kltdata/matlab64')
    micrograph.psd = mat_to_npy_vec('psd', '/home/dalitcohen/Documents/kltdata/matlab64')

def getting_np_files(kltpicker):
    # PREPROCESS:
    kltpicker.rsamp_length = np.load('/home/dalitcohen/Documents/kltdata/numpy/rsamp_length.npy')
    kltpicker.j_samp = np.load('/home/dalitcohen/Documents/kltdata/numpy/j_samp.npy')
    kltpicker.cosine = np.load('/home/dalitcohen/Documents/kltdata/numpy/cosine.npy')
    kltpicker.sine = np.load('/home/dalitcohen/Documents/kltdata/numpy/sine.npy')
    kltpicker.j_r_rho = np.load('/home/dalitcohen/Documents/kltdata/numpy/j_r_rho.npy')
    kltpicker.quad_ker = np.load('/home/dalitcohen/Documents/kltdata/numpy/quad_ker.npy')
    kltpicker.quad_nys = np.load('/home/dalitcohen/Documents/kltdata/numpy/quad_nys.npy')
    kltpicker.rho = np.load('/home/dalitcohen/Documents/kltdata/numpy/rho.npy')
    kltpicker.rsamp_r = np.load('/home/dalitcohen/Documents/kltdata/numpy/rsamp_r.npy')
    kltpicker.r_r = np.load('/home/dalitcohen/Documents/kltdata/numpy/r_r.npy')
    kltpicker.rad_mat = np.load('/home/dalitcohen/Documents/kltdata/numpy/rad_mat.npy')

    # micrograph.approx_noise_psd = np.load('/home/dalitcohen/Documents/kltdata/numpy/approx_noise_psd.npy')
    # micrograph.approx_noise_var = np.load('/home/dalitcohen/Documents/kltdata/numpy/approx_noise_var.npy')
    # micrograph.approx_clean_psd = np.load('/home/dalitcohen/Documents/kltdata/numpy/approx_clean_psd.npy')
    # micrograph.noise_mc = np.load('/home/dalitcohen/Documents/kltdata/numpy/noise_mc.npy')
    # micrograph.psd = np.load('/home/dalitcohen/Documents/kltdata/numpy/psd.npy')
    # micrograph.eig_func = np.load('/home/dalitcohen/Documents/kltdata/numpy/eig_func.npy')
    # micrograph.eig_val = np.load('/home/dalitcohen/Documents/kltdata/numpy/eig_val.npy')
    # micrograph.num_of_func = 400
