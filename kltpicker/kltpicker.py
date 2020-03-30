
class KLTPicker:
    """
    KLTpicker object that holds all variables that are used in the computations.

    ...
    Attributes
    ----------
    particle_size : float
        Size of particles to look for in micrographs.
    input_dir : str
        Directory from which to read .mrc files.
    output_dir : str
        Output directory in which to write results.
    gpu_use : bool
        Optional - whether to use GPU or not.
    mgscale : float
        Scaling parameter.
    max_order : int
        Maximal order of eigenfunctions.
    micrographs : np.ndarray
        Array of 2-D micrographs.
    patch_size_pick_box : int
        Particle box size to use.
    num_of_particles : int
        Number of particles to pick per micrograph.
    num_of_noise_images : int
        Number of noise images.
    threshold : float
        Threshold for the picking.
    patch_size : int
        Approximate size of particle after downsampling.
    patch_size_func : int
        Size of disc for computing the eigenfunctions.
    max_iter : int
        Maximal number of iterations for PSD approximation.
    rsamp_length : int
    rad_mat : np.ndarray
    quad_ker : np.ndarray
    quad_nys : np.ndarray
    rho : np.ndarray
    j_r_rho : np.ndarray
    j_samp : np.ndarray
    cosine : np.ndarray
    sine : np.ndarray
    rsamp_r : np.ndarray
    r_r : np.ndarray


    Methods
    -------
    preprocess()
        Initializes parameters needed for the computation.
    get_micrographs()
        Reads .mrc files, downsamples them and adds them to the KLTpicker object.
    """
    def __init__(self, args):
        self.particle_size = args.particle_size
        self.input_dir = args.input_dir
        self.output_dir = args.output_dir
        self.output_noise = '%s/PickedNoise_ParticleSize_%d' % (args.output_dir, args.particle_size)
        self.output_particles = '%s/PickedParticles_ParticleSize_%d' % (args.output_dir, args.particle_size)
        self.gpu_use = args.gpu_use
        self.mgscale = 100/args.particle_size
        self.max_order = args.max_order
        self.micrographs = np.array([])
        self.quad_ker = 0
        self.quad_nys = 0
        self.rho = 0
        self.j_r_rho = 0
        self.j_samp = 0
        self.cosine = 0
        self.sine = 0
        self.rsamp_r = 0
        self.r_r = 0
        self.patch_size_pick_box = np.floor(self.mgscale * args.particle_size)
        self.num_of_particles = args.num_of_particles
        self.num_of_noise_images = args.num_of_noise_images
        self.threshold = args.threshold
        self.show_figures = args.show_figures
        patch_size = np.floor(0.8 * self.mgscale * args.particle_size)  # need to put the 0.8 somewhere else.
        if np.mod(patch_size, 2) == 0:
            patch_size -= 1
        self.patch_size = patch_size
        patch_size_function = np.floor(0.4 * self.mgscale * args.particle_size)  # need to put the 0.4 somewhere else.
        if np.mod(patch_size_function, 2) == 0:
            patch_size_function -= 1
        self.patch_size_func = int(patch_size_function)
        self.max_iter = args.max_iter
        self.rsamp_length = 0
        self.rad_mat = 0

    def preprocess(self):
        """Initializes parameters."""
        radmax = np.floor((self.patch_size_func - 1) / 2)
        x = np.arange(-radmax, radmax+1, 1).astype('float64')
        X, Y = np.meshgrid(x, x)
        rad_mat = np.sqrt(np.square(X) + np.square(Y))
        rsamp = rad_mat.transpose().flatten()
        self.rsamp_length = rsamp.size
        theta = np.arctan2(Y, X).transpose().flatten()
        quad_ker_sample_points = lgwt(NUM_QUAD_KER, 0, np.pi)
        rho = quad_ker_sample_points.x
        quad_ker = quad_ker_sample_points.w
        rho = np.flipud(rho.astype('float64'))
        quad_ker = np.flipud(quad_ker.astype('float64'))
        quad_nys_sample_points = lgwt(NUM_QUAD_NYS, 0, radmax)
        r = quad_nys_sample_points.x
        quad_nys = quad_nys_sample_points.w
        r = np.flipud(r.astype('float64'))
        quad_nys = np.flipud(quad_nys.astype('float64'))
        r_r = np.outer(r, r)
        r_rho = np.outer(r, rho)
        rsamp_r = np.outer(np.ones(len(rsamp)), r)
        rsamp_rho = np.outer(rsamp, rho)
        j_r_rho = np.zeros([NUM_QUAD_KER, NUM_QUAD_NYS, self.max_order]).astype('float64')
        j_samp = np.zeros([len(rsamp), NUM_QUAD_NYS, self.max_order]).astype('float64')
        cosine = np.zeros([len(theta), self.max_order]).astype('float64')
        sine = np.zeros([len(theta), self.max_order]).astype('float64')
        for n in range(self.max_order):
            j_r_rho[:, :, n] = ssp.jv(n, r_rho).astype('float64')
            j_samp[:, :, n] = ssp.jv(n, rsamp_rho).astype('float64')
            if n != 0:
                cosine[:, n] = np.cos(n * theta).astype('float64')
                sine[:, n] = np.sin(n * theta).astype('float64')
        self.quad_ker = quad_ker
        self.quad_nys = quad_nys
        self.rho = rho
        self.j_r_rho = j_r_rho
        self.j_samp = j_samp
        self.cosine = cosine
        self.sine = sine
        self.rsamp_r = rsamp_r
        self.r_r = r_r
        self.rad_mat = rad_mat

    def get_micrographs(self):
        """Reads .mrc files, downsamples them and adds them to the KLTpicker object."""
        micrographs = []
        mrc_files = glob.glob("%s/*.mrc" % self.input_dir)
        for mrc_file in mrc_files:
            mrc = mrcfile.open(mrc_file)
            mrc_data = mrc.data.astype('float64').transpose()
            mrc.close()
            mrc_size = mrc_data.shape
            mrc_data = np.rot90(mrc_data)
            data = downsample(mrc_data[np.newaxis, :, :], int(np.floor(self.mgscale * mrc_size[0])))[0]
            if np.mod(data.shape[0], 2) == 0:  # Odd size is needed.
                data = data[0:-1, :]
            if np.mod(data.shape[1], 2) == 0:  # Odd size is needed.
                data = data[:, 0:-1]
            pic = data   # For figures before standardization.
            data = data - np.mean(data.transpose().flatten())
            data = data/np.linalg.norm(data, 'fro')
            mc_size = data.shape
            micrograph = Micrograph(data, pic, mc_size, mrc_file.split('/')[-1], mrc_size)
            micrographs.append(micrograph)
        micrographs = np.array(micrographs)
        self.micrographs = micrographs

