#!/usr/bin/env python

from pathlib import Path
import warnings
from sys import exit
import argparse
#import matplotlib.pyplot as plt
import numpy as np
from kltpicker.kltpicker import KLTPicker
from kltpicker.util import trig_interpolation
warnings.filterwarnings("ignore")

# Globals:
PERCENT_EIG_FUNC = 0.99
EPS = 10 ** (-2)  # Convergence term for ALS.
PERCENT_EIG_FUNC = 0.99
NUM_QUAD_NYS = 2 ** 10
NUM_QUAD_KER = 2 ** 10
MAX_FUN = 400


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Input directory.', default='c:/users/dalit/Documents/uni/year3/shkol_work')
    parser.add_argument('--output_dir', help='Output directory.', default='c:/users/dalit/Documents/uni/year3/shkol_work')
    parser.add_argument('-s', '--particle_size', help='Expected size of particles in pixels.', default=300, type=int)
    parser.add_argument('--num_of_particles',
                        help='Number of particles to pick per micrograph. If set to -1 will pick all particles.',
                        default=-1, type=int)
    parser.add_argument('--num_of_noise_images', help='Number of noise images to pick per micrograph.',
                        default=0, type=int)
    parser.add_argument('--max_iter', help='Maximum number of iterations.', default=6 * (10 ** 4), type=int)
    parser.add_argument('--gpu_use', action='store_true', default=False)
    parser.add_argument('--max_order', help='Maximum order of eigenfunction.', default=100, type=int)
    parser.add_argument('--percent_eigen_func', help='', default=0.99, type=float)
    parser.add_argument('--max_functions', help='', default=400, type=int)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose.', default=False)
    parser.add_argument('--threshold', help='Threshold for the picking', default=0, type=float)
    parser.add_argument('--show_figures', action='store_true', help='Show figures', default=False)
    parser.add_argument('--preprocess', action='store_false', help='Do not run preprocessing.', default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    num_files = len(list(Path(args.input_dir).glob("*.mrc")))
    if num_files > 0:
        print("Running on %i files." % len(list(Path(args.input_dir).glob("*.mrc"))))
    else:
        print("Could not find any .mrc files in %s. \nExiting..." % args.input_dir)
        exit(0)
    picker = KLTPicker(args)
    if args.preprocess:
        print("Preprocessing...")
        picker.preprocess()
        print("Preprocess finished.")
    else:
        print("Skipping preprocessing.")
    picker.get_micrographs()
    for micrograph in picker.micrographs:
        print("Processing %s" % micrograph.mrc_name)
        print("Cutoff filter...")
        micrograph.cutoff_filter(picker.patch_size)
        print("Done cutoff filter.\nEstimating RPSD I...")
        micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
        print("Done estimating RPSD I.")
        # if picker.show_figures:
        #     plt.figure(1)
        #     plt.plot(micrograph.r * np.pi, micrograph.approx_clean_psd, label='Approx Clean PSD')
        #     plt.title('Approx Clean PSD stage I')
        #     plt.legend()
        #     plt.show()
        #     plt.figure(2)
        #     plt.plot(micrograph.r * np.pi, micrograph.approx_noise_psd, label='Approx Noise PSD')
        #     plt.title('Approx Noise PSD stage I')
        #     plt.legend()
        #     plt.show()
        micrograph.approx_noise_psd = micrograph.approx_noise_psd + np.median(micrograph.approx_noise_psd) / 10
        print("Prewhitening...")
        micrograph.prewhiten_micrograph()
        print("Done prewhitening.\nEstimating RPSD II...")
        micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
        print("Done estimating RPSD II.\nConstructing KLT templates...")
        # if picker.show_figures:
        #     plt.figure(3)
        #     plt.plot(micrograph.r * np.pi, micrograph.approx_clean_psd, label='Approx Clean PSD')
        #     plt.title('Approx Clean PSD stage II')
        #     plt.legend()
        #     plt.show()
        #     plt.figure(4)
        #     plt.plot(micrograph.r * np.pi, micrograph.approx_noise_psd, label='Approx Noise PSD')
        #     plt.title('Approx Noise PSD stage II')
        #     plt.legend()
        #     plt.show()
        micrograph.psd = np.abs(trig_interpolation(np.pi * micrograph.r.astype('float64'), micrograph.approx_clean_psd,
                                                   picker.rho.astype('float64')))
        # if picker.show_figures:
        #     plt.figure(5)
        #     plt.plot(picker.rho, micrograph.psd)
        #     plt.title('Clean Sig Samp at nodes max order %i, percent of eig %f' % (picker.max_order, PERCENT_EIG_FUNC))
        #     plt.show()
        micrograph.construct_klt_templates(picker)
        print("Done constructing KLT templates.\nPicking particles...")
        num_picked_particles, num_picked_noise = micrograph.detect_particles(picker)
        print("Picked %i particles and %i noise images from %s.\n\n" % (
        num_picked_particles, num_picked_noise, micrograph.mrc_name))
    print("Finished successfully.")


if __name__ == "__main__":
    main()

