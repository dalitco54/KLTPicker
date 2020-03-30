#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
from kltpicker.picker import Picker
from kltpicker.util import trig_interpolation

# Globals:
PERCENT_EIG_FUNC = 0.99


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', 'particle_size', type=float, help='Expected size of particles in pixels.')
    parser.add_argument('-i', 'input_dir', type=str, help='Input directory.')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory.')
    parser.add_argument('--num_of_particles', type=int,
                        help='Number of particles to pick per micrograph. If set to -1 will pick all particles.',
                        default=-1)
    parser.add_argument('--num_of_noise_images', type=int, help='Number of noise images to pick per micrograph.',
                        default=0)
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations.', default=6 * (10 ** 4))
    parser.add_argument('--gpu_use', type=bool, action='store_const', default=0)
    parser.add_argument('--max_order', type=int, help='Maximum order of eigenfunction.', default=100)
    parser.add_argument('--percent_eigen_func', type=float, help='', default=0.99)
    parser.add_argument('--max_functions', type=int, help='', default=400)
    parser.add_argument('-v', '--verbose', type=bool, help='Verbose.', default=0)
    parser.add_argument('--threshold', type=float, help='Threshold for the picking', default=0)
    parser.add_argument('--show_figures', type=bool, action='store_const', help='Show figures', default=0)
    parser.add_argument('--preprocess', type=bool, action='store_const', help='Run preprocessing.', default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    picker = Picker(args)
    print("Starting preprocessing.")
    picker.preprocess()
    print("Preprocess finished.\nFetching micrographs...")
    picker.get_micrographs()
    print("Fetched micrographs.\nCutoff filter...")
    for micrograph in picker.micrographs:
        micrograph.cutoff_filter(picker.patch_size)
        print("Done cutoff filter.\nEstimating RPSD I...")
        micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
        print("Done estimating RPSD I.")
        if picker.show_figures:
            plt.figure(1)
            plt.plot(micrograph.r * np.pi, micrograph.approx_clean_psd, label='Approx Clean PSD')
            plt.title('Approx Clean PSD stage I')
            plt.legend()
            # plt.show()
            plt.figure(2)
            plt.plot(micrograph.r * np.pi, micrograph.approx_noise_psd, label='Approx Noise PSD')
            plt.title('Approx Noise PSD stage I')
            plt.legend()
            # plt.show()
        micrograph.approx_noise_psd = micrograph.approx_noise_psd + np.median(micrograph.approx_noise_psd) / 10
        micrograph.prewhiten_micrograph()
        print("Done prewhitening.\nEstimating RPSD II...")
        micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
        print("Done estimating RPSD II.\nConstructing KLT templates...")
        if picker.show_figures:
            plt.figure(3)
            plt.plot(micrograph.r * np.pi, micrograph.approx_clean_psd, label='Approx Clean PSD')
            plt.title('Approx Clean PSD stage II')
            plt.legend()
            # plt.show()
            plt.figure(4)
            plt.plot(micrograph.r * np.pi, micrograph.approx_noise_psd, label='Approx Noise PSD')
            plt.title('Approx Noise PSD stage II')
            plt.legend()
            # plt.show()
        micrograph.psd = np.abs(trig_interpolation(np.pi * micrograph.r.astype('float64'), micrograph.approx_clean_psd,
                                                   picker.rho.astype('float64')))
        if picker.show_figures:
            plt.figure(5)
            plt.plot(picker.rho, micrograph.psd)
            plt.title('Clean Sig Samp at nodes max order %i, percent of eig %f' % (picker.max_order, PERCENT_EIG_FUNC))
            # plt.show()
        micrograph.construct_klt_templates(picker)
        print("Done constructing KLT templates.\nPicking particles...")
        micrograph.detect_particles(picker)
    print("")


if __name__ == "__main__":
    main()
