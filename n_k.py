from typing import TextIO

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal


np.random.seed(1)


def grep_diele_matrix(imag=True, prefix=""):
    """
    :param imag: extract imaginary and real part from VASP output file vasprun.xml
    :param prefix: defined in the main function
    :return: separate imaginary and real part dataFrame csv files.
    """
    input_file = open(prefix + "vasprun.xml", "r")
    outfile = open(prefix + "E_imag.csv", "w") if imag else open(prefix + "E_real.csv", "w")
    tag = False
    count = 0
    start_tag = "<imag>" if imag else "<real>"

    column_names = "Energy,xx,yy,zz,xy,xz,yz\n"
    outfile.write(column_names)

    for line in input_file:
        if line.strip() == start_tag:
            tag = True
        elif tag:
            if line.strip() == "</set>":
                tag = False
                break
            if count >= 10:
                data_list = line.split()[1:-1]
                outfile.write(','.join(data_list) + '\n')
            count += 1

    input_file.close()
    outfile.close()


def calc_nk(prefix=""):
    """
    :param prefix: defined in the main function
    :return: refraction index for rhombhedral(xx==yy==zz, xy==xz==yz), hexgonal(xx==yy, zz, and xy==xz==yz==0), cubic structure
    """
    image_df = pd.read_csv(prefix + "E_imag.csv")
    real_df = pd.read_csv(prefix + "E_real.csv")

    # calculate epsilon_real & image rhombhedral phase
    epsilon_para_real = real_df.loc[:]['xx'] - real_df.loc[:]['xy']
    epsilon_ver_real = real_df.loc[:]['zz'] + 2 * real_df.loc[:]['xy']
    epsilon_para_image = image_df.loc[:]['xx'] - image_df.loc[:]['xy']
    epsilon_ver_image = image_df.loc[:]['zz'] + 2 * image_df.loc[:]['xy']
    energy_ev = image_df.loc[:]['Energy']

    # calculate n and k in both parallel and vertical direction
    n_para = (((epsilon_para_image ** 2 + epsilon_para_real ** 2) ** 0.5 + epsilon_para_real) / 2) ** 0.5
    n_ver = (((epsilon_ver_image ** 2 + epsilon_ver_real ** 2) ** 0.5 + epsilon_ver_real) / 2) ** 0.5
    kappa_para = (((epsilon_para_image ** 2 + epsilon_para_real ** 2) ** 0.5 - epsilon_para_real) / 2) ** 0.5
    kappa_ver = (((epsilon_ver_image ** 2 + epsilon_ver_real ** 2) ** 0.5 - epsilon_ver_real) / 2) ** 0.5
    wavelength = 1240 * energy_ev ** (-1)

    # put the energy, n and k (in both parallel and vertical direction)
    energy_n_k = n_para.to_frame(name='n_para')
    # energy_n_k.insert(0, 'energy', image_df[:]['Energy'])
    energy_n_k.insert(0, 'wavelength', wavelength)
    energy_n_k['n_ver'], energy_n_k['kappa_para'], energy_n_k['kappa_ver'] = n_ver, kappa_para, kappa_ver
    energy_n_k.to_csv(prefix + 'n_k.csv')


def smooth_nk(prefix=""):
    data = pd.read_csv(prefix + 'n_k.csv')
    X = data.loc[:, ['wavelength']].values

    y_n_para = data.loc[:, ['n_para']].values.ravel()
    y_n_ver = data.loc[:, ['n_ver']].values.ravel()
    y_k_para = data.loc[:, ['kappa_para']].values.ravel()
    y_k_ver = data.loc[:, ['kappa_ver']].values.ravel()

    y_n_para_smooth = scipy.signal.savgol_filter(y_n_para, 51, 3)
    y_n_ver_smooth = scipy.signal.savgol_filter(y_n_ver, 51, 3)
    y_k_para_smooth = scipy.signal.savgol_filter(y_k_para, 51, 3)
    y_k_ver_smooth = scipy.signal.savgol_filter(y_k_ver, 51, 3)

    fig, ax = plt.subplots()
    ax.plot(X, y_n_para_smooth, label="n_para.")
    ax.plot(X, y_n_ver_smooth, label="n_ver.")
    ax.plot(X, y_k_para_smooth, label="k_para")
    ax.plot(X, y_k_ver_smooth, label="k_ver.")
    legend = ax.legend(loc='center right', shadow=False, fontsize='x-large')
    legend.get_frame().set_facecolor('None')

    plt.xlim((300, 1700))
    plt.title(prefix + "In2Se3", fontsize=18)
    plt.xlabel('Wavelength (nm)', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(prefix)
    plt.show()


def plot_nk(prefix=""):
    n_k = pd.read_csv(prefix + 'n_k.csv', index_col=0)
    n_k = n_k.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    n_k.plot(kind='line', x='wavelength', y=n_k.columns[1:])
    plt.plot(volume=True)
    plt.xlim((300, 1700))
    plt.title(prefix, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Wavelength (nm)')
    plt.savefig(prefix + "1")
    plt.show()


def main():
    for phase in ['beta_', 'alpha_', 'gamma_']:
        grep_diele_matrix(imag=True, prefix=phase)
        grep_diele_matrix(imag=False, prefix=phase)
        calc_nk(prefix=phase)
        plot_nk(prefix=phase)
        smooth_nk(prefix=phase)

if __name__ == "__main__":
    main()
