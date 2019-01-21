import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def grep_diele_matrix(imag = True, prefix = ""):
    input_file = open(prefix + "vasprun.xml", "r")
    outfile =  open(prefix + "E_imag.csv", "w") if imag else open(prefix + "E_real.csv", "w")
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
                outfile.write(','.join(data_list)+'\n')
            count += 1

    input_file.close()
    outfile.close()


def calc_nk(prefix = ""):
    image_df = pd.read_csv(prefix + "E_imag.csv")
    real_df = pd.read_csv(prefix + "E_real.csv")

    #calculate epsilon_real & image
    epsilon_para_real = real_df.loc[:]['xx'] - real_df.loc[:]['xy']
    epsilon_ver_real = real_df.loc[:]['xx'] + 2*real_df.loc[:]['xy']
    epsilon_para_image = image_df.loc[:]['xx'] - image_df.loc[:]['xy']
    epsilon_ver_image = image_df.loc[:]['xx'] + 2*image_df.loc[:]['xy']

    # calculate n and k in both parallel and vertical direction
    n_para = (((epsilon_para_image ** 2 + epsilon_para_real ** 2) ** 0.5 + epsilon_para_real)/2) ** 0.5
    n_ver = (((epsilon_ver_image ** 2 + epsilon_ver_real ** 2) ** 0.5 + epsilon_ver_real) / 2) ** 0.5
    kappa_para = (((epsilon_para_image ** 2 + epsilon_para_real ** 2) ** 0.5 - epsilon_para_real)/2) ** 0.5
    kappa_ver = (((epsilon_ver_image ** 2 + epsilon_ver_real ** 2) ** 0.5 - epsilon_ver_real) / 2) ** 0.5

    #put the energy, n and k (in both parallel and vertical direction)
    energy_n_k = n_para.to_frame(name='n_para')
    energy_n_k.insert(0, 'energy', image_df[:]['Energy'])
    energy_n_k['n_ver'], energy_n_k['kappa_para'], energy_n_k['kappa_ver']=n_ver, kappa_para, kappa_ver
    energy_n_k.to_csv(prefix + 'n_k.csv')


def plot_nk(prefix = ""):
    n_k = pd.read_csv(prefix + 'n_k.csv', index_col = 0)
    n_k.plot(kind='line', x = 'energy', y = n_k.columns[1:])
    plt.xlim((0,5))
    plt.title(prefix)
    plt.show()

def linear_regression():

    #imput is the experiment result with n and k values for different samples
    exp_sample = pd.read_csv('n.csv')

    beta_n_para = pd.read_csv('beta_n_k.csv', index_col = 0).loc[:,['n_para']]
    alpha_n_para = pd.read_csv('alpha_n_k.csv', index_col = 0).loc[:,['n_para']]

    X = (beta_n_para - alpha_n_para).values
    y = exp_sample[:,['415']].values - alpha_n_para.values

    reg_model = LinearRegression()
    trained_model = reg_model.fit(X, y)
    trained_model.score(X, y)
    print(trained_model.coef_)


def main():
    for phase in ['beta_', 'alpha_']:
        grep_diele_matrix(imag = True, prefix = phase)
        grep_diele_matrix(imag = False, prefix = phase)
        calc_nk(prefix = phase)
        # plot_nk(prefix = phase)

    linear_regression()


if __name__ == "__main__":
    main()