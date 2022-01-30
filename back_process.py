import numpy as np
import matplotlib
import shutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import datetime
from playsound import playsound
from scipy.integrate import odeint
import scipy.sparse as sparse
from scipy import signal
from scipy.fftpack import fft, fftshift
import os
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def triangle(length, amplitude):
    section = length // 4
    for direction in (1, -1):
        for i in range(section):
            yield i * (amplitude / section) * direction
        for i in range(section):
            yield (amplitude - (i * (amplitude / section))) * direction


def arnold_tongue_save(gamma_0, mu, nu, file):
    nu_positive_grid = np.arange(0, 2, 0.01)
    nu_negative_grid = - np.flip(nu_positive_grid)
    nu_grid = np.append(nu_negative_grid, nu_positive_grid)
    plt.plot(nu_positive_grid, np.sqrt(nu_positive_grid ** 2 + mu ** 2), c='k', linestyle='--')
    plt.fill_between(nu_positive_grid, np.ones(len(nu_positive_grid)) * mu,
                     np.sqrt(nu_positive_grid ** 2 + mu ** 2),
                     facecolor=(92 / 255, 43 / 255, 228 / 255, 0.4))
    plt.plot(nu_negative_grid, np.sqrt(nu_negative_grid ** 2 + mu ** 2), c='k', linestyle='--')
    plt.fill_between(nu_negative_grid, np.ones(len(nu_negative_grid)) * mu,
                     np.sqrt(nu_negative_grid ** 2 + mu ** 2),
                     facecolor=(0, 1, 0, 0.4))
    plt.plot(nu_grid, np.ones(len(nu_grid)) * mu, c='k', linestyle='--')
    plt.fill_between(nu_grid, 2, np.sqrt(nu_grid ** 2 + mu ** 2),
                     facecolor=(1, 0, 0, 0.4))
    plt.fill_between(nu_grid, np.ones(len(nu_grid)) * mu, 0,
                     facecolor=(1, 1, 0, 0.4))
    plt.scatter(nu, gamma_0, c='k', zorder=10)
    plt.title('Arnold Tongue', size='25')
    plt.xlabel('$\\nu$', size='25')
    plt.ylabel('$\gamma$', size='25')
    plt.xlim([-1, 1])
    plt.ylim([0, 1])
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(file + '/arnold_tongue.png')
    plt.close()


def arnold_tongue_show(gamma_0, mu, nu):
    nu_positive_grid = np.arange(0, 2, 0.01)
    nu_negative_grid = - np.flip(nu_positive_grid)
    nu_grid = np.append(nu_negative_grid, nu_positive_grid)
    plt.plot(nu_positive_grid, np.sqrt(nu_positive_grid ** 2 + mu ** 2), c='k', linestyle='--')
    plt.fill_between(nu_positive_grid, np.ones(len(nu_positive_grid)) * mu,
                     np.sqrt(nu_positive_grid ** 2 + mu ** 2),
                     facecolor=(92 / 255, 43 / 255, 228 / 255, 0.4))
    plt.plot(nu_negative_grid, np.sqrt(nu_negative_grid ** 2 + mu ** 2), c='k', linestyle='--')
    plt.fill_between(nu_negative_grid, np.ones(len(nu_negative_grid)) * mu,
                     np.sqrt(nu_negative_grid ** 2 + mu ** 2),
                     facecolor=(0, 1, 0, 0.4))
    plt.plot(nu_grid, np.ones(len(nu_grid)) * mu, c='k', linestyle='--')
    plt.fill_between(nu_grid, 2, np.sqrt(nu_grid ** 2 + mu ** 2),
                     facecolor=(1, 0, 0, 0.4))
    plt.fill_between(nu_grid, np.ones(len(nu_grid)) * mu, 0,
                     facecolor=(1, 1, 0, 0.4))
    plt.scatter(nu, gamma_0, c='k', zorder=10)
    plt.title('Arnold Tongue', size='25')
    plt.xlabel('$\\nu$', size='25')
    plt.ylabel('$\gamma$', size='25')
    plt.xlim([-1, 1])
    plt.ylim([0, 1])
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def arnold_tongue_exp_show(gamma_0, mu, nu):
    nu_positive_grid = np.arange(0, 2, 0.01)
    nu_negative_grid = - np.flip(nu_positive_grid)
    nu_grid = np.append(nu_negative_grid, nu_positive_grid)
    plt.plot(nu_positive_grid, np.sqrt(nu_positive_grid ** 2 + mu ** 2), c='k', linestyle='--')
    plt.fill_between(nu_positive_grid, np.ones(len(nu_positive_grid)) * mu,
                     np.sqrt(nu_positive_grid ** 2 + mu ** 2),
                     facecolor=(92 / 255, 43 / 255, 228 / 255, 0.4))
    plt.plot(nu_negative_grid, np.sqrt(nu_negative_grid ** 2 + mu ** 2), c='k', linestyle='--')
    plt.fill_between(nu_negative_grid, np.ones(len(nu_negative_grid)) * mu,
                     np.sqrt(nu_negative_grid ** 2 + mu ** 2),
                     facecolor=(0, 1, 0, 0.4))
    plt.plot(nu_grid, np.ones(len(nu_grid)) * mu, c='k', linestyle='--')
    plt.fill_between(nu_grid, 2, np.sqrt(nu_grid ** 2 + mu ** 2),
                     facecolor=(1, 0, 0, 0.4))
    plt.fill_between(nu_grid, np.ones(len(nu_grid)) * mu, 0,
                     facecolor=(1, 1, 0, 0.4))
    plt.scatter(nu, gamma_0, c='k', zorder=10)
    plt.title('Arnold Tongue', size='25')
    plt.xlabel('$\\nu$', size='25')
    plt.ylabel('$\gamma$', size='25')
    plt.xlim([-0.25, 0.25])
    plt.ylim([0, 0.25])
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

def campos_ligeros(campos, n, Nt, Nx, T):
    t_ligero = np.linspace(0, T, int(Nt / n))
    campos_light = []
    for k in range(len(campos)):
        campo_ligero = np.zeros((int(Nt / n), Nx))
        for i in range(0, len(campos[k][:, 0]) - 1, n):
            campo_ligero[int(i / n), :] = campos[k][i, :]
        campos_light.append(campo_ligero)
    return campos_light, t_ligero


def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)


def nombre_pndls_LLG(gamma, mu, nu):
    mu_st = str(round(float(mu), 4))
    gamma_st = str(round(float(gamma), 4))
    nu_st = str(round(float(nu), 4))
    nombre = '/gaussian/mu=' + mu_st + '/gamma=' + gamma_st + '/nu=' + nu_st
    return nombre

def nombre_pndls_gaussian(gamma, mu, nu, sigma):
    sigma_st = str(round(float(sigma), 4))
    mu_st = str(round(float(mu), 4))
    gamma_st = str(round(float(gamma), 4))
    nu_st = str(round(float(nu), 4))
    nombre = '/gaussian/mu=' + mu_st + '/gamma=' + gamma_st + '/nu=' + nu_st + '/sigma=' + sigma_st
    return nombre

def nombre_pndls_squared(gamma, mu, nu, length):
    length_st = str(round(float(length), 4))
    mu_st = str(round(float(mu), 4))
    gamma_st = str(round(float(gamma), 4))
    nu_st = str(round(float(nu), 4))
    nombre = '/squared/mu=' + mu_st + '/gamma=' + gamma_st + '/nu=' + nu_st + '/length=' + length_st
    return nombre

def nombre_pndls_bigaussian(gamma, mu, nu, sigma1, sigma2, dist, fase):
    gamma_st = str(truncate(gamma, 4))
    mu_st = str(truncate(mu, 4))
    nu_st = str(truncate(nu, 4))
    sigma1_st = str(truncate(sigma1, 4))
    sigma2_st = str(truncate(sigma2, 4))
    dist_st = str(truncate(dist, 4))
    fase_st = str(truncate(fase / np.pi, 4)) + 'pi'
    nombre = '/bigaussian/mu=' + mu_st + '/gamma=' + gamma_st + '/nu=' + nu_st + '/fase=' + fase_st + '/sigma_1=' + \
             sigma1_st + '_sigma_2=' + sigma2_st + '\\distancia=' + dist_st
    return nombre


def guardar_txt(path, file, **kwargs): # upgradear a diccionario para nombre de variables
    if file == 'no':
        pathfile = path
    else:
        pathfile = path + file
    if os.path.exists(pathfile) == False:
        os.makedirs(pathfile)
    for key, value in kwargs.items():
        np.savetxt(pathfile + '\\' + key + ".txt", value)


def guardar_csv(path, file, **kwargs): # upgradear a diccionario para nombre de variables
    if file == 'no':
        pathfile = path
    else:
        pathfile = path + file
    if os.path.exists(pathfile) == False:
        os.makedirs(pathfile)
    for key, value in kwargs.items():
        np.savetxt(pathfile + '\\' + key + ".csv", value)


def random_transposition(k, N):
    return np.transpose(np.array([k] * N))