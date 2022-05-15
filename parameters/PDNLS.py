from back_process import *
from functions import *
from time_integrators import *

if __name__ == '__main__':
    unidad = 1
    g = 9790 * unidad
    L = 480 * unidad
    l_x = L / 3
    l_y = 16 * unidad  # +- 2mm
    d = 20 * unidad
    n = 5
    m = 1
    a_ang = 8.1 / 2
    f = 13.5

    a_mm = 1 + (a_ang / 12) * unidad
    sigma = l_x / (2 * 2.3482)
    k_x = np.pi * n / l_x
    k_y = np.pi * m / l_y
    k = np.sqrt(0 * k_x ** 2 + k_y ** 2)
    tau = np.tanh(k * d)
    w_1 = np.sqrt(g * k * tau)
    w = (f / 2) * 2 * np.pi
    GAMMA = 4 * w ** a_mm

    alpha = (1 / (4 * k ** 2)) * (1 + k * d * ((1 - tau ** 2) / tau))  # término difusivo
    beta = (k ** 2 / 64) * (6 * tau ** 2 - 5 + 16 * tau ** (-2) - 9 * tau ** (-4))  # término no lineal
    gamma_0 = GAMMA / (4 * g)  # amplitud adimensional
    nu = 0.5 * ((w / w_1) ** 2 - 1)
    mu = 0.0125
    length = l_x

    arnold_tongue_show(gamma_0, mu, nu)

    print('alpha = ' + str(alpha))
    print('beta = ' + str(beta))
    print('gamma = ' + str(gamma_0))
    print('nu = ' + str(nu))
    print('mu = ' + str(mu))
    print('sigma = ' + str(sigma))
    print((2 * w_1) / (2 * np.pi))