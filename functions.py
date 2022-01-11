from back_process import *


def sparse_DD(Nx, dx):
    data = np.ones((3, Nx))
    data[1] = -2 * data[1]
    diags = [-1, 0, 1]
    D2 = sparse.spdiags(data, diags, Nx, Nx) / (dx ** 2)
    D2 = sparse.lil_matrix(D2)
    D2[0, -1] = 1 / (dx ** 2)
    D2[-1, 0] = 1 / (dx ** 2)
    return D2


def Dxx(DD, f):
    dd_f = DD.dot(f)
    return dd_f


def equations_FD(eq, field_slices, parameters, x_grid, operator):
    if eq == 'PNDLS_forced':
        U_1 = field_slices[0]
        U_2 = field_slices[1]

        alpha = parameters[0]
        beta = parameters[1]
        gamma = parameters[2]
        gamma_1 = gamma[0]
        gamma_2 = gamma[1]
        mu = parameters[3]
        nu = parameters[4]

        ddU_1 = Dxx(operator, U_1)
        ddU_2 = Dxx(operator, U_2)

        F = alpha * ddU_2 + (beta * (U_1 ** 2 + U_2 ** 2) + nu + gamma_2) * U_2 + (gamma_1 - mu) * U_1

        G = -alpha * ddU_1 - (beta * (U_1 ** 2 + U_2 ** 2) + nu - gamma_2) * U_1 - (gamma_1 + mu) * U_2
    elif eq == 'LLG':
        U_1 = field_slices[0]
        U_2 = field_slices[1]

        nu = parameters[0]
        gamma = parameters[1]
        Gamma = parameters[2]
        delta = parameters[3]
        mu = parameters[4]
        alpha = parameters[5]
        c = parameters[6]

        ddU_1 = Dxx(operator, U_1)
        ddU_2 = Dxx(operator, U_2)

        F = ddU_2 + (nu + gamma * Gamma + (U_1 ** 2 + U_2 ** 2) - 3 * gamma * delta * U_1 * U_2) * U_2 + (gamma - mu + (1 - gamma * alpha) * (U_1 ** 2 + U_2 ** 2) - gamma * delta * U_1 * U_1) * U_1

        G = - ddU_1 - (nu - gamma * Gamma + (U_1 ** 2 + U_2 ** 2) + 3 * gamma * delta * U_1 * U_2) * U_1 - (gamma + mu - (c + gamma * alpha) * (U_1 ** 2 + U_2 ** 2) - gamma * delta * U_2 * U_2) * U_2
    return np.array([F, G])


def equations_FFT(eq, field_slices, parameters, x_grid, kappa):
    if eq == 'PNDLS_forced':
        U_1 = field_slices[0]
        U_2 = field_slices[1]

        alpha = parameters[0]
        beta = parameters[1]
        gamma = parameters[2]
        gamma_1 = gamma[0]
        gamma_2 = gamma[1]
        mu = parameters[3]
        nu = parameters[4]

        Uhat_1 = np.fft.fft(U_1)
        Uhat_2 = np.fft.fft(U_2)
        dd_Uhat_1 = -np.power(kappa, 2) * Uhat_1
        dd_Uhat_2 = -np.power(kappa, 2) * Uhat_2
        U_1 = np.fft.ifft(Uhat_1)
        U_2 = np.fft.ifft(Uhat_2)
        ddU_1 = np.fft.ifft(dd_Uhat_1)
        ddU_2 = np.fft.ifft(dd_Uhat_2)

        F = alpha * ddU_2 + (beta * (U_1 ** 2 + U_2 ** 2) + nu + gamma_2) * U_2 + (gamma_1 - mu) * U_1

        G = -alpha * ddU_1 - (beta * (U_1 ** 2 + U_2 ** 2) + nu - gamma_2) * U_1 - (gamma_1 + mu) * U_2
    elif eq == 'PDNLS':
        U_1 = field_slices[0]
        U_2 = field_slices[1]

        alpha = parameters[0]
        beta = parameters[1]
        gamma_0 = parameters[2]
        mu = parameters[3]
        nu = parameters[4]

        Uhat_1 = np.fft.fft(U_1)
        Uhat_2 = np.fft.fft(U_2)
        dd_Uhat_1 = -np.power(kappa, 2) * Uhat_1
        dd_Uhat_2 = -np.power(kappa, 2) * Uhat_2
        U_1 = np.fft.ifft(Uhat_1)
        U_2 = np.fft.ifft(Uhat_2)
        ddU_1 = np.fft.ifft(dd_Uhat_1)
        ddU_2 = np.fft.ifft(dd_Uhat_2)


        F = alpha * ddU_2 + (beta * (U_1 ** 2 + U_2 ** 2) + nu) * U_2 + (gamma_0 - mu) * U_1

        G = - alpha * ddU_1 - (beta * (U_1 ** 2 + U_2 ** 2) + nu) * U_1 - (gamma_0 + mu) * U_2
    return np.array([F.real, G.real])