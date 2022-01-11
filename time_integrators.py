from functions import *
from back_process import *


def RK4_complexfields_FD(eq, fields, parameters, x_grid, dt, Nt, operator):
    N_campos = len(fields)
    for i in range(Nt - 1):
        field_slices = []
        for k in range(N_campos):
            field_slices.append(fields[k][i, :])
        k_1 = equations_FD(eq, field_slices, parameters, x_grid, operator)
        k_2 = equations_FD(eq, field_slices + 0.5 * dt * k_1, parameters, x_grid, operator)
        k_3 = equations_FD(eq, field_slices + 0.5 * dt * k_2, parameters, x_grid, operator)
        k_4 = equations_FD(eq, field_slices + dt * k_3, parameters, x_grid, operator)
        fields[:, i + 1, :] = fields[:, i, :] + dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    return fields


def RK4_complexfields_FFT(eq, fields, parameters, x_grid, dt, Nt, kappa):
    N_campos = len(fields)
    for i in range(Nt - 1):
        field_slices = []
        for k in range(N_campos):
            field_slices.append(fields[k][i, :])
        k_1 = equations_FFT(eq, field_slices, parameters, x_grid, kappa)
        k_2 = equations_FFT(eq, field_slices + 0.5 * dt * np.array(k_1), parameters, x_grid, kappa)
        k_3 = equations_FFT(eq, field_slices + 0.5 * dt * np.array(k_2), parameters, x_grid, kappa)
        k_4 = equations_FFT(eq, field_slices + dt * np.array(k_3), parameters, x_grid, kappa)
        fields[:, i + 1, :] = fields[:, i, :] + dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
    return fields