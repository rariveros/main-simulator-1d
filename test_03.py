from back_process import *
from functions import *
from time_integrators import *

if __name__ == '__main__':
    [tmin, tmax, dt] = [0, 30, 0.01]
    [xmin, xmax, dx] = [-1, 1, 0.001]
    t_grid = np.arange(tmin, tmax + dt, dt)
    x_grid = np.arange(xmin, xmax, dx)
    T = tmax
    Nt = t_grid.shape[0]
    Nx = x_grid.shape[0]
    print(Nx)
    print(Nt)

    u = np.zeros((Nt, Nx))

    k = 10
    omega_01 = 0
    omega_02 = 2
    delta_0 = 0.005
    sigma = 0.4
    A_0 = 1
    envelope = A_0 #* np.exp(- x_grid ** 2 / (2 *sigma ** 2))
    # Initial Conditions
    for i in range(Nt):
        delta_i = delta_0 * np.cos(np.pi * k * x_grid + omega_01 * t_grid[i]) / np.abs(np.cos(np.pi * k * x_grid + omega_01 * t_grid[i]))
        zeta_i = delta_i * np.sin(omega_02 * t_grid[i])
        u[i, :] = envelope * np.cos(np.pi * k * (x_grid + zeta_i) + omega_01 * t_grid[i])

    pcm = plt.pcolormesh(x_grid, t_grid, u, cmap='RdBu', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$u(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([x_grid[0], x_grid[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    plt.close()
