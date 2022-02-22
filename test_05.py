from back_process import *
from functions import *
from time_integrators import *

if __name__ == '__main__':
    [tmin, tmax, dt] = [0, 60, 0.01]
    [xmin, xmax, dx] = [-1.5, 1.5, 0.001]
    t_grid = np.arange(tmin, tmax + dt, dt)
    x_grid = np.arange(xmin, xmax, dx)
    T = tmax
    Nt = t_grid.shape[0]
    Nx = x_grid.shape[0]
    print(Nx)
    print(Nt)

    u_real = np.zeros((Nt, Nx))
    u_img = np.zeros((Nt, Nx))

    k = 10
    q = 0.1
    omega_01 = 0.2
    omega_02 = 1
    A_r = 1
    A_i = 0
    delta_0 = 0.01
    sigma = 0.4
    envelope = np.exp(- x_grid ** 2 / (2 *sigma ** 2))
    # Initial Conditions
    for i in range(Nt):
        u_real[i, :] = A_r * envelope * np.cos(np.pi * k * (x_grid + delta_0 * np.sin(omega_02 * t_grid[i])))
        u_img[i, :] = A_i * envelope * np.sin(np.pi * k * (x_grid + delta_0 * np.sin(omega_02 * t_grid[i])))

    pcm = plt.pcolormesh(x_grid, t_grid, np.sqrt(u_real ** 2 + u_img ** 2), cmap='jet', shading='auto')
    #pcm = plt.pcolormesh(x_grid, t_grid, u_real, cmap='RdBu', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$|u|$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([x_grid[0], x_grid[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
    plt.close()