from back_process import *
from functions import *
from time_integrators import *

if __name__ == '__main__':
    # Midiendo tiempo inicial
    now = datetime.datetime.now()
    print('Hora de Inicio: ' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second))
    time_init = time.time()

    # Definiendo parámetros
    eq = 'PDNLS'
    alpha = 1
    beta = 1
    gamma_0 = 1.1
    mu = 0.45
    nu = 1
    sigma = 16

    # Ploteo de Lengua de Arnold
    plot_parameters = 'si'
    if plot_parameters == 'si':
        arnold_tongue_show(gamma_0, mu, nu)


    # Definición de la grilla
    [tmin, tmax, dt] = [0, 1000, 0.001]
    [xmin, xmax, dx] = [-8, 8, 0.1]
    t_grid = np.arange(tmin, tmax + dt, dt)
    x_grid = np.arange(xmin, xmax, dx)
    T = tmax
    Nt = t_grid.shape[0]
    Nx = x_grid.shape[0]
    kappa = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    print(Nx)
    print(Nt)

    # Initial Conditions
    U_1 = np.zeros((Nt, Nx))
    U_2 = np.zeros((Nt, Nx))

    U_10 = 0.01 * np.random.rand(Nx)
    U_20 = 0.01 * np.random.rand(Nx)
    U_1[0, :] = U_10
    U_2[0, :] = U_20

    # Empaquetamiento de parametros, campos y derivadas para integración
    fields = np.array([U_1, U_2])
    parameters = [alpha, beta, gamma_0, mu, nu]

    # Integración temporal
    final_fields = RK4_complexfields_FFT(eq, fields, parameters, x_grid, dt, Nt, kappa)

    # Midiendo tiempo final
    now = datetime.datetime.now()
    print('Hora de Término: ' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second))
    time_fin = time.time()
    print(str(time_fin - time_init) + ' seg')

    # Aligerando campos
    ratio = 10
    U1_light = final_fields[0, 0:-1:ratio, :]
    U2_light = final_fields[1, 0:-1:ratio, :]
    t_light = t_grid[0:-1:ratio]

    # Definiendo variables finales
    modulo_light = np.power(np.power(U1_light, 2) + np.power(U2_light, 2), 0.5)
    arg_light = np.arctan2(U1_light, U2_light)

    # Guardando datos
    file = 'F:/mnustes_science/simulation_data/FFT/28_01_2022/homogeneous'
    subfile = nombre_pndls_gaussian(gamma_0, mu, nu, sigma)
    parameters_np = np.array([alpha, beta, gamma_0, mu, nu])
    if not os.path.exists(file + subfile):
        os.makedirs(file + subfile)
    np.savetxt(file + subfile + '/field_real.txt', U1_light, delimiter=',')
    np.savetxt(file + subfile + '/field_img.txt', U2_light, delimiter=',')
    np.savetxt(file + subfile + '/X.txt', x_grid, delimiter=',')
    np.savetxt(file + subfile + '/T.txt', t_light, delimiter=',')
    np.savetxt(file + subfile + '/parameters.txt', parameters_np, delimiter=',')

    # Gráficos
    pcm = plt.pcolormesh(x_grid, t_light, modulo_light, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$R(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([x_grid[0], x_grid[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(file + subfile + '/module_spacetime.png')
    plt.close()

    pcm = plt.pcolormesh(x_grid, t_light, arg_light, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\\varphi(x, t)$', rotation=0, size=20, labelpad=-20, y=1.1)
    plt.xlim([x_grid[0], x_grid[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(file + subfile + '/arg_spacetime.png')
    plt.close()

    nu_positive_grid = np.arange(0, 2, 0.01)
    nu_negative_grid = - np.flip(nu_positive_grid)
    nu_grid = np.append(nu_negative_grid, nu_positive_grid)
    plt.plot(nu_positive_grid, np.sqrt(nu_positive_grid ** 2 + mu ** 2), c='k', linestyle='--')
    plt.fill_between(nu_positive_grid, np.ones(len(nu_positive_grid)) * mu, np.sqrt(nu_positive_grid ** 2 + mu ** 2),
                     facecolor=(92 / 255, 43 / 255, 228 / 255, 0.4))
    plt.plot(nu_negative_grid, np.sqrt(nu_negative_grid ** 2 + mu ** 2), c='k', linestyle='--')
    plt.fill_between(nu_negative_grid, np.ones(len(nu_negative_grid)) * mu, np.sqrt(nu_negative_grid ** 2 + mu ** 2),
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
    plt.savefig(file + subfile + '/arnold_tongue.png')
    plt.close()