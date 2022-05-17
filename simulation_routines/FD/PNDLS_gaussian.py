from back_process import *
from functions import *
from time_integrators import *


if __name__ == '__main__':
    # Midiendo tiempo inicial
    now = datetime.datetime.now()
    print('Hora de Inicio: ' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second))
    time_init = time.time()

    # Definiendo parámetros
    project_name = '/gaussian_15_05'
    disc = 'F:/'
    eq = 'PNDLS_forced'
    for i in range(13):
        print('i = ' + str(i))
        alpha = 1
        beta = 1
        gamma_0 = 0.14 + 0.01 * i
        mu = 0.1
        nu = 0.15
        sigma = 10

        gamma_str = str(int(gamma_0 * 100) * 0.01)
        nu_str = str(int(nu * 100) * 0.01)
        mu_str = str(int(mu * 100) * 0.01)

        print('gamma = ' + gamma_str)
        print('nu = ' + nu_str)
        print('mu = ' + mu_str)

        # Ploteo de Lengua de Arnold
        plot_parameters = 'no'
        if plot_parameters == 'si':
            arnold_tongue_show(gamma_0, mu, nu)

        # Definición de la grilla
        [tmin, tmax, dt] = [0, 1500, 0.005]
        [xmin, xmax, dx] = [-60, 60, 0.1]
        t_grid = np.arange(tmin, tmax + dt, dt)
        x_grid = np.arange(xmin, xmax, dx)
        T = tmax
        Nt = t_grid.shape[0]
        Nx = x_grid.shape[0]

        # Initial Conditions
        U_1 = np.zeros((Nt, Nx))
        U_2 = np.zeros((Nt, Nx))

        U_10 = 0.01 * np.random.rand(Nx)
        U_20 = 0.01 * np.random.rand(Nx)
        U_1[0, :] = U_10
        U_2[0, :] = U_20

        # Empaquetamiento de parametros, campos y derivadas para integración
        fields = np.array([U_1, U_2])
        forcing_real = np.exp(- x_grid ** 2 / (2 * sigma ** 2))
        forcing_img = np.zeros(Nx)
        gamma = gamma_0 * np.array([forcing_real, forcing_img])
        parameters = [alpha, beta, gamma, mu, nu]
        D2 = sparse_DD(Nx, dx)

        # Integración temporal
        final_fields = RK4_complexfields_FD(eq, fields, parameters, x_grid, dt, Nt, D2)

        # Midiendo tiempo final
        now = datetime.datetime.now()
        print('Hora de Término: ' + str(now.hour) + ':' + str(now.minute) + ':' + str(now.second))
        time_fin = time.time()
        print(str(time_fin - time_init) + ' seg')

        # Aligerando campos
        ratio = 10
        U1_light = final_fields[0, 0:-1:ratio, :]
        U2_light = final_fields[1, 0:-1:ratio, :]
        U_complex = U1_light + 1j * U2_light
        t_light = t_grid[0:-1:ratio]

        # Definiendo variables finales
        modulo_light = np.absolute(U_complex)
        arg_light = np.angle(U_complex)
        arg_light = (2*np.pi + arg_light) * (arg_light < 0) + arg_light*(arg_light > 0)
        analytical_signal = hilbert(U1_light[-1, :])
        amplitude_envelope = np.abs(analytical_signal)

        # Guardando datos
        file = disc + 'mnustes_science/simulation_data/FD' + project_name
        subfile = nombre_pndls_gaussian(gamma_0, mu, nu, sigma)
        parameters_np = np.array([alpha, beta, gamma_0, mu, nu])
        if not os.path.exists(file + subfile):
            os.makedirs(file + subfile)
        np.savetxt(file + subfile + '/field_real.txt', U1_light, delimiter=',')
        np.savetxt(file + subfile + '/field_img.txt', U2_light, delimiter=',')
        np.savetxt(file + subfile + '/forcing_real.txt', gamma[0], delimiter=',')
        np.savetxt(file + subfile + '/forcing_img.txt', gamma[1], delimiter=',')
        np.savetxt(file + subfile + '/parameters.txt', parameters_np, delimiter=',')
        np.savetxt(file + subfile + '/final_envelope.txt', amplitude_envelope, delimiter=',')
        np.savetxt(file + subfile + '/X.txt', x_grid, delimiter=',')
        np.savetxt(file + subfile + '/T.txt', t_light, delimiter=',')

        # Gráficos
        plt.plot(x_grid, modulo_light[0, :], label='$R(x, 0)$')
        plt.plot(x_grid, arg_light[0, :], label='$\phi(x, 0)$')
        plt.plot(x_grid, U1_light[0, :], label='$\psi(x, 0)_{R}$')
        plt.plot(x_grid, U2_light[0, :], label='$\psi(x, 0)_{I}$')
        plt.legend()
        plt.grid()
        plt.xlim([x_grid[0], x_grid[-1]])
        plt.savefig(file + subfile + '/initial_conditions.png', dpi=300)
        plt.close()

        pcm = plt.pcolormesh(x_grid, t_light, modulo_light, cmap='jet', shading='auto')
        cbar = plt.colorbar(pcm, shrink=1)
        cbar.set_label('$R(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
        plt.xlim([x_grid[0], x_grid[-1]])
        plt.xlabel('$x$', size='20')
        plt.ylabel('$t$', size='20')
        plt.grid(linestyle='--', alpha=0.5)
        plt.title('$\gamma = $' + str(gamma_str) + '    $\\nu = $' + str(nu_str) + '    $\mu = $' + str(mu_str), size='20')
        plt.savefig(file + subfile + '/module_spacetime.png', dpi=300)
        plt.close()

        pcm = plt.pcolormesh(x_grid, t_light, arg_light, cmap='jet', shading='auto')
        cbar = plt.colorbar(pcm, shrink=1)
        cbar.set_label('$\\varphi(x, t)$', rotation=0, size=20, labelpad=-20, y=1.1)
        plt.xlim([x_grid[0], x_grid[-1]])
        plt.xlabel('$x$', size='20')
        plt.ylabel('$t$', size='20')
        plt.grid(linestyle='--', alpha=0.5)
        plt.title('$\gamma = $' + str(gamma_str) + '    $\\nu = $' + str(nu_str) + '    $\mu = $' + str(mu_str), size='20')
        plt.savefig(file + subfile + '/arg_spacetime.png', dpi=300)
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

        plt.plot(x_grid, gamma[0], c='b', label="$\gamma_{R}(x)$")
        plt.plot(x_grid, gamma[1], c='r', label="$\gamma_{I}(x)$")
        plt.legend(loc="upper right", fontsize=18)
        plt.title('Forcing at $\gamma_0 = $' + str(gamma_0), size='23')
        plt.xlabel('$x$', size='20')
        plt.ylabel('Amplitude', size='20')
        plt.xlim([x_grid[0], x_grid[-1]])
        plt.grid(linestyle='--', alpha=0.5)
        plt.savefig(file + subfile + '/forcing.png')
        plt.close()

        fig = plt.figure()
        plt.title('$\gamma = $' + str(gamma_str) + '    $\\nu = $' + str(nu_str) + '    $\mu = $' + str(mu_str), size='20')
        fig.set_figheight(8)
        fig.set_figwidth(8)
        ax2 = plt.subplot2grid(shape=(4, 4), loc=(1, 0), colspan=4, rowspan=4)
        ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=4)

        ax1.plot(x_grid, forcing_real, c='k')
        ax1.set_xlim([x_grid[0], x_grid[-1]])
        ax1.set_ylabel("Forcing $\gamma (x)$", fontsize=14)
        ax1.grid(alpha=0.5, c='k', linestyle='--')

        ax2.grid(alpha=0.5, c='b', linestyle='--')
        ax2.plot(x_grid, modulo_light[-1, :], c='b', label="$R(x)$")
        ax2.set_xlabel('x', fontsize=20)
        ax2.set_ylabel("Amplitude $R(x)$", fontsize=20)
        ax2.set_xlim([x_grid[0], x_grid[-1]])
        ax2.grid(alpha=0.5)
        ax2b = ax2.twinx()
        ax2b.plot(x_grid, arg_light[-1, :], c='r', label="$\phi(x)$")
        ax2b.set_ylabel("Phase $\phi(x)$", fontsize=20)
        ax2b.set_ylim([0, 2 * np.pi])
        ax2b.grid(alpha=0.5, c='r', linestyle='--')
        plt.savefig(file + subfile + '/final_profile.png', dpi=300)
        plt.close()

        fig = plt.figure()
        fig.suptitle('Final Profile', fontsize=25)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        ax2 = plt.subplot2grid(shape=(4, 4), loc=(1, 0), colspan=4, rowspan=4)
        ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=4)

        ax1.plot(x_grid, forcing_real, c='k')
        ax1.set_xlim([x_grid[0], x_grid[-1]])
        ax1.set_ylabel("Forcing $\gamma (x)$", fontsize=14)
        ax1.grid(alpha=0.5, linestyle='--')

        ax2.plot(x_grid, U1_light[-1, :], c='b', label="$R(x)$")
        ax2.set_xlabel('x', fontsize=20)
        ax2.set_ylabel("Amplitude $R(x)$", fontsize=20)
        ax2.set_xlim([x_grid[0], x_grid[-1]])
        ax2.grid(alpha=0.5, c='b', linestyle='--')
        ax2.plot(x_grid, amplitude_envelope, c='r', label="$\phi(x)$")

        plt.savefig(file + subfile + '/final_envelope.png', dpi=300)
        plt.close()