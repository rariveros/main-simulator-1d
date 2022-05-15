from back_process import *
from functions import *
from time_integrators import *

if __name__ == '__main__':
    disco = 'F'
    initial_dir_data = str(disco) + ':/mnustes_science/simulation_data'
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(parent=root, initialdir=initial_dir_data, title='Elección de carpeta')
    print('Processing ' + str(directory))

    Z_img= np.loadtxt(directory + '/field_img.txt', delimiter=',')
    Z_real = np.loadtxt(directory + '/field_real.txt', delimiter=',')
    T = np.loadtxt(directory + '/T.txt', delimiter=',')
    X = np.loadtxt(directory + '/X.txt', delimiter=',')
    Z_complex = Z_real + 1j * Z_img
    Z_module = np.abs(Z_complex)
    Z_arg = np.angle(Z_complex)

    pcm = plt.pcolormesh(X, T, Z_module, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$R(x, t)$', rotation=0, size=20, labelpad=-27, y=1.1)
    plt.xlim([X[0], X[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(directory + '/module_spacetime.png', dpi=300)
    plt.close()

    pcm = plt.pcolormesh(X, T, Z_arg, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    cbar.set_label('$\\varphi(x, t)$', rotation=0, size=20, labelpad=-20, y=1.1)
    plt.xlim([X[0], X[-1]])
    plt.xlabel('$x$', size='20')
    plt.ylabel('$t$', size='20')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig(directory + '/arg_spacetime.png', dpi=300)
    plt.close()