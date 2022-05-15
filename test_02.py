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

    Z_img = np.loadtxt(directory + '/field_img.txt', delimiter=',')
    Z_real = np.loadtxt(directory + '/field_real.txt', delimiter=',')
    forcing_real = np.loadtxt(directory + '/forcing_img.txt', delimiter=',')
    T = np.loadtxt(directory + '/T.txt', delimiter=',')
    X = np.loadtxt(directory + '/X.txt', delimiter=',')
    Z_complex = Z_real + 1j * Z_img
    Z_module = np.abs(Z_complex)
    Z_arg = np.angle(Z_complex)

    fig = plt.figure()
    fig.suptitle('Final Profile', fontsize=25)
    fig.set_figheight(8)
    fig.set_figwidth(8)
    ax2 = plt.subplot2grid(shape=(4, 4), loc=(1, 0), colspan=4, rowspan=4)
    ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0), colspan=4)

    ax1.plot(X, forcing_real, c='k')
    ax1.set_xlim([X[0], X[-1]])
    ax1.set_ylabel("Forcing $\gamma (x)$", fontsize=14)
    ax1.grid(alpha=0.5, linestyle='--')

    ax2.plot(X, Z_module[-1, :], c='b', label="$R(x)$")
    ax2.set_xlabel('x', fontsize=20)
    ax2.set_ylabel("Amplitude $R(x)$", fontsize=20)
    ax2.set_xlim([X[0], X[-1]])
    ax2.grid(alpha=0.5, c='b', linestyle='--')
    ax2b = ax2.twinx()
    ax2b.plot(X, Z_arg[-1, :], c='r', label="$\phi(x)$")
    ax2b.set_ylabel("Phase $\phi(x)$", fontsize=20)
    ax2b.set_ylim([-3.14, 3.14])
    ax2b.grid(alpha=0.5, c='r', linestyle='--')

    plt.savefig(directory + '/final_profile.png', dpi=300)
    plt.show()
    plt.close()