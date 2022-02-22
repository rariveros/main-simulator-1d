from back_process import *
from functions import *
from time_integrators import *

if __name__ == '__main__':
    D2 = sparse_DD(10, 0.1)
    pcm = plt.pcolormesh(D2, 10, 10, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, shrink=1)
    plt.show()