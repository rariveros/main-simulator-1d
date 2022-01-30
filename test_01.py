from back_process import *
from functions import *
from time_integrators import *

if __name__ == '__main__':
    [xmin, xmax, dx] = [-40, 40, 0.1]
    length = 20
    x_grid = np.arange(xmin, xmax, dx)
    forcing_real = []
    for i in range(len(x_grid)):
        if - length / 2 < x_grid[i] < - length / 4:
            forcing_real_i = (x_grid[i] + length / 2) / (- length / 4 + length / 2)
        elif - length / 4 < x_grid[i] < 0:
            forcing_real_i = (x_grid[i] - 0) / (- length / 4 + 0)
        elif 0 < x_grid[i] < length / 4:
            forcing_real_i = -(x_grid[i] + 0) / (- length / 4 + 0)
        elif length / 4 < x_grid[i] < length / 2:
            forcing_real_i = (x_grid[i] - length / 2) / (length / 4 - length / 2)
        else:
            forcing_real_i = 0
        forcing_real.append(forcing_real_i)
    forcing_real = np.array(forcing_real)
    plt.plot(forcing_real)
    plt.title("Triangular window")
    plt.ylabel("Amplitude")
    plt.xlabel("Sample")
    plt.show()