from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from Integration import gauss_lobatto_jacobi_quadrature1D

def Analytical(k):
    u = lambda x: - np.cos(k * x) / (k * np.sin(k))
    u_x = lambda x: np.sin(k * x) / np.sin(k)
    return (u, u_x)

class FDM_Helmholtz():

    def __init__(self, k: float, a: float, b: float, ga: float, gb: float, N: int):
        # Basic Setup
        self.a, self.b, self.ga, self.gb = a, b, ga, gb
        self.k = k  # k (float): Equation coefficient.
        self.N = N  # N (int): Number of discretization points.
        self.h = 1 / N  # length of segmentation
        self.x = np.linspace(0, 1, N + 1)

        # Get basis functions and derivative functions
        self.bases = list(self.phi(n) for n in range(self.N + 1))
        self.sol, self.der = None, None

        # Initialize Linear System: Ac = b
        self.A = np.zeros((N + 1, N + 1), dtype=float)
        self.c = None
        self.d = np.zeros(N + 1, dtype=float)

        # Get the quadrature points
        if self.N * 10 > 1000:
            self.N_quad = 1000
        else:
            self.N_quad = self.N * 10
        self.roots, self.weights = gauss_lobatto_jacobi_quadrature1D(self.N_quad, self.a, self.b)
        self.roots, self.weights = self.roots.numpy(), self.weights.numpy()

    def solve(self):
        for i in range(self.N + 1):
            self.A[i, i] = self.MatrixA(i, i)
            if i == 0:
                self.A[i, i + 1] = self.MatrixA(i, i + 1)
            elif i == self.N:
                self.A[i, i - 1] = self.MatrixA(i, i + 1)
            else:
                self.A[i, i - 1] = self.MatrixA(i, i - 1)
                self.A[i, i + 1] = self.MatrixA(i, i + 1)
        self.d[-1] = - self.h
        try:
            self.c = np.linalg.solve(self.A, self.d)
        except Exception:
            self.c = np.linalg.eig(self.A)[0]

        # --- solution u(x) ---
        self.sol = lambda x: np.sum(np.array(
            [self.c[i] * self.phi(i)(x) for i in range(self.N + 1)]
        ), axis=0)

        return self.sol

    def MatrixA(self, i: int, j: int): # for matrix A
        if i == j:
            if i == 0 or i == self.N:
                return -1 + (self.h**2) * (self.k**2)
            else:
                return -2 + (self.h**2) * (self.k**2)
        elif abs(i - j) == 1:
            return 1
        else:
            return 0

    def phi(self, i: int) -> Callable:
        x_i = self.x[i]
        x_l = self.x[i - 1] if i != 0 else x_i
        x_r = self.x[i + 1] if i != self.N else x_i
        step = lambda x: np.heaviside(x - x_l, 1) - np.heaviside(x - x_r, 0)
        return lambda x: step(x) * (1 - np.abs(x - x_i) / self.h)

    def __call__(self, x: float):
        return self.c

plt.rcParams["figure.autolayout"] = True
k_list = [4, 8, 16, 32, 40, 80]
N_list = list(n for n in range(3, 201, 1))
k_show_list = k_list
N_show_list = [3, 4, 5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 300]

parent_dir = '/Users/pc/Desktop/FEM_Plot/'
errors_studyN = {}
first_eligible_kN = {}
for k in k_list:
    # making path regards to k
    path_k = {}
    time_single, time_cumulated = {}, {}
    time_elapsed = 0

    if k in k_show_list:
        directory = 'FDM k = %s/' % int(k)
        path_k[k] = os.path.join(parent_dir, directory)
        if os.path.exists(path_k[k]):
            pass
        else:
            os.mkdir(path_k[k])
        first_eligible_kN[k] = 0

    for N in N_list:
        time_start = time.process_time()
        a, b = 0, 1
        ga, gb = 0, 1
        solver = FDM_Helmholtz(k=k, a=a, b=b, ga=ga, gb=gb, N=N)

        # Exact Solution
        u, u_x = Analytical(k)[0], Analytical(k)[1]
        # FEM Solution
        solver.solve()
        sol = solver.sol
        errors_list = []
        # plot solution and derivatives
        if k in k_show_list and N in N_show_list:
            xpts = np.linspace(a, b, 100)
            sol_pts = sol(xpts)
            upts = u(xpts)
            sol_pts = (sol_pts-min(sol_pts))/(max(sol_pts) - min(sol_pts))
            upts = (upts - min(upts))/(max(upts)-min(upts))
            plt.figure(figsize=(8, 5))

            plt.subplot(211)
            plt.title(f'k={round(k)}, N={N}\n')
            plt.plot(xpts, sol_pts, label='FDM solution')
            plt.plot(xpts, upts, label='Analytical solution')
            plt.xlabel('x')
            plt.ylabel(r"$u(x)$")
            plt.legend()

            # errors of solution
            plt.subplot(212)
            errors = sol_pts - upts
            plt.plot(xpts, errors, label='Solution Errors')
            plt.xlabel('x')
            plt.ylabel(r"$u(x) - \tilde u(x)$")
            plt.legend()
            plt.savefig(path_k[k] + 'FDM_N = %s_k = %s.png' % (N, int(k)))
            error_res = sum([abs(e)**2 for e in errors]) / len(errors)

            if error_res <= 1e-3:
                if first_eligible_kN[k]:
                    pass
                else:
                    first_eligible_kN[k] = N

        # time elapsed
        single_time = time.process_time() - time_start
        time_single[N] = single_time
        time_elapsed += single_time
        time_cumulated[N] = time_elapsed

        errors_studyN[N] = error_res

    # Plot time elapsed
    plt.figure(figsize=(6, 4))
    plt.suptitle(f'k={round(k)}')
    tpts_single, tpts_cumulated = list(time_single.values()), list(time_cumulated.values())
    plt.subplot(211)
    plt.plot(N_list, tpts_single, label='single time elapsed')
    plt.legend()
    plt.subplot(212)
    plt.plot(N_list, tpts_cumulated, label='cumulated time elapsed')
    plt.legend()
    plt.savefig(path_k[k] + 'Time Elapsed k = %s.png'%int(k))

    # plot errors caused
    npts = list(errors_studyN.keys())
    epts = list(errors_studyN[N] for N in npts)
    plt.figure(figsize = (6, 4))
    plt.plot(npts, epts)
    plt.xlabel('N')
    plt.ylabel('Error')
    plt.title('Errors for k = %s' % k)
    plt.savefig(path_k[k] + 'FDM_k = %s.png'%k)

    print('Passed k = %s' % int(k))

for k in list(first_eligible_kN.keys()):
    if first_eligible_kN[k] == 0:
        print('For k = %s, the max number of partitions N = %s is not enough to achieve errors less than or equal to 0.0001. '%(k, N_list[-1]))
    else:
        print('For k = %s, the first eligible number of segmentation of errors less than or equal to 0.0001 is N = %s.'%(k, first_eligible_kN[k]))