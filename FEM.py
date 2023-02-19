from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from Integration import gauss_lobatto_jacobi_quadrature1D
"""
u_xx + k^2 * u = 0
u_x(0) = ga, u_x(1) = gb

IMPORTANT Conclusion
For k = 4, the first eligible number of segmentation of errors less than or equal to 0.1 is N = 13.
For k = 8, the first eligible number of segmentation of errors less than or equal to 0.1 is N = 18.
For k = 16, the max number of partitions N = 98 is not enough to achieve errors less than or equal to 0.1.
For k = 40, the max number of partitions N = 98 is not enough to achieve errors less than or equal to 0.1.
For k = 80, the max number of partitions N = 98 is not enough to achieve errors less than or equal to 0.1.
For k = 100, the max number of partitions N = 98 is not enough to achieve errors less than or equal to 0.1.
"""

def Analytical(k):
    u = lambda x: - np.cos(k * x) / (k * np.sin(k))
    u_x = lambda x: np.sin(k * x) / np.sin(k)
    return (u, u_x)

class FEM_Helmholtz():

    def __init__(self, k: float, a: float, b: float, ga: float, gb: float, N: int):
        # Basic Setup
        self.a, self.b, self.ga, self.gb = a, b, ga, gb
        self.k = k  # k (float): Equation coefficient.
        self.N = N  # N (int): Number of discretization points.
        self.h = 1 / N  # length of segmentation
        self.x = np.linspace(0, 1, N + 1)

        # Get basis functions and derivative functions
        self.bases = list(self.phi(n) for n in range(self.N + 1))
        self.bases_der = list(self.phi_x(n) for n in range(self.N + 1))
        self.sol, self.der = None, None

        # Initialize Linear System: Ac = b
        self.A = np.zeros((N + 1, N + 1), dtype=float)
        self.c = None
        self.d = np.zeros(N + 1, dtype=float)

        # Initialize Linear System: Ac = b for FDM
        self.FDM_A = np.zeros((N + 1, N + 1), dtype=float)
        self.FDM_c = None
        self.FDM_d = np.zeros(N + 1, dtype=float)

        # Get the quadrature points
        if self.N * 10 > 1000:
            self.N_quad = 1000
        else:
            self.N_quad = self.N * 10
        self.roots, self.weights = gauss_lobatto_jacobi_quadrature1D(self.N_quad, self.a, self.b)
        self.roots, self.weights = self.roots.numpy(), self.weights.numpy()

    def solve(self):  # To solve the linear system: Ac = d in order to get u(x)
        """
        [- int(Ri_x * Rj_x) + k**2 * int(Ri * Rj)] * u
        = u_x(a) * theta(a) - u_x(b) * theta(b)

        =>

        Matrix_A * c = Vector_d
        Matrix_A = [- int(Ri_x *Rj_x) + k**2 * int(Ri * Rj)]
        c = u
        Vector_d = u_x(a)  * theta(a)      - u_x(b)  * theta(b)
                 = self.ga * phi_j(self.a) + self.gb * phi_j(self.b)
        """
        for i in range(self.N + 1):
            # --- vector d ---
            self.d[i] = self.RHS(i)
            # --- matrix A ---
            self.A[i, i] = self.LHS(i, i)
            if i == 0:
                self.A[i, i + 1] = self.LHS(i, i + 1)
            elif i == self.N:
                self.A[i, i - 1] = self.LHS(i, i - 1)
            else:
                self.A[i, i - 1] = self.LHS(i, i - 1)
                self.A[i, i + 1] = self.LHS(i, i + 1)

        # --- vector c ---
        self.c = np.linalg.solve(self.A, self.d)

        # --- solution u(x) ---
        self.sol = lambda x: np.sum(np.array(
            [self.c[i] * self.phi(i)(x) for i in range(self.N + 1)]
        ), axis=0)

        # --- derivative u_x(x) ---
        self.der = lambda x: np.sum(np.array(
            [self.c[i] * self.phi_x(i)(x) for i in range(self.N + 1)]
        ), axis=0)

    def LHS(self, i: int, j: int): # for matrix A
        if i == j:
            if i == 0 or i == self.N:
                return - 1 / self.h + self.k ** 2 * self.h / 3
            else:
                return - 2 / self.h + self.k ** 2 * self.h * 2 / 3
        elif abs(i - j) == 1:
            return 1 / self.h + self.k ** 2 * self.h / 6
        else:
            return - 0 + self.k ** 2 * 0

    def RHS(self, j): # for vector d
        phi_j = self.bases[j]
        return self.ga * phi_j(self.a) - self.gb * phi_j(self.b)

    def phi(self, i: int) -> Callable:
        x_i = self.x[i]
        x_l = self.x[i - 1] if i != 0 else x_i
        x_r = self.x[i + 1] if i != self.N else x_i
        step = lambda x: np.heaviside(x - x_l, 1) - np.heaviside(x - x_r, 0)
        return lambda x: step(x) * (1 - np.abs(x - x_i) / self.h)

    def phi_x(self, i: int) -> Callable:
        x_i = self.x[i]
        x_l = self.x[i - 1] if i != 0 else x_i
        x_r = self.x[i + 1] if i != self.N else x_i
        step = lambda x: np.heaviside(x - x_l, 1) - np.heaviside(x - x_r, 0)
        if i == 0:
            return lambda x: step(x) * (- 1 / self.h)
        elif i == self.N:
            return lambda x: step(x) * (+ 1 / self.h)
        else:
            return lambda x: step(x) * (- 1 / self.h) * np.sign(x - x_i)

    def error(self, u: Callable) -> Callable:
        u2  = lambda x: abs(u(x) - self.sol(x)) ** 2
        err_u = sum(u2(self.x[i]) for i in range(self.N + 1)) / self.N
        return err_u

    def __call__(self, x: float):
        return self.sol(x), self.der(x)

plt.rcParams["figure.autolayout"] = True
k_list = list(k for k in range(4, 41, 1))
N_list = list(n for n in range(3, 201, 1))
k_show_list = [4, 8, 16, 32, 40]
k_list = k_show_list
N_show_list = [3, 5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200]

H1error, sol_L2error, der_L2error = [], [], []
parent_dir = '/Users/pc/Desktop/FEM_Plot/'

errors_study_k = {}
for N in N_show_list:
    errors_study_k[N] = []
first_eligible_kN = {}

for k in k_list:
    errors_study_N = {}
    time_single, time_cumulated = {}, {}
    time_elapsed = 0

    # making path regards to k
    path_k = {}
    if k in k_show_list:
        directory = 'FEM k = %s/' % int(k)
        path_k[k] = os.path.join(parent_dir, directory)
        if os.path.exists(path_k[k]):
            pass
        else:
            os.mkdir(path_k[k])
        first_eligible_kN[k] = 0

    for N in N_list:
        # basic setting
        time_start = time.process_time()
        a, b = 0, 1
        ga, gb = 0, 1
        solver = FEM_Helmholtz(k=k, a=a, b=b, ga=ga, gb=gb, N=N)

        # Plotting the basis functions
        """if k in k_show_list and N in N_show_list:
            plt.rcParams['figure.figsize'] = [10, 5]
            fig, ax = plt.subplots()
            xpts = np.linspace(a - .05, b + .05, 1000)
            for base in solver.bases:
                ypts = base(xpts)
                ax.plot(xpts, ypts)
            ax.set_ylim([-1, +2])
            ax.set_xlim([a - .5, b + .5])
            ax.grid()
            ax.set(xlabel='x', ylabel='Basis functions')
            plt.savefig(path_k[k] + 'Basis Function of N = %s k = %s' % (N, int(k)))"""

        # Exact Solution
        u = Analytical(k)[0]
        # FEM Solution
        solver.solve()
        sol = solver.sol
        # Errors
        errs = solver.error(u)
        errors_study_N[N] = errs
        if N in N_show_list:
            errors_study_k[N].append([k, errs])

        # plot solution and derivatives
        if k in k_show_list and N in N_show_list:
            xpts = np.linspace(a, b, 200)
            sol_pts = sol(xpts)
            upts = u(xpts)
            sol_pts = (sol_pts - min(sol_pts)) / (max(sol_pts) - min(sol_pts))
            upts = (upts - min(upts)) / (max(upts) - min(upts))
            plt.figure(figsize=(8, 5))
            # solution
            plt.subplot(211)
            plt.title(f'k={round(k)}, N={N}\n')
            plt.plot(xpts, sol_pts, label='FEM solution')
            plt.plot(xpts, upts, label='Analytical solution')
            plt.xlabel('x')
            plt.ylabel(r"$u(x)$")
            plt.legend()
            # errors of solution
            plt.subplot(212)
            plt.plot(xpts, sol_pts - upts, label='Solution Errors')
            plt.xlabel('x')
            plt.ylabel(r"$u(x) - \tilde u(x)$")
            plt.legend()
            plt.savefig(path_k[k] + 'FEM_N = %s_k = %s.png' % (N, int(k)))

            if errs <= 1e-3:
                if first_eligible_kN[k]:
                    pass
                else:
                    first_eligible_kN[k] = N

        # time elapsed
        single_time = time.process_time() - time_start
        time_single[N] = single_time
        time_elapsed += single_time
        time_cumulated[N] = time_elapsed

    if k in k_show_list:
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
        plt.figure(figsize=(6, 4))
        plt.suptitle('Errors of k = %s'%int(k))
        Npts = list(errors_study_N.keys())
        Sol_epts = list(errors_study_N.values())
        plt.plot(Npts, Sol_epts, label='Solution Error')
        plt.legend()
        plt.savefig(path_k[k] + 'Errors of k = %s.png'%int(k))

    print('Passed k = %s' % int(k))

"""for N in N_show_list:
    errors_value = errors_study_k[N]
    kpts = list(k[0] for k in errors_value)
    k_error_pts = list(k[1] for k in errors_value)
    plt.figure(figsize=(6, 4))
    plt.suptitle('Errors of N = %s' % N)
    plt.plot(kpts, k_error_pts, label=['H1 Error', 'Solution L2 Error', 'Derivative L2 Error'])
    plt.legend()
    plt.savefig(parent_dir + '(N,k) Errors' + 'Errors of N = %s.png' % N)
"""
for k in list(first_eligible_kN.keys()):
    if first_eligible_kN[k] == 0:
        print('For k = %s, the max number of partitions N = %s is not enough to achieve errors less than or equal to 0.0001.'%(k, N_list[-1]))
    else:
        print('For k = %s, the first eligible number of segmentation of errors less than or equal to 0.0001 is N = %s.'%(k, first_eligible_kN[k]))