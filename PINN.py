import time
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from test_functions import call_tf
from Integration import gauss_lobatto_jacobi_quadrature1D
import os


E_plot_list = [0, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

parent_dir = '/Users/pc/Desktop/VPINN_Plot/'
tf_type = 'chebyshev'
k = 4
# size of the hidden layers (N x D)
depth = 1 # number D (num of cols)
width = 10 # number N (num of rows)
testfuncnum = 20

# Values
a,  b  = 0, 1
ga, gb = 0, 1
penalty = None

directory = 'VPINN %s D = %s k = %s/' % (tf_type, depth, int(k))
path = os.path.join(parent_dir, directory)
if os.path.exists(path):
    pass
else:
    os.mkdir(path)

def changeType(x, target='Tensor'):
    if type(x).__name__ != target:
        if target == 'Tensor':
            return torch.tensor(x)

def plot_PINN(deriv: Callable, device, a, b, N, u, epoch):
    xpts = torch.linspace(a, b, N * 5).float().view(-1, 1)

    upts = u(xpts.numpy().reshape(-1))  # reshape(-1): # put multidim points into one dim vector
    rpts = deriv(0, xpts.to(device))

    upts = (upts - min(upts)) / (max(upts) - min(upts))
    rpts = (rpts - min(rpts)) / (max(rpts) - min(rpts))

    with torch.no_grad():
        xpts = xpts.numpy().reshape(-1)
        upts = upts.reshape(-1)  # exact solution
        rpts = rpts.cpu().numpy().reshape(-1)  # fem solution

    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.title(f'k={round(k)}, e={epoch}\n')
    plt.plot(xpts, rpts, label='Deep Learning Solution')
    plt.plot(xpts, upts, label='Analytical Solution')
    plt.xlabel('x')
    plt.ylabel(r"$u(x)$")
    plt.legend()
    plt.subplot(212)
    plt.plot(xpts, upts - rpts, label='Errors')
    plt.xlabel('x')
    plt.ylabel(r"$u(x) - \tilde u(x)$")
    plt.legend()
    plt.savefig(path + '/sol - epoch%s.png' % (epoch))

def plot_history(time_epoch_single, time_epoch_cumulated):
    epts = list(time_epoch_single.keys())
    tpts_single = list(time_epoch_single.values())
    tpts_cumulated = list(time_epoch_cumulated.values())
    plt.figure(figsize=(8, 5))
    plt.subplot(211)
    plt.plot(epts, tpts_single, label = 'single time')
    plt.legend()
    plt.subplot(212)
    plt.plot(epts, tpts_cumulated, label = 'cumulated time')
    plt.legend()
    plt.savefig(path + '/history.png')

class VPINN_HelmholtzImpedance(nn.Module):

    def __init__(self, k: float, a: float, b: float, ga: float, gb: float, K, *, N_quad = 80, layers = [1, 10, 1], seed = None, penalty = None):

        if seed: torch.manual_seed(seed) # Ensure reproducibility
        assert layers[-1] == 1

        # Neural network system
        super().__init__()
        self.device = torch.device('cpu')
        self.history = {'epochs': [],
                        'losses': [],
                        'errors': None,
                        }
        self.epoch = 0
        self.time_elapsed = 0
        self.penalty = penalty

        # Store equation parameters
        self.k = changeType(k, 'Tensor').float().to(self.device)
        self.a = changeType(a, 'Tensor').float().view(-1, 1).to(self.device).requires_grad_()
        self.b = changeType(b, 'Tensor').float().view(-1, 1).to(self.device).requires_grad_()
        self.ga = changeType(ga, 'Tensor').float().view(-1, 1).to(self.device).requires_grad_()
        self.gb = changeType(gb, 'Tensor').float().view(-1, 1).to(self.device).requires_grad_()
        self.N_quad = N_quad
        self.K = K

        # Define modules
        self.length = len(layers)      # Number of layers
        self.lins   = nn.ModuleList()  # Linear blocks
        self.drops  = nn.ModuleList()  # Dropout

        # Define the hidden layers: 1 -> 10
        #   nn.Linear: Apply a linear transformation to the incoming data
        #       in_features (int) – size of each input sample
        #       out_features (int) – size of each output sample
        for input, output in zip(layers[0:-2], layers[1:-1]):
            self.lins.append(nn.Linear(input, output, bias=True))

        # Define the output layer: 10 -> 1
        self.lins.append(nn.Linear(layers[-2], layers[-1], bias=True))

        # Initialize weights and biases:
        # Combine Linear Transformation and Activation Function
        #   nn.init.xavier_normal_: Fills the input Tensor with values according to the method
        #       described in Understanding the difficulty of training deep feedforward neural networks
        #       - Glorot, X. & Bengio, Y. (2010), using a normal distribution.
        #   gain: activation function operation, using tanh here.
        #   nn.init.uniform_: Fills the input Tensor with values drawn from the uniform distribution
        for lin in self.lins[:-1]: # self.lins = [(1, 10), (10, 1)] till now
            nn.init.xavier_normal_(lin.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.uniform_(lin.bias, a=int(self.a), b=int(self.b))
        # Weights at output layer
        nn.init.xavier_normal_(self.lins[-1].weight, gain=nn.init.calculate_gain('tanh'))
        # Bias at output layer: Assumed as zero
        nn.init.zeros_(self.lins[-1].bias)

        # Approximate the integral using the Gauss-Jacobi quadrature
        roots, weights = gauss_lobatto_jacobi_quadrature1D(self.N_quad, 0, 1)
        # requires_grad_(): tell autograd to begin recording operations on a tensor.
        roots = roots.float().view(-1, 1).to(self.device).requires_grad_()
        weights = weights.float().view(-1, 1).to(self.device)
        self.quadpoints = (roots, weights)

    def forward(self, x): # forward propagation
        for i, f in zip(range(self.length), self.lins):
            if i == len(self.lins) - 1:
                # Last layer
                x = f(x)
            else:
                # Hidden layers
                x = torch.tanh(f(x))
        return x

    def train_(self, tf_type, K, epochs: int, optimizer, scheduler, exact):
        self.history['errors'] = {'tot': [], 'sol': [], 'der': []}
        self.train()

        epochs += self.epoch - 1 if self.epoch else self.epoch
        time_epochs_single = {}
        time_epochs_cumulated = {}
        Solution_epoch_errors = {}
        Loss_epoch = {}
        for e in range(self.epoch, epochs + 1):
            time_start = time.process_time()
            loss = 0

            for index in range(1, K+1):
                loss += self.loss(index, tf_type, self.quadpoints) / K

            if self.penalty:
                u   = lambda x: self.deriv(0, x)
                u_x = lambda x: self.deriv(1, x)
                loss_ga = self.ga + u_x(self.a) + self.k * u(self.a)
                loss_gb = self.gb + u_x(self.b) + self.k * u(self.b)
                pen = self.penalty / 2 * (loss_ga.pow(2) + loss_gb.pow(2))
                loss += pen[0]

            error_epoch = self.Error(exact[0])
            Error_epoch = error_epoch.detach().numpy()
            Solution_epoch_errors[e] = Error_epoch
            Loss_epoch[e] = loss.item()

            if e in E_plot_list or e == epochs:
                # Calculate minutes elapsed
                cumulated_time = self.time_elapsed
                single_time    = time.process_time() - time_start

                time_epochs_single[e]    = single_time
                time_epochs_cumulated[e] = cumulated_time

                # Store the loss and the error
                self.history['epochs'].append(e)
                self.history['losses'].append(loss.item())
                error = self.Error(exact[0])
                self.history['errors']['sol'].append(error.item())

                # Plot the solution
                plot_PINN(self.deriv, self.device, int(self.a), int(self.b), self.N_quad, exact[0], e)

                # Plot the history
                print('Epoch %s - cumulated time elapsed: %s, singular time elapsed: %s, Loss = %s, H1-error = %s'%
                      (e, round(cumulated_time, 3), round(single_time, 3), "{:4e}".format(loss.item(), 3), "{:4e}".format(error.item(), 3)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.time_elapsed += time.process_time() - time_start
            self.epoch += 1
        plot_history(time_epochs_single, time_epochs_cumulated)

        plt.figure(figsize=(6, 3))
        plt.suptitle('Errors of k = %s' % int(self.k))

        epoch_pts = list(Solution_epoch_errors.keys())
        sol_error_pts = [Solution_epoch_errors[e] for e in epoch_pts]
        plt.plot(epoch_pts, sol_error_pts, label='Solution Error')
        plt.legend()
        plt.savefig(path + '/errors.png')

        plt.figure(figsize=(6, 3))
        plt.suptitle('Loss of k = %s' % int(self.k))
        epoch_pts = list(Loss_epoch.keys())
        Loss_pts = [Loss_epoch[e] for e in epoch_pts]
        plt.plot(epoch_pts, Loss_pts, label='Loss')
        plt.legend()
        plt.savefig(path + '/loss.png')

    def intg(self, func: Callable, quadpoints: tuple = None) -> float:
        if not quadpoints: quadpoints = self.quadpoints
        roots, weights = quadpoints
        return torch.sum(func(roots) * weights) * (roots[-1] - roots[0]) / 2

    def deriv(self, n: int, x: torch.tensor):  # To compute f_x and f_xx derivatives
        # n: order of derivative
        if n not in [0, 1, 2]:  # constrain on at most 2nd order derivatives
            raise ValueError('n = {%s} is not a valid derivative.' % n)
        f = self(x)
        # autograd: implement automatic differentiation of arbitrary scalar valued functions.
        # autograd.grad: Computes and returns the sum of gradients of outputs with respect to the inputs.
        if n >= 1:
            grad = torch.ones(f.size(), dtype = f.dtype, device = f.device)
            f_x = torch.autograd.grad(f, x, grad_outputs=grad, create_graph=True, allow_unused=False)[0]
        if n >= 2:
            grad = torch.ones(f_x.size(), dtype=f.dtype, device=f.device)
            f_xx = torch.autograd.grad(f_x, x, grad_outputs=grad, create_graph=True, allow_unused=False)[0]
        if n == 0:
            return f.view(-1, 1)
        elif n == 1:
            return f_x
        elif n == 2:
            return f_xx

    def loss(self, index, tf_type, quadpoints):
        phi_i   = lambda x: self.deriv(0, x)
        phi_i_x = lambda x: self.deriv(1, x)
        phi_j   = lambda x: call_tf(x, index, 0, tf_type)
        phi_j_x = lambda x: call_tf(x, index, 1, tf_type)

        # LHS = [- int(Ri_x * Rj_x) + k**2 * int(Ri * Rj)] * u
        LHS = - self.intg(lambda x: phi_i_x(x) * phi_j_x(x), quadpoints) \
              + self.k.pow(2) * self.intg(lambda x: phi_i(x) * phi_j(x), quadpoints)

        # RHS = [u_x(a) * theta(a) - u_x(b) * theta(b)]
        RHS = self.ga * phi_j(self.a) + self.gb * phi_j(self.b)

        return (LHS - RHS).pow(2)

    def Error(self, phi_i: Callable) -> float:
        def sol(x):
            sol = phi_i(x.detach().view(-1).cpu().numpy())
            return torch.Tensor([sol]).T.to(self.device)
        phi_i2   = lambda x: (sol(x) - self(x)).pow(2).sum(axis=1)
        self.eval()
        err_phi_i   = torch.sqrt(self.intg(phi_i2))
        return err_phi_i

    def __len__(self):
        return self.length - 1  # Number of hidden layers / depth

def main(seed, k, epochs, lr, tf_type):
    # Exact solution
    u   = lambda x: - np.cos(k * x) / (k * np.sin(k))
    u_x = lambda x: np.sin(k * x) / np.sin(k)

    # Model
    model = VPINN_HelmholtzImpedance(k = k, a = a, b = b, ga = ga, gb = gb, K = testfuncnum,
                                     layers = [1] + [width for _ in range(depth)] + [1],
                                     seed = seed, penalty = penalty)

    # Least-squares initialization
    if depth == 1:
        # Create a new model for using it's lhsrhs method
        md = VPINN_HelmholtzImpedance(k = k, a = a, b = b, ga = ga, gb = gb, K = testfuncnum,
                                     layers = [1] + [width for _ in range(depth)] + [1],
                                     seed = seed, penalty = penalty)

        # Initialize the first layer of both models
        weights = torch.normal(mean=(k ** .80), std=(k ** .5), size=(width, 1))
        biases = -weights[:, 0] * torch.linspace(a, b, width).float()
        md.lins[-2].weight = nn.Parameter(weights)
        md.lins[-2].bias = nn.Parameter(biases)
        model.lins[-2].weight = nn.Parameter(weights)
        model.lins[-2].bias = nn.Parameter(biases)

        A = np.zeros((testfuncnum, width), dtype=float)
        for col in range(width):
            cs = torch.zeros_like(md.lins[-1].weight)
            cs[:, col] = 1.
            md.lins[-1].weight = nn.Parameter(cs)
            md.lins[-1].bias = nn.Parameter(torch.zeros_like(md.lins[-1].bias))
            for row in range(1, testfuncnum+1):
                residual = md.loss(row, tf_type, md.quadpoints)
                A[row-1, col] = residual.item()

        D = np.zeros(testfuncnum, dtype=float)
        for row in range(1, testfuncnum+1):
            residual = md.loss(row, tf_type, md.quadpoints)
            D[row-1] = residual.item()

        c = np.linalg.lstsq(A, D, rcond=None)[0]
        carray = np.array([c])
        cs = torch.tensor(carray).float().to(model.device)
        model.lins[-1].weight = nn.Parameter(cs)
        model.lins[-1].bias = nn.Parameter(torch.zeros_like(model.lins[-1].bias))

    stages = []
    while True:
        try:
            # milestones, gamma: used for optimizing the learning rates
            #   milestone: list of epoch indices. Must be increasing
            #   gamma: multiplicative factor of learning rate decay. Default: 0.1.
            milestones, gamma = [], 0.1
            stages.append({
                'epochs': epochs,
                'lr': lr,
                'milestones': milestones,
                'gamma': gamma,
            })

            # optimizing the weights and biases
            optimizer_params = []
            for lin in model.lins[:-1]:
                optimizer_params.append({'params': lin.weight, 'lr': lr})
                optimizer_params.append({'params': lin.bias,   'lr': lr})
            optimizer_params.append({'params': model.lins[-1].weight, 'lr': lr})
            optimizer_params.append({'params': model.lins[-1].bias,   'lr': lr})

            optimizer = torch.optim.Adam(
                optimizer_params,
                )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma, last_epoch=-1)

            model.train_(tf_type, testfuncnum, epochs, optimizer, scheduler, exact=(u, u_x))
        except KeyboardInterrupt:
            pass

        print('Yes PASS >-<')
        break

if __name__ == '__main__':
    seed = None
    k = k
    epochs = 20000 +1
    lr = 1e-03
    tf_type = tf_type
    main(seed, k, epochs, lr, tf_type)