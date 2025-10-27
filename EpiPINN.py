import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



"""EpiPINN.py

Catalog:
- class PINN: Base neural network class.
- function toeplitz: Generate Toeplitz matrix for efficient Caputo differentiation.
- function torch_caputo_l1_diff: Numerical Caputo differentiation of Pytorch time series tensor.
- function train_stage1: Stage 1 of PINN training, modeled after Zinhi paper.
- function train_stage2: Stage 2 of PINN training, modeled after Zinhi paper.
"""



class PINN(nn.Module):
    """Base neural network class used by EpiPINN class"""
    
    def __init__(self, hidden_size, depth):
        """Constructor for sequential neural network for the SEIRD model.

        This constructs a sequential neural network with a certain number of layers and a fixed number of perceptrons per layer. Weights can be initialized using Xavier Uniform initialization. There is one input dimension for time, and five output dimensions for S, E, I, R, and D components, respectively. Tanh activation functions are used between hidden layers, and Softmax is applied to the output to enforce population conservation.
        
        Arguments:
            hidden_size (int): Number of perceptrons per layer.
            depth (int): Number of layers.
        
        Returns:
            None
        """
        super().__init__()
        # input t
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        # Output layer with 5 units for (s, e, i, r, d)
        layers.append(nn.Linear(hidden_size, 5))
        # Add softmax to enforce all components are positive and sum to 1
        layers.append(nn.Softmax(dim=1))
        
        self.net = nn.Sequential(*layers)

        self.init_weights() # Call the initialization method

    def init_weights(self):
        """Applies Xavier Uniform initialization."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear): # only initialize linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: # initialize biases to zero
                    nn.init.constant_(module.bias, 0)

    def forward(self, t):
        """Applies the neural network to time points t.

        Arguments:
            t (torch.Tensor): Input data, i.e. n values inside a (n, 1) tensor.

        Returns:
            torch.Tensor of dimension (n, 5) containing the corresponing outputs from the neural network function.
        """
        return self.net(t)



class EpiPINN(nn.Module):
    """Outer class containing a sequential neural network, epidemiological parameters, and loss functions."""
    
    def __init__(self, hidden_size, depth, initial_params):
        """Constructor for a SEIRD PINN.

        Arguments:
            hidden_size (int): Number of perceptrons per layer.
            depth (int): Number of layers.
            initial_params (dict): Dictionary containing initial guesses for epidemiological parameters.

        Returns:
            None
        """
        super().__init__()
        
        self.pinn = PINN(hidden_size, depth) 
        # trainable params
        # x + torch.log(-torch.expm1(-x))
        self.raw_beta = nn.Parameter(torch.tensor([initial_params['beta'] + np.log(-np.expm1(-initial_params['beta']))], dtype=torch.float32))
        self.raw_sigma = nn.Parameter(torch.tensor([initial_params['sigma'] + np.log(-np.expm1(-initial_params['sigma']))], dtype=torch.float32))
        self.raw_gamma = nn.Parameter(torch.tensor([initial_params['gamma'] + np.log(-np.expm1(-initial_params['gamma']))], dtype=torch.float32))
        self.raw_mu = nn.Parameter(torch.tensor([initial_params['mu'] + np.log(-np.expm1(-initial_params['mu']))], dtype=torch.float32))
        # Init z_alpha such that the init alpha is close to 1.0
        self.z_alpha = nn.Parameter(torch.tensor([initial_params['z_alpha']], dtype=torch.float32)) # sigmoid(2.94) is approx 0.95
        
        self.min_alpha = initial_params['min_alpha'] # Example minimum value for alpha
        self.dt = initial_params['dt']

    def beta(self):
        """Return susceptible-to-exposed rate beta. Applies softplus to raw stored value."""
        return nn.functional.softplus(self.raw_beta)

    def sigma(self):
        """Return exposed-to-infected rate sigma. Applies softplus to raw stored value."""
        return nn.functional.softplus(self.raw_sigma)

    def gamma(self):
        """Return infected-to-recovered rate gamma. Applies softplus to raw stored value."""
        return nn.functional.softplus(self.raw_gamma)
        
    def mu(self):
        """Return infected-to-dead rate beta. Applies softplus to raw stored value."""
        return nn.functional.softplus(self.raw_mu)

    def alpha(self):
        """Return fractional derivative order alpha. Applies rescaled sigmoid to raw stored value."""
        # Restrict alpha to a specific range, (min_alpha, 1.0] 
        return self.min_alpha + (1.0 - self.min_alpha) * torch.sigmoid(self.z_alpha)
    
    def forward(self, t):
        """Applies the neural network to time points t.

        Arguments:
            t (torch.Tensor): Input data, i.e. n values inside a (n, 1) tensor.

        Returns:
            torch.Tensor of dimension (n, 5) containing the corresponing outputs from the neural network function.
        """
        return self.pinn(t)

    def get_loss_ic(self, ts, ic, y_pred=None):
        """Get initial condition loss of model.

        Arguments:
            ts (torch.tensor): Time points for the model, assuming t[0] is the initial time. Only t[0] is needed, but the tensor dimensions must be consistent.
            ic (torch.tensor): The initial state to enforce.
            y_pred=None (torch.tensor): Predictions at time points, if already computed. Only y[0] is needed, but the tensor dimensions must be consistent.

        Returns:
            squared l2 norm of initial condition error
        """
        # IC loss
        t_initial = ts[0].unsqueeze(0) # get t_0
        y_initial_pred = self.forward(t_initial) if y_pred == None else y_pred[0, :].unsqueeze(0)
        return nn.functional.mse_loss(y_initial_pred, ic)

    def get_loss_data(self, t_data, y_data, y_data_pred=None):
        """Get data loss of model.

        Arguments:
            t_data (torch.tensor): Training data times.
            y_data (torch.tensor): Corresponding state data.
            y_data_pred=None (torch.tensor): Predictions at data points, if already computed.

        Returns:
            MSE loss (squared l2 norm) of data error
        """
        # Data Loss
        y_data_pred = self.forward(t_data) if y_data_pred == None else y_data_pred
        return nn.functional.mse_loss(y_data_pred, y_data)

    def get_loss_phys(self, t_colloc, y_colloc_pred=None):
        """Get physics loss of model.

        Arguments:
            t_colloc (torch.tensor): Times at which to compute the loss.
            y_colloc_pred=None (torch.tensor): Predictions at collocation points, if already computed.

        Returns:
            squared l2 norm of residual
        """
        # Phys Loss
        y_colloc_pred = self.forward(t_colloc)
        s,e,i,r,d = y_colloc_pred.unbind(1)
        s = s.unsqueeze(1)
        e = e.unsqueeze(1)
        i = i.unsqueeze(1)
        r = r.unsqueeze(1)
        d = d.unsqueeze(1)
        ds_dt = torch_caputo_l1_diff(s, self.alpha(), self.dt)
        de_dt = torch_caputo_l1_diff(e, self.alpha(), self.dt)
        di_dt = torch_caputo_l1_diff(i, self.alpha(), self.dt)
        dr_dt = torch_caputo_l1_diff(r, self.alpha(), self.dt)
        dd_dt = torch_caputo_l1_diff(d, self.alpha(), self.dt)

        # calculate RHS of equation 4
        num_living =  1 - d
        f_s = -self.beta() * s * i / num_living
        f_e = (self.beta() * s * i / num_living) - self.sigma() * e
        f_i = (self.sigma() * e) - (self.gamma()+ self.mu()) * i
        f_r = self.gamma() * i
        f_d = self.mu() * i

        # calc residuals (LHS - RHS = 0)
        residual_s = ds_dt - f_s
        residual_e = de_dt - f_e
        residual_i = di_dt - f_i
        residual_r = dr_dt - f_r
        residual_d = dd_dt - f_d

        all_residuals = torch.cat([residual_s, residual_e, residual_i, residual_r, residual_d], dim=1)
        loss_phys = torch.mean(all_residuals**2)
        return loss_phys



# Credit Google Gemini for help developing this efficient Toeplitz matrix generation function
def toeplitz(c, r):
    """
    Creates a Toeplitz matrix from a given first column (c) and first row (r).

    Args:
        c (torch.Tensor): A 1D tensor representing the first column of the Toeplitz matrix.
        r (torch.Tensor): A 1D tensor representing the first row of the Toeplitz matrix.
                          The first element of r should be equal to the first element of c.

    Returns:
        torch.Tensor: The constructed Toeplitz matrix.
    """
    if c[0] != r[0]:
        raise ValueError("The first element of 'c' and 'r' must be the same.")

    vals = torch.cat((r, c[1:].flip(0)))
    shape = len(c), len(r)
    i, j = torch.ones(*shape).nonzero().T
    return vals[j - i].reshape(*shape)



def torch_caputo_l1_diff(psis, alpha, dt=1.0, ts=None):
    """Compute approximate Caputo fractional derivative using the L1 scheme.

    Args:
        psi (torch.tensor): A series of function values. In agreement with pytorch machine learning conventions, this tensor is two-dimensional, and the first dimension corresponds to the dependent variable.
        alpha (torch.tensor): The order of the derivative, in the range (0.0, 1.0].
        dt=1.0 (float): If the corresponding dependent variable series is not provided, the uniform step size between function evaluations.
        ts=None (np.ndarray): The corresponding dependent variable series, used if there is a variable step size.

    Returns:
        An torch.tensor containing the series of fractional derivatives.
    """
    psips = torch.zeros(psis.shape, dtype=psis.dtype, device=psis.device)
    C = dt ** (-alpha) / torch.exp(torch.lgamma(2.0 - alpha))
    dpsis = psis[1:, :] - psis[:-1, :]
    if ts is None:
        rs = torch.arange(psis.shape[0] - 1, dtype=psis.dtype, device=psis.device)
        ws = (rs + 1) ** (1.0 - alpha) - rs ** (1.0 - alpha)
        row = torch.zeros(psis.shape[0] - 1, dtype=psis.dtype, device=psis.device)
        row[0] = ws[0] # Lower triangular matrix
        A = toeplitz(ws, row)
        psips[1:, :] = C * A @ dpsis
    else:
        raise NotImplementedError('Variable time step differentiation not yet implemented!')
    return psips



def train_stage1(model, ts, ys, t_colloc, ic, optimizer, epochs=1000, patience=500, pr=0):
    """Stage one of EpiPINN training process.

    In this stage, only the weights of the neural network are trained to minimize the data and initial condition loss.

    Arguments:
        model (EpiPINN): An instantiated fractional SEIRD model to train.
        ts (torch.tensor): Time values for time-series data.
        ys (torch.tensor): States (s, e, i, r, d) for time-series data.
        optimizer: Pytorch training optimizer.
        epochs=1000 (Int): How many epochs to perform gradient descent.
        patience=500 (int): After how many epochs to terminate training, if the loss doesn't improve.
        pr=0 (Int): Print progress every pr epochs. If 0, nothing is printed.

    Returns:
        losses, losses_data, losses_ic, losses_phys
    """
    losses = []
    losses_data = []
    losses_ic = []
    losses_phys = []

    # Ensure epidemiological parameters are not trained
    model.raw_beta.requires_grad = False
    model.raw_sigma.requires_grad = False
    model.raw_gamma.requires_grad = False
    model.raw_mu.requires_grad = False
    model.z_alpha.requires_grad = False

    # Set to training mode
    model.train()

    # Train with early stopping
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        predictions = model(ts)
        
        # Compute losses separately, then combine
        loss_data = model.get_loss_data(ts, ys, y_data_pred=predictions)
        loss_ic = model.get_loss_ic(t_colloc, ic, y_pred=predictions)
        loss_phys = model.get_loss_phys(t_colloc)
        loss = loss_data + loss_ic # Physics is not used for gradient descent

        # Record losses
        losses.append(loss.item() + loss_phys.item()) # Physics is recored, not mimimized
        losses_data.append(loss_data.item())
        losses_ic.append(loss_ic.item())
        losses_phys.append(loss_phys.item())

        # Adjust weights to minimize loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > patience or loss.item() < 1e-6:
            
            break

        # Print progress if desired
        if pr != 0 and (epoch + 1) % pr == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}, Patience: {patience_counter}')
    
    # Set to evaluation mode
    model.eval()
        
    return losses, losses_data, losses_ic, losses_phys



def train_stage2(model, ts, ys, t_colloc, ic, optimizer, epochs=1000, patience=500, weight_ic_phys=1e6, pr=0):
    """Stage two of EpiPINN training process.

    In this stage, both the weights of the neural network and epidemiological parameters are trained to minimize the data, initial condition, and physics losses.

    Arguments:
        model (EpiPINN): An instantiated fractional SEIRD model to train.
        ts (torch.tensor): Time values for time-series data.
        ys (torch.tensor): States (s, e, i, r, d) for time-series data.
        optimizer: Pytorch training optimizer.
        epochs=1000 (int): How many epochs to perform gradient descent.
        patience=500 (int): After how many epochs to terminate training, if the loss doesn't improve.
        weight_ic_phys=1e6 (float): By how much to scale initial condition and physics loss in the total loss, causing training to favor minimizing these. 
        pr=0 (int): Print progress every pr epochs. If 0, nothing is printed.

    Returns:
        losses, losses_data, losses_ic, losses_phys, alphas, betas, sigmas, gammas, mus
    """
    losses = []
    losses_data = []
    losses_ic = []
    losses_phys = []
    alphas = []
    betas = []
    sigmas = []
    gammas = []
    mus = []

    # Ensure epidemiological parameters are trained
    model.raw_beta.requires_grad = True
    model.raw_sigma.requires_grad = True
    model.raw_gamma.requires_grad = True
    model.raw_mu.requires_grad = True
    model.z_alpha.requires_grad = True

    # Set to training mode
    model.train()

    # Train with early stopping
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        def closure():
            predictions = model(ts)
            
            # Compute losses separately, then combine
            loss_data = model.get_loss_data(ts, ys, y_data_pred=predictions)
            loss_ic = model.get_loss_ic(t_colloc, ic, y_pred=predictions)
            loss_phys = model.get_loss_phys(t_colloc)
        
            loss = loss_data + weight_ic_phys * (loss_ic + loss_phys)
    
            # Record losses
            losses.append(loss.item())
            losses_data.append(loss_data.item())
            losses_ic.append(loss_ic.item())
            losses_phys.append(loss_phys.item())
    
            # Record epidemiological parameters
            alphas.append(model.alpha().item())
            betas.append(model.beta().item())
            sigmas.append(model.sigma().item())
            gammas.append(model.gamma().item())
            mus.append(model.mu().item())
    
            # Adjust weights to minimize loss
            optimizer.zero_grad()
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Early stopping
        if losses[-1] < best_loss:
            best_loss = losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > 500 or losses[-1] < 1e-6:
            break

        # Print progress if desired
        if pr != 0 and (epoch + 1) % pr == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {losses[-1]:.6f}, Patience: {patience_counter}')


    # Set to evaluation mode
    model.eval()
    
    return losses, losses_data, losses_ic, losses_phys, alphas, betas, sigmas, gammas, mus