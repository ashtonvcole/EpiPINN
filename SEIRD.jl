using FdeSolver
using Plots

# Inputs
E0 = 0.01;             # intial value of exposed
tSpan = [0, 1000];       # [intial time, final time]
y0 = [1 - E0, E0, 0, 0, 0];   # initial values [S0,I0,R0]
alpha = [0.9, 0.9, 0.9, 0.9, 0.9];          # order of derivatives
h = 5;                # step size of computation (default = 0.01)
par = [0.25, 0.13, 0.052, 0.005];      # parameters

## ODE model
function F(t, y, par)

    # parameters
    beta = par[1]     # exposure rate
    sigma = par[2]    # incubation rate
    gamma = par[3]    # recovery rate
    mu = par[4]       # death rate

    S = y[1]   # Susceptible
    E = y[2]   # Exposed
    I = y[3]   # Infectious
    R = y[4]   # Recovered
    D = y[5]   # Dead

    # System equation
    dSdt = - beta .* S .* I ./ (1 .- D)
    dEdt = beta .* S .* I ./ (1 .- D) .- sigma .* E
    dIdt = sigma .* E .- (gamma + mu) .* I
    dRdt = gamma .* I
    dDdt = mu .* I

    return [dSdt, dEdt, dIdt, dRdt, dDdt]

end

## Jacobian of ODE system
function JacobF(t, y, par)

    # parameters
    beta = par[1]     # exposure rate
    sigma = par[2]    # incubation rate
    gamma = par[3]    # recovery rate
    mu = par[4]       # death rate

    S = y[1]   # Susceptible
    E = y[2]   # Exposed
    I = y[3]   # Infectious
    R = y[4]   # Recovered
    D = y[5]   # Dead

    # System equation
    J11 = -beta * I / (1 - D)
    J12 = 0
    J13 = -beta * S / (1 - D)
    J14 = 0
    J15 = -beta * S * I / (1 - D)^2
    J21 = beta * I / (1 - D)
    J22 = -sigma
    J23 = beta * S / (1 - D)
    J24 = 0
    J25 = beta * S * I / (1 - D)^2
    J31 = 0
    J32 = sigma
    J33 = -(gamma + mu)
    J34 = 0
    J35 = 0
    J41 = 0
    J42 = 0
    J43 = gamma
    J44 = 0
    J45 = 0
    J51 = 0
    J52 = 0
    J53 = mu
    J54 = 0
    J55 = 0

    J = [J11 J12 J13 J14 J15
         J21 J22 J23 J24 J25
         J31 J32 J33 J34 J35
         J41 J42 J43 J44 J45
         J51 J52 J53 J54 J55]

    return J

end

## Solution
t, Yapp = FDEsolver(F, tSpan, y0, alpha, par, JF = JacobF, h = h);

# Plot
plot(t, Yapp, linewidth = 5, title = "Numerical solution of SEIRD model",
     xaxis = "Time (t)", yaxis = "SEIRD populations", label = ["Susceptible" "Exposed" "Infectious" "Recovered" "Dead"]);
savefig("example3.png"); nothing # hide