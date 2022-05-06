# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
Bryant McArthur
Sec 002
Jan 27 2022
"""
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
from autograd import numpy as anp
from autograd import grad


# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize variables
    converged = False
    x0 = (a+b)/2
    gold = (1 + np.sqrt(5))/2
    
    #Iterate through maxiters
    for i in range(maxiter):
        c = (b-a)/gold
        atilda = b-c
        btilda = a+c
        if f(atilda) <= f(btilda):
            b = btilda
        else:
            a = atilda
        x1 = (a+b)/2
        
        #If converges break
        if np.abs(x0-x1) < tol:
            converged = True
            break
        
        x0 = x1
        
    return x1,converged,i+1



# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False
    
    #Iterate through
    for i in range(maxiter):
        x1 = x0 - df(x0)/d2f(x0)
        
        #Break if converges
        if np.abs(x0-x1) < tol:
            converged = True
            break
        
        x0 = x1
        
    return x1,converged,i+1




# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    converged = False
    dfx0 = df(x0)
    
    #Iterate
    for i in range(maxiter):
        dfx1 = df(x1)
        x2 = (x0*dfx1 - x1*dfx0)/(dfx1-dfx0)
        
        #Break if converges
        if np.abs(x2-x1) < tol:
            converged = True
            break
        
        #Reset variables for next iteration
        x0 = x1
        x1 = x2
        dfx0 = dfx1
        
    return x2, converged, i+1



# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    
    #Follow the pseudo code
    Dfp = Df(x).T @ p
    fx = f(x)
    while (f(x+alpha*p) > (fx + c*alpha*Dfp)):
        alpha = rho*alpha
    
    return alpha




if __name__ == "__main__":
    f = lambda x: np.exp(x) - 4*x
    print(opt.golden(f, brack=(0,3), tol = .001))
    print(golden_section(f, 0,3, tol=.001))
    df = lambda x: 2*x + 5*np.cos(5*x)
    d2f = lambda x: 2 - 25*np.sin(5*x)
    print(opt.newton(df, x0=0, fprime=d2f, tol=1e-10, maxiter=500))
    print(newton1d(df,d2f,0,1e-10,500))
    f = lambda x: x**2 + np.sin(x) + np.sin(10*x)
    df = lambda x: 2*x + np.cos(x) + 10*np.cos(10*x)
    print(opt.newton(df,0,tol=1e-10,maxiter=500))
    print("Secant", secant1d(df,0,-1,1e-10,maxiter=500))
    x2,c,i = secant1d(df,0,-1,1e-10,maxiter=500)
    domain = np.linspace(-1,1,100)
    plt.plot(domain,f(domain))
    plt.scatter(x2,f(x2))
    plt.show()
    f = lambda x: x[0]**2 + x[1]**2 + x[2]**2
    Df = lambda x: np.array([2*x[0], 2*x[1], 2*x[2]])
    x = anp.array([150.,.03,40.])
    p = anp.array([-.5,-100.,-4.5])
    phi = lambda alpha: f(x+alpha*p)
    dphi = grad(phi)
    alpha, _ = opt.linesearch.scalar_search_armijo(phi, phi(0.), dphi(0.))
    print(alpha)
    print(backtracking(f,Df,x,p))
    pass