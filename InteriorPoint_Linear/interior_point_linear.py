# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
Bryant McArthur
Sec 002
March 24
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt
from autograd import grad, jacobian, elementwise_grad, holomorphic_grad


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m,n = len(A),len(A.T)
    x, lam, mu = starting_point(A, b, c)
    
    for i in range(niter):
    
        def F(x, lam, mu):
            """Return 1D Array with 2n+m entries"""
            row1 = A.T@lam + mu - c
            row2 = A@x - b
            row3 = np.diag(mu)@x
            return np.hstack([row1, row2, row3]) #Concatenate everything
        
        
        Farray = F(x,lam,mu)
        
        def DF():
            """compute search direction"""
            sigma = 1/10
            v = (x.T@mu)/n
            
            #Find Df block matrix
            DF = np.block([[np.zeros((n,n)), A.T, np.eye(len(x))],
                           [A, np.zeros((m,m)), np.zeros((m,n))],
                           [np.diag(mu), np.zeros((n,m)), np.diag(x)]])
            
            sig_v_e = sigma*v*np.ones(n)
            
            newton = np.hstack((np.zeros(n), np.zeros(m), sig_v_e)) #Concatenate for my delta vector
            
            lu, piv = la.lu_factor(DF)
            
            delta = la.lu_solve((lu,piv), -Farray + newton)
        
            return delta, v
        
        #Get v for checking convergence
        delta, v = DF()
        
        #Extract delta variables from vector
        mus = delta[n+m:]
        lams = delta[n:n+m]
        xs = delta[:n]
        
        
        def step_size():
            """compute step size after search direction"""
            #Mask it up
            mu_mask = mus < 0
            x_mask = xs < 0
            
            if np.mean(mus>0) == 1:
                alpha_max = 1
            else:
                alpha_max = np.min((-mu/mus)[mu_mask])
                
            if np.mean(xs > 0) == 1:
                delta_max = 1
            else:
                delta_max = np.min((-x/xs)[x_mask])
            
            a = min(1, .95*alpha_max)
            d = min(1, .95*delta_max)
            
            return a,d
            
        a, d = step_size()
        
        #Calculate everything for next iteration
        xk = x + d*xs
        lamk = lam + a*lams
        muk = mu + a*mus
        
        x = xk
        lam = lamk
        mu = muk
        
        if v < tol:
            break
        
    return x, c.T@x


np.random.seed(2)
A, b, c, x = randomLP(7,5)
point, value = interiorPoint(A,b,c)
#print(np.allclose(x, point[:5]))
        
def test():
    for _ in range(200):
        j = np.random.randint(3,10)
        A,b,c,x = randomLP(j,j)
        point,value = interiorPoint(A,b,c)
        if np.allclose(x, point[:j]) == False:
            print("fail")
            print("square j", j)
    for _ in range(150):
        n = np.random.randint(3,10)
        m = np.random.randint(n,20)
        A,b,c,x = randomLP(m,n)
        point, value = interiorPoint(A,b,c)
        if np.allclose(x, point[:n]) == False:
            print("fail")
            print(n,m)
            
#test()

def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    
    with open(filename, 'r') as myfile:
        data1 = myfile.read().split()
    
    data = np.loadtxt(filename)
    
    #LAD
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]
    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)
    sol = interiorPoint(A, y, c, niter=10)[0]
    beta = sol[m:m+n] - sol[m+n:m+2*n]
    b = sol[m+2*n] - sol[m+2*n+1]
    
    
    y = []
    x = []
    for i in range(len(data1)):
        if i % 2 == 0:
            y.append(float(data1[i]))
        else:
            x.append(float(data1[i]))
    
    #Plot it
    slope, intercept = linregress(x,y)[:2]
    domain = np.linspace(0,10,200)
    plt.plot(domain, domain*beta + b, label = "Interior Point Method")
    plt.plot(domain, domain*slope + intercept, label = "Least Squares Line")
    plt.scatter(x,y, marker = 'o')
    plt.legend(loc = 0)
    
    return

#leastAbsoluteDeviations()


