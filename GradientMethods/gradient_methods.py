# gradient_methods.py
"""Volume 2: Gradient Descent Methods.
Bryant McArthur
2/17/22
Sec 002
"""

from scipy import optimize as opt
import numpy as np
from matplotlib import pyplot as plt



# Problem 1
def steepest_descent(f, Df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the exact method of steepest descent.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    b = False
    phi = lambda alpha: f(x0-alpha*Df(x0))
    #Dphi = grad(phi)
    alpha = 1
    
    for k in range(maxiter):
        #Find optimal alpha using Newton's One-Dimensional Method
        #alpha = newton(phi,alpha,Dphi)[0]
        alpha = opt.minimize_scalar(phi, method='brent').x
        
        #Find x1 from (12.13)
        x1 = x0 - alpha*Df(x0)
        
        if np.linalg.norm(Df(x0), ord=np.inf) < tol:
            b = True
            break
        x0 = x1
        
    return x1,b,k+1


# Problem 2
def conjugate_gradient(Q, b, x0, tol=1e-4):
    """Solve the linear system Qx = b with the conjugate gradient algorithm.

    Parameters:
        Q ((n,n) ndarray): A positive-definite square matrix.
        b ((n, ) ndarray): The right-hand side of the linear system.
        x0 ((n,) ndarray): An initial guess for the solution to Qx = b.
        tol (float): The convergence tolerance.

    Returns:
        ((n,) ndarray): The solution to the linear system Qx = b.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize the variables
    converged = False
    maxiter = 100
    r0 = Q@x0-b
    
    d0 = -r0
    k = 0
    
    #If we havne't reached the max iters and if r0 is less than our tolerance iterate
    while (np.linalg.norm(r0) >= tol) and k < maxiter:
        
        #Find alpha
        alpha = (r0.T@r0)/(np.dot(d0.T@Q,d0))
        
        #Follow the algorithm
        x1 = x0 + alpha*d0
        r1 = r0 + alpha*Q@d0
        beta = (r1.T@r1)/(r0.T@r0)
        d1 = -r1 + beta*d0
        k += 1
        
        #Reassign variables for next iteration
        x0 = x1
        d0 = d1
        r0 = r1
        
    #Determine whether or not you converged.
    if k < maxiter:
        converged = True
    
    return x1, converged, k

Q = np.array([[2,0],[0,4]])
b = np.array([[1],[8]])

#print(conjugate_gradient(Q, b, np.array([[0],[1]])))



# Problem 3
def nonlinear_conjugate_gradient(f, df, x0, tol=1e-5, maxiter=100):
    """Compute the minimizer of f using the nonlinear conjugate gradient
    algorithm.

    Parameters:
        f (function): The objective function. Accepts a NumPy array of shape
            (n,) and returns a float.
        Df (function): The first derivative of f. Accepts and returns a NumPy
            array of shape (n,).
        x0 ((n,) ndarray): The initial guess.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        ((n,) ndarray): The approximate minimum of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #Initialize variables
    converged = False
    r0 = -df(x0).T
    d0 = r0.copy()
    
    #Set phi function to optimize in order to find alpha
    phi = lambda alpha: f(x0+alpha*d0)
    alpha = opt.minimize_scalar(phi).x
    
    x0 = x0 + alpha*d0
    k = 1
    
    #Iterate while not yet converged
    while (np.linalg.norm(r0) >= tol) and (k < maxiter):
        
        #Follow the algorithm
        r1 = -df(x0).T
        beta = (r1.T@r1)/(r0.T@r0)
        
        d0 = r1 + beta*d0
        
        phi = lambda alpha: f(x0+alpha*d0)
        alpha = opt.minimize_scalar(phi).x
        
        x0 = x0 + alpha*d0
        k += 1
        
        r0 = r1.copy()
        
    #Determine whether it converged or not.
    if k < maxiter:
        converged = True

    return x0, converged, k

#print(opt.fmin_cg(opt.rosen, np.array([10,10]), fprime=opt.rosen_der))
#print(nonlinear_conjugate_gradient(opt.rosen, opt.rosen_der, np.array([10,10]),maxiter=1000))
    
#print(nonlinear_conjugate_gradient(f, df, x0))


# Problem 4
def prob4(filename="linregression.txt",
          x0=np.array([[-3482258], [15], [0], [-2], [-1], [0], [1829]])):
    """Use conjugate_gradient() to solve the linear regression problem with
    the data from the given file, the given initial guess, and the default
    tolerance. Return the solution to the corresponding Normal Equations.
    """
    #Open the data
    data = np.loadtxt(filename)
    
    #Create b and A from the data
    b = data[:,0]
    A = data[:,1:]
    
    #Set a column of 1s that you'll column stack
    ones = np.ones(len(A))
    Q = np.column_stack([ones,A])
    
    #Find b and Q
    b = Q.T@b
    Q = Q.T@Q
    
    #Call conjugaet
    soln, b, k = conjugate_gradient(Q, b, x0)
    
    return soln
        
    
#print(prob4())


# Problem 5
class LogisticRegression1D:
    """Binary logistic regression classifier for one-dimensional data."""

    def fit(self, x, y, guess):
        """Choose the optimal beta values by minimizing the negative log
        likelihood function, given data and outcome labels.

        Parameters:
            x ((n,) ndarray): An array of n predictor variables.
            y ((n,) ndarray): An array of n outcome variables.
            guess (array): Initial guess for beta.
        """
        
        n = len(x)
        
        #Negative logliklihood
        def f(beta):
            b0 = beta[0]
            b1 = beta[1]
            s = 0
            
            #Iterate through adding up s
            for i in range(n):
                s += np.log(1 + np.exp(-(b0 + b1 * x[i])))
                s += (1 - y[i]) * (b0 + b1 * x[i])
            
            return s
        
        #Find the minimizer
        minim = opt.fmin_cg(f, guess)
        
        self.b0 = minim[0]
        self.b1 = minim[1]

    def predict(self, x):
        """Calculate the probability of an unlabeled predictor variable
        having an outcome of 1.

        Parameters:
            x (float): a predictor variable with an unknown label.
        """
        
        return 1/(1+np.exp(-(self.b0+self.b1*x)))


# Problem 6
def prob6(filename="challenger.npy", guess=np.array([20., -1.])):
    """Return the probability of O-ring damage at 31 degrees Farenheit.
    Additionally, plot the logistic curve through the challenger data
    on the interval [30, 100].

    Parameters:
        filename (str): The file to perform logistic regression on.
                        Defaults to "challenger.npy"
        guess (array): The initial guess for beta.
                        Defaults to [20., -1.]
    """
    #Open the data
    data = np.load(filename)
    x = data[:,0]
    y = data[:,1]
    
    #Call the class we created and fit our guess to x and y.
    model = LogisticRegression1D()
    model.fit(x,y,guess)
    
    #Create a linspace of the temperature
    temp = np.linspace(30,100,1000)
    #Find the damage from the prediction
    damage = [model.predict(t) for t in temp]
    
    #Plot it
    plt.title("Probability of Damage")
    plt.scatter(x,y,color='blue', label='Previous Damage')
    plt.plot(temp,damage)
    plt.scatter(31,model.predict(31),label = 'Damage at launch')
    plt.xlabel("Temperature")
    plt.ylabel("Damage")
    plt.legend(loc=0)
    plt.show()
    
    
        
