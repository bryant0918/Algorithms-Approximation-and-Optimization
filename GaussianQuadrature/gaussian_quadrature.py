# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
Bryant McArthur
Sec 002
January 20
"""

import numpy as np
from scipy import linalg as la
from scipy.integrate import quad,nquad
from scipy.stats import norm
from matplotlib import pyplot as plt

class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        #Raise error if its not Legendre or chebyshev
        if polytype != "legendre" and polytype != "chebyshev":
            raise ValueError("Polytype must be legendre or chebyshev")
            
        #Store the attributes
        self.polytype = polytype
        self.points, self.weights = self.points_weights(n)
        
        if polytype == "legendre":
            self.winv = lambda x: 1
        else:
            self.winv = lambda x: np.sqrt(1-x**2)
        

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        #Initialize everything
        alpha,beta,weights = [],[],[]
        
        #Find alpha and beta for legendre
        if self.polytype == "legendre":
            for k in range(1,n+1):
                alpha.append(0)
                beta.append(k**2/(4*k**2-1))
                
        #Find alpha and beta for chebyshev  
        else:
            for k in range(n):
                alpha.append(0)
                if k ==0:
                    beta.append(.5)
                else:
                    beta.append(.25)
                    
        #Create the diagonal matrices and the jacobi
        diag = np.diag(alpha)
        diagtop = np.diag(np.sqrt(beta[:n-1]),1)
        diagbot = np.diag(np.sqrt(beta[:n-1]),-1)
        jacobi = diag + diagtop + diagbot
        
        #Find the eigenvalues and eigenvectors of the jacobi
        eigvals,eigvecs = la.eig(jacobi)
        
        #Find the weights from the eigenvectors
        if self.polytype == "legendre":
            weights = 2*eigvecs[0,:]**2
            
        else:
            weights = np.pi*eigvecs[0,:]**2
        
        return eigvals,weights
            

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        return np.dot(f(self.points)*self.winv(self.points),self.weights)

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        #Initialize h and appproximate
        h = lambda x: f((b-a)/2*x+(a+b)/2)
        approx = self.basic(h)
        
        return (b-a)/2*approx

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        #Initialize functions and values
        h = lambda x,y: f((b1-a1)/2*x+(a1+b1)/2, (b2-a2)/2*y+(a2+b2)/2)
        n = len(self.points)
        g = lambda x,y: h(x,y)* self.winv(x) * self.winv(y)
        approx = 0
        
        #Iterate through double sum according to equation
        for i in range(n):
            for j in range(n):
                approx += self.weights[i] * self.weights[j] * g(self.points[i], self.points[j])
        
        return np.real((b1-a1)*(b2-a2)/4*approx)


# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    f = lambda x: (1/np.sqrt(2*np.pi))*np.exp((-x**2)/2)
    F = norm.cdf(2)-norm.cdf(-3)
    lerror,cerror,serror = [],[],[]
    domain = [i*5 for i in range(1,11)]
    
    for n in domain:
        #Find the Legendre error
        GQ = GaussianQuadrature(n)
        approx = GQ.integrate(f,-3,2)
        lerror.append(np.abs(F-approx))
        
        #Find the Chebyshev error
        GQ = GaussianQuadrature(n,'chebyshev')
        approx = GQ.integrate(f,-3,2)
        cerror.append(np.abs(F-approx))
        
        #Append the scipy error
        approx = quad(f,-3,2)[0]
        serror.append(np.abs(F-approx))
          
    #Plot it
    plt.plot(domain,lerror,label = "Legendre Error")
    plt.plot(domain,cerror,label = "Chebyshev Error")
    plt.plot(domain,serror,label = "Scipy Error")
    plt.legend(loc=0)
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Error")
    plt.show()
    
    return

#prob5()


if __name__ == "__main__":
    GQ = GaussianQuadrature(100)
    #print(GQ.points_weights(5))
    
    f = lambda x: 1/ np.sqrt(1-x**2)
    #print(quad(f,-1,1)[0])
    
    #print(GQ.basic(f))
    
    f = lambda x: (1/np.sqrt(2*np.pi))*np.exp((-x**2)/2)
    #print(norm.cdf(2)-norm.cdf(-3))
    #print(GQ.integrate(f,-3,2))
    
    f = lambda x,y: np.sin(x) + np.cos(y)
    print(nquad(f,[[-10,10],[-1,1]])[0])
    print(GQ.integrate2d(f,-10,10,-1,1))
    GQ = GaussianQuadrature(100,'chebyshev')
    print(GQ.integrate2d(f,-10,10,-1,1))
    
    f = lambda x,y: x**2*y**2
    print(nquad(f,[[-1,2],[-3,4]])[0])
    print(GQ.integrate2d(f,-1,2,-3,4))
    GQ = GaussianQuadrature(100,'chebyshev')
    print(GQ.integrate2d(f,-1,2,-3,4))
    
    
    pass