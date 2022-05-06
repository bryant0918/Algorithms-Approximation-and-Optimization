# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
Bryant McArthur
Sec 002
January 13
"""
import numpy as np
from numpy.fft import fft
from matplotlib import pyplot as plt
from scipy import interpolate as ty
from scipy import linalg as la
# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    
    #Initialize everything
    n = len(xint)
    m = len(points)
    L = np.zeros(m)
    
    #Calculate Denominator
    def denom(xj,xint,n):
        denominator = np.product(xj-np.delete(xint,j))
        return denominator
    
    #Iterate through j
    for j in range(n):
        denominator = denom(xint[j],xint,j)
        numerator = np.product(points-np.delete(xint,j).reshape(n-1,1),axis=0)
        #Calculate each Lj and stack it to L
        Lj = numerator/denominator
        L = np.vstack((L,Lj))
    
    #Delete the top row of zeros and calculate p
    L = np.delete(L,0,axis=0)
    p = yint @ L
    
    
    return p
        

# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        #Initialize Constructors
        self.x = xint
        self.y = yint
        self.n = len(xint)
        
        w = np.ones(self.n) # Array for storing barycentric weights.
        # Calculate the capacity of the interval.
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(self.n-1)
        for j in range(self.n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle] # Randomize order of product.
            w[j] /= np.product(temp)
        
        self.w = w
        

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        x = self.x
        
        #Calculate the numerator and denominator
        top = np.array([np.sum([((self.w[j]*self.y[j]/(zed-self.x[j]))) for j in range(self.n)]) for zed in points])
        bottom = np.array([np.sum([((self.w[j])/(zed-self.x[j])) for j in range(self.n)]) for zed in points])
        
        #Create a mask to keep what we want
        mask = [True if zed in x else False for zed in points]
        masky = [True if x in points else False for x in x]
        
        
        
        replace = self.y[masky]
        
        baryint = top/bottom
        baryint[mask] = replace
        
        return baryint

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        #Add new values to existing values and update n
        self.x = np.hstack((self.x,xint))
        self.y = np.hstack((self.y,yint))
        self.n = len(self.x)
        
        #Recalculate Weights
        w = np.ones(self.n) # Array for storing barycentric weights.
        # Calculate the capacity of the interval.
        C = (np.max(self.x) - np.min(self.x)) / 4
        shuffle = np.random.permutation(self.n-1)
        for j in range(self.n):
            temp = (self.x[j] - np.delete(self.x, j)) / C
            temp = temp[shuffle] # Randomize order of product.
            w[j] /= np.product(temp)
        
        self.w = w


# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    #Initialize Everything
    domain = np.linspace(-1,1,400)
    p = [2**p for p in range(2,9)]
    f = lambda x: 1/(1+25*x**2)
    normal = f(domain)
    unierror, cheberror = [], []
    
    #Iterate through powers of 2
    for n in p:
        x = np.linspace(-1,1,n)
        y = np.array(f(x))
        tilda = ty.BarycentricInterpolator(x,y)
        
        #Find the error for uniform points
        unierror.append(la.norm(normal-tilda(domain),ord=np.inf))
        
        #Find the error for chebyshev points
        chebyextrema = [np.cos(j*np.pi/n) for j in range(n)]
        
        #Find the y values of chebyshev extremizers
        chebyy = []
        for i in range(n):
            chebyy.append(f(chebyextrema[i]))
        
        cheby = ty.BarycentricInterpolator(chebyextrema,chebyy)
        cheberror.append(la.norm(normal-cheby(domain),ord=np.inf))
        
    #Plot it
    plt.loglog(p,unierror,label="Uniform points")
    plt.loglog(p,cheberror,label="Chebyshev points")
    plt.xlabel("n-value")
    plt.ylabel("Error")
    plt.legend(loc="upper left")
    plt.title("Error of Barycentric Interpolation")
    plt.show()
        
        
    
#prob5()

# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    #Use the code in the book
    y = np.cos((np.pi*np.arange(2*n))/n)
    samples = f(y)
    
    coeffs = np.real(fft(samples))[:n+1]/n
    coeffs[0] = coeffs[0]/2
    coeffs[n] = coeffs[n]/2
    
    return coeffs


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    #Load data
    data = np.load("airdata.npy")
    
    #Initialize everything
    f = lambda a,b,n: .5*(a+b + (b-a) * np.cos(np.arange(n+1)*np.pi/n))
    a,b = 0,366-1/24
    domain = np.linspace(0,b,8784)
    points = f(a,b,n)
    temp = np.abs(points - domain.reshape(8784,1))
    temp2 = np.argmin(temp,axis=0)
    
    #Calculate the interpolation
    poly = ty.BarycentricInterpolator(domain[temp2], data[temp2])
    
    #Plot it
    plt.subplot(121).plot(domain,poly(domain),label="Interpolation")
    plt.xlabel("Day of the year")
    plt.title("Interpolation")
    plt.subplot(122).plot(domain,data,label="Airquality",alpha = .75)
    plt.xlabel("Day of the year")
    plt.title("Actual Airquality")
    plt.ylabel("PM2.5 Concentration")
    plt.suptitle("PM2.5 Concentration throughout the year")
    plt.tight_layout()
    plt.show()
    
#prob7(46)



if __name__ == "__main__":
    f = lambda x: 1/(1+25*x**2)
    xint = np.linspace(-1,1,5)
    xint2 = np.linspace(-.99,.99,20)
    yint = f(xint)
    yint2 = f(xint2)
    points = np.linspace(-1,1,100)
    
    p = lagrange(xint,yint,points)
    bary = Barycentric(xint,yint)
    bary.add_weights(xint2, yint2)
    b = bary(points)
    """
    plt.plot(points,f(points),label="Runges")
    #plt.plot(points,p,label="Lagrange")
    plt.plot(points,b,label="Barycentric")
    plt.legend(loc='upper left')
    #plt.ylim(0,1)
    plt.show()
    
    """
    
    pass