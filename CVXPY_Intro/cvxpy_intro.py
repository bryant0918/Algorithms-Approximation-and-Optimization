# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Bryant McArthur
Sec 002
March 10
"""

import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Set up objective function
    x = cp.Variable(3, nonneg = True)
    c = np.array([2,1,3])
    objective = cp.Minimize(c.T@x)
    
    #Set up constraints
    G = np.array([[1,2,0],[0,1,-4],[-2,-10,-3]])
    P = np.eye(3)
    b = np.array([3,1,-12])
    constraints = [G@x <= b.T, P@x >= 0]
    
    #Define the problem and solve
    problem = cp.Problem(objective, constraints)
    optimalval = problem.solve()
    xval = x.value
    
    return xval, optimalval
    
#print(prob1())


# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Get the number of variables
    n = len(A.T)
    
    #Set up the objective function
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x,1))
    
    #Set up the constraint
    constraint = [A@x == b]
    
    #Define the problem and solve
    problem = cp.Problem(objective, constraint)
    optimalval = problem.solve()
    xval = x.value
    
    return xval, optimalval

A = np.array([[1,2,1,1],[0,3,-2,-1]])
b = np.array([7,4])

#print(l1Min(A,b))
    
    

# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Set up the objecctive function
    x = cp.Variable(6, nonneg=True)
    c = np.array([4,7,6,8,8,9])
    objective = cp.Minimize(c.T@x)
    
    #Set up G from supply
    G = np.array([[1,1,0,0,0,0],[0,0,1,1,0,0],[0,0,0,0,1,1]])
    g = np.array([7,2,4])
    
    #Set up A from demand
    A = np.array([[1,0,1,0,1,0],[0,1,0,1,0,1]])
    b = np.array([5,8])
    
    constraints = [G@x <= g.T, A@x == b.T]
    
    #Define problem and solve
    problem = cp.Problem(objective, constraints)
    optimalval = problem.solve()
    xval = x.value
    
    return xval, optimalval

#print(prob3())

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Set up quadratic optimization problem
    Q = np.array([[3,2,1],[2,4,2],[1,2,3]])
    r = np.array([3,0,1])
    x = cp.Variable(3)
    
    #Our problem is unconstrained
    prob = cp.Problem(cp.Minimize(.5*cp.quad_form(x,Q)+r.T@x))
    
    #Solve
    optimalval = prob.solve()
    xval = x.value
    
    return xval, optimalval


#print(prob4())

# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #Set up objective function
    n = len(A.T)
    x = cp.Variable(n,nonneg=True)
    objective = cp.Minimize(cp.norm(A@x-b,2))
    
    #set up constraint, same as one-norm of x
    constraint = [np.ones(n)@x.T==np.ones(n)]
    
    #Define problem and solve
    problem = cp.Problem(objective, constraint)
    optimalval = problem.solve()
    xval = x.value
    
    return xval, optimalval
    
#print(prob5(A,b))


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    #Load in the data
    data = np.load("food.npy", allow_pickle = True)
    
    #Find the number of variables
    n = len(data)
    
    #Set up objective function from the price (first column)
    x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(data[:,0]@x)
    
    #Find the servings from first column to multiply everything by
    servings = data[:,1]
    
    #Set up the constraints
    G = np.array([servings*data[:,2],
                  servings*data[:,3],
                  servings*data[:,4],
                  -servings*data[:,5],
                  -servings*data[:,6],
                  -servings*data[:,7]])
    g = np.array([2000,65,50,-1000,-25,-46])
    constraint = [G@x <= g.T]
    
    #Define the problem and solve
    problem = cp.Problem(objective, constraint)
    optimalval = problem.solve()
    xval = x.value
    
    return xval, optimalval

#print(prob6())

"""
xval, optimalval = prob6()
print(xval)

print(np.argmax(xval))
"""


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    