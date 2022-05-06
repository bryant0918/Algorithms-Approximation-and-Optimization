"""Volume 2: Simplex

Bryant McArthur
Feb 24, 2022
Sec 002
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        #Check if feasible then construct
        if np.all(b>0):
            self.c = c
            self.A = A
            self.b = b
            (self.m,self.n) = A.shape
        else:
            raise ValueError("b must be element-wise greater than 0")
            
        #Generate the dictioary
        self._generatedictionary(c,A,b)

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        #I thought this was more natural for c to be ((n,1) ndarray)
        if np.ndim(c) == 2:
            self.D = np.block([[0,self.c.T,np.zeros(self.m)],[self.b,-self.A,-np.eye(self.m)]])
        
        #The test case will run this one:
        if np.ndim(c) == 1:
            self.D = np.hstack((np.array([[d] for d in np.concatenate((np.array([0]),b))]),
                                np.vstack((np.concatenate((c, np.zeros(self.m))).T,-np.hstack((A,np.eye(self.m)))))))


    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        #Find the first negative entry of the top row besides the zeroth entry
        for j in range(1,self.m+self.n+1):
            if self.D[0,j] < 0:
                return j

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        #Check boundedness
        if np.all(self.D[:,index] >= 0):
            raise ValueError("The feasible set is unbounded and there is no solution since All entries of pivot column are non-negative.")
        
        #Set r to be infinity so that its easy to find an r less than the current r
        r = np.inf
        for i in range(1,self.m+1):
            #If negative
            if self.D[i,index] < 0:
                #Find the ratio
                if -self.D[i,0]/self.D[i,index] < r:
                    r = -self.D[i,0]/self.D[i,index]
                    I = i  #Save the Index
        
        return I
        
        

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        #Find pivot column and row
        j = self._pivot_col()
        i = self._pivot_row(j)
        
        #Find the lengths of columns and rows
        lr = len(self.D[:,0])
        lc = len(self.D[0,:])
        
        #Create empty matrix
        t = np.zeros((lr,lc))
        
        #Define the pivot row and element then scale
        pivotrow = self.D[i,:]
        element = 1/self.D[i,j]
        row = pivotrow*element
        
        #Iterate through
        for k in range(len(self.D[:,j])):
            #Set c
            c = self.D[k,j]
            #Skip the pivot row
            if list(self.D[k,:]) == list(pivotrow):
                continue
            #Do what is necessary
            else:
                t[k,:] = list(self.D[k,:]-row*c)
                
        #Somehow our row got negated
        t[i,:] = list(-row)
        #We are done
        self.D = t
        
        
    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        #Pivot while all coefficients are nonpositive
        while np.mean(self.D[0,1:] < 0) > 0:
            self.pivot()
        
        #Initialize empty dictionaries
        dependents = {}
        independents = {}
        
        f = self.D[0,1:]
        
        #Iterate through columns
        for j in range(len(f)+1):
            if self.D[0,j] > 0:
                independents[j-1] = 0  #We always set independent variables to zero
            else:
                #Iterate through rows
                for i in range(len(self.D[:,j])):
                    if self.D[i,j] == -1:
                        dependents[j-1] = self.D[i,0]
            
        return self.D[0,0], dependents, independents
        

# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    #Load data and save accordingly
    data = np.load(filename)
    a = data['A']
    p = data['p']
    resources = data['m']
    d = data['d']
    
    #Turn into a minimization problem
    c = -p
    n = len(a[0])
    I = np.eye(n)
    
    #Define b and A
    b = np.concatenate((resources, d))
    A = np.vstack((a,I))
    
    #Run our simplex solver class
    ss = SimplexSolver(c, A, b)
    minimum, dependents, independents = ss.solve()
    
    quantity = []
    #Iterate through keys to find quantity for each product.
    for i in range(len(c)):
        if i in independents.keys():
            quantity.append(independents[i])
        elif i in dependents.keys():
            quantity.append(dependents[i])
            
    return np.array(quantity)
    
if __name__ == "__main__":
    """
    c = np.array([[-3],[-2]])
    A = np.array([[1,-1],[3,1],[4,3]])
    b = np.array([[2],[5],[7]])
    
    ss = SimplexSolver(c,A,b)
    
    print(ss.solve())
    """
    
    c = np.array([-3,-2])
    A = np.array([[1,-1],[3,1],[4,3]])
    b = np.array([2,5,7])
    
    ss = SimplexSolver(c,A,b)
    
    print(ss.solve())
    
    print(prob6())
    pass
    
