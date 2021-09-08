# python_intro.py
"""Python Essentials: Introduction to Python.
Bryant McArthur
Math 321 Sec 002
September 8
"""

#Problem 1
def isolate(a, b, c, d, e):
    """Prints the first three arguments separated by 5 spaces, then prints the rest with a single space between each output"""

    print(a, "    ", b, "    ", c, d, e)
    
    return


#Problem 2
def first_half(string):
    """returns the first half of the string excluding the middle character"""
    
    n = len(string)
    
    newstring = string[:n//2] 
    
    return newstring


def backward(first_string):
    """returns the string backward"""
    
    newstring = first_string[::-1]
    
    return newstring



#Problem 3
def list_ops():
    """Performs a lot of random operations on a list"""
    
    mylist = ["bear", "ant", "cat", "dog"]
    
    mylist.append("eagle")
    
    mylist.remove("cat")
    mylist.insert(2,"fox")
    mylist.pop(1)
    
    mylist.sort(reverse=True)
    
    x = mylist.index("eagle")
    mylist.remove("eagle")
    mylist.insert(x,"hawk")
    
    y = len(mylist)
    mylist[y-1] = mylist[y-1] + "hunter"
    
    return mylist


#Problem 4
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """
    
    nlist = [((-1)**(i+1))/i for i in range (1,n+1)]
    
    return sum(nlist)


import numpy as np
#import numpy module

def prob5(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    
    B = A
    mask = B < 0 #Same as np.array([i < 0 for i in B])
    B[mask] = 0
    
    return B


def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    # Create Matrices A, B, and C
    A = np.arange(6).reshape(3,2).transpose()
    B = np.full((3,3),3)
    B = np.tril(B)
    C = np.diag([-2,-2,-2])
    # Create new matrices stacking each column
    D = np.hstack((np.zeros((3,3)), A.T, np.eye(3)))
    E = np.hstack((A,np.zeros((2,2)),np.zeros((2,3))))
    F = np.hstack((B,np.zeros((3,2)),C))
    # Create final matrix by stacking the rows D, E and F
    G = np.vstack((D,E,F))
    
    return G


def prob7(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob7(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    
    sums = np.sum(A, axis=1) #A matrix of sums along each row
    
    A = A.T/sums
         
    return A.T

def prob8():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy") # Upload grid
    
    # Find max along horizontal
    rowwinner = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])
    # Find max along verticle
    columnwinner = np.max(grid.T[:,:-3] * grid.T[:,1:-2] * grid.T[:,2:-1] * grid.T[:,3:])
    # Find max along right diagonal
    rdiagonal = np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:,3:])
    # Find max along left diagonal
    ldiagonal = np.max(grid[:-3,3:] * grid[1:-2,2:-1] * grid[2:-1,1:-2] * grid[3:,:-3])
    
    
    return max(rowwinner, columnwinner, rdiagonal, ldiagonal)































