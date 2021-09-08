# python_intro.py
"""Python Essentials: Introduction to Python.
<Name>
<Class>
<Date>
"""

#Problem 1
def isolate(a, b, c, d, e):
    """Prints the first three arguments separated by 5 spaces, then prints the rest with a single space between each output"""

    print(a, "    ", b, "    ", c, d, e)
    
    return

print(isolate(1,2,3,4,5))


#Problem 2
def first_half(string):
    """returns the first half of the string excluding the middle character"""
    
    n = len(string)
    
    newstring = string[:n//2]
    
    return newstring

print(first_half("hello"))


def backward(first_string):
    """returns the string backward"""
    
    newstring = first_string[::-1]
    
    return newstring

print(backward("hello!"))


#Problem 3
def list_ops():
    """Performs a lot of random operations on a list"""
    
    mylist = ["bear", "ant", "cat", "dog"]
    
    mylist.append("eagle")
    print(mylist)
    mylist.remove("cat")
    mylist.insert(2,"fox")
    mylist.pop(1)
    print(mylist)
    mylist.sort(reverse=True)
    print(mylist)
    x = mylist.index("eagle")
    mylist.remove("eagle")
    mylist.insert(x,"hawk")
    print(mylist)
    y = len(mylist)
    mylist[y-1] = mylist[y-1] + "hunter"
    
    return mylist

print(list_ops())


#Problem 4
def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """
    
    
    nlist = [((-1)**(i+1))/i for i in range (1,n+1)]
    
    return sum(nlist)
print(alt_harmonic(500000))


import numpy as np
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

print(prob5(np.array([-3,-1,3])))

def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    
    A = np.arange(6).reshape(3,2).transpose()
    B = np.full((3,3),3)
    B = np.tril(B)
    C = np.diag([-2,-2,-2])
    print(A)
    print(B)
    print(C)
    print(np.shape(A.T))
    
    D = np.hstack((np.zeros((3,4)), A.T, np.eye(3)))
    print(np.shape(D))
    E = np.hstack((A,np.zeros((2,3)),np.zeros((2,3))))
    print(np.shape(E))
    F = np.hstack((B,np.zeros((3,3)),C))
    print(np.shape(F))
    G = np.vstack((D,E,F))
    print(G)
    
    return G

print(prob6())


def prob7(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob7(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    
    sums = np.sum(A, axis=1)
    
    A = A.T/sums
         
    return A.T

print(prob7(np.array([[1,1,0],[0,1,0],[1,1,1]])))

def prob8():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")
    
    print(grid)
    
    
    return

print(prob8())































