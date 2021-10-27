# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
Bryant McArthur
Math 321
10/21/21
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
from scipy import stats
#from matplotlib import pyplot as plt


# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    
    d = min(la.norm(X - z, axis = 1))
    x = np.argmin(la.norm(X - z, axis = 1))
    
    return X[x], d




# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if type(x) != np.ndarray:
            raise TypeError("x is not a NumPy array")
        self.value = x
        self.left, self.right, self.pivot = None, None, None
        
        

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        
        newnode = KDTNode(data)
        
        #If the data isn't the right dimension
        if len(data) != self.k and self.k is not None:
            raise ValueError("Data does not have the right dimension")
        
        #If you are inserting the first node
        if self.root is None:
            self.root = newnode
            self.root.pivot = 0
            self.k = len(data)
        
        
        else:
            def step(current, data):
                """Recursively step through the tree to find where to put the new node"""
                
                #If data is equal to the current value
                if np.allclose(data, current.value):
                    raise ValueError("Data is already in the tree")
                
                #If the data is less than the current value
                elif data[current.pivot] < current.value[current.pivot]:
                    #If there is no left child add the new node there
                    if current.left is None:
                        current.left = newnode
                        if current.pivot == self.k - 1:
                            newnode.pivot = 0
                        else:
                            newnode.pivot = current.pivot + 1
                        return
                    
                    #If there is a left child step to the left
                    else:
                        return step(current.left, data)
                    
                #If the data is more than the current value
                elif data[current.pivot] >= current.value[current.pivot]:
                    #If there is no right child add the new node there
                    if current.right is None:
                        current.right = newnode
                        if current.pivot == self.k - 1:
                            newnode.pivot = 0
                        else:
                            newnode.pivot = current.pivot + 1
                        return 
                    
                    #If there is a right child step to the right
                    else:
                        return step(current.right, data)
                else:
                    print(data[current.pivot], current.value[current.pivot])
                    print("Something jank is happening")
                    
            return step(self.root, data)
        


    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        
        def kdssearch(current, nearest, d):
            if current is None:
                return nearest, d
            
            x = current.value
            i = current.pivot
            
            #Check if current is closer to z than nearest
            if la.norm(x - z) < d:
                nearest = current
                d = la.norm(x - z)
            #Search to the left    
            if z[i] < x[i]:
                nearest, d = kdssearch(current.left, nearest, d)
                #Search to the right if needed
                if z[i] + d > x[i]:
                    neaerest, d = kdssearch(current.right, nearest, d)
            #Search to the right
            else:
                nearest, d = kdssearch(current.right, nearest, d)
                #Search to the left if needed
                if z[i] - d <= x[i]:
                    nearest, d = kdssearch(current.left, nearest, d)
                    
            return nearest, d
        
        node, d = kdssearch(self.root, self.root, la.norm(self.root.value - z))
        return node.value, d
    

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        """Initialize constructors"""
        self.n_neighbors = n_neighbors
        self.tree = None
        self.labels = None
        
    def fit(self, X, y):
        """Create tree and labels"""
        self.tree = KDTree(X)
        self.labels = y
        
    def predict(self, z):
        """Predict the most common label of the elements of X that are nearest to z."""
        d,x = self.tree.query(x = z, k = self.n_neighbors)
        
        return stats.mode(self.labels[x])[0][0]
        


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    #Extract the data
    data = np.load("mnist_subset.npz")
    X_train = data["X_train"].astype(np.float)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float)
    y_test = data["y_test"]
    """
    plt.imshow(X_test[0].reshape((28,28)),cmap="gray")
    plt.show()
    """
    #Create your KDtree and classification for the trains
    knc = KNeighborsClassifier(n_neighbors)
    knc.fit(X_train,y_train)
    correct = 0
    
    #Iterate throught the rows of X_test
    for i in range(len(y_test)):
        a = knc.predict(X_test[i])
        #See if the prediction actually equals the y_test label
        if a == y_test[i]:
            correct += 1
            
    accuracy = correct / len(y_test)
    
    return accuracy
    
    
    
    
    
#print(prob6(7))
    
    
if __name__ == "__main__":
    X = np.random.randint(1,10,size=(7,3))
    y = np.random.randint(1,10, size=(7,1))
    w = np.array([["Jared"],["Bryant"],["Ty"]])
    #print(X)
    #print(y)
    """
    (m,n) = np.shape(X)
    kdt = KDT()
    for i in range(m):
        kdt.insert(X[i])
    """    
    #print(kdt)
    #print(kdt.query([3,5,2]))
    
    knc = KNeighborsClassifier(4)
    
    knc.fit(X, y)
    #print(knc.predict([2,3,4]))
    
    pass

