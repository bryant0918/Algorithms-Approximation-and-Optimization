# binary_trees.py
"""Volume 2: Binary Trees.
Bryant McArthur
Math 321 Sec 002
October 7
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from matplotlib import pyplot as plt
import numpy as np
import time


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        
        def finding(data, currentnode):
            if currentnode is None:
                raise ValueError(str(data) + " is not in the tree.")
            if data == currentnode.value:
                return currentnode
            else:
                return finding(data, currentnode.next)
        
        return finding(data, self.head)
    

class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        node = BSTNode(data)
        
        if self.root == None:
            self.root = node
        else:
            def step(current,node):
                    
                if data == current.value:
                    raise ValueError("Data is already in the BST")
                    
                if data < current.value:
                    if current.left is None:
                        current.left = node
                        node.prev = current
                        return
                    else:
                        return step(current.left,node)
                else:
                    if current.right is None:
                        current.right = node
                        node.prev = current
                        return
                    else:
                        return step(current.right,node)
                
            return step(self.root,node)

    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """
        
        target = self.find(data)
        onright = False
        
        def step(current):
                if current.right is None:
                    return current
                else:
                    return step(current.right)
        
        """If the target is the root"""
        if target == self.root:
            
            #If the root has no children
            if (self.root.left == None) and (self.root.right == None):
                self.root = None
                
            #If the root has one child
            elif (self.root.left == None) or (self.root.right == None):
                
                if self.root.left == None:
                    child = self.root.right
                else:
                    child = self.root.left
                    
                self.root = child
                self.root.prev = None

            #If the root has two children
            else:    
                predecessor = step(target.left)
                value = predecessor.value
                
                self.remove(value)
                
                target.value = value
                
            return
        
        #If the target has no children
        elif (target.left is None) and (target.right is None):
            parent = target.prev
            if target.value < parent.value:
                parent.left = None
            else:
                parent.right = None
            return
        
        #If the target has one child
        elif (target.left == None) or (target.right == None):
            parent = target.prev
            if target.left == None:
                child = target.right
            else:
                child = target.left
            
            if target.value > parent.value:
                onright = True
            
            if onright is True:
                parent.right = child
                child.prev = parent
                
            else:
                parent.left = child
                child.prev = parent
            
            return
        
        #If the target is just in the middle somewhere
        else:
                
            predecessor = step(target.left)
            value = predecessor.value
            
            self.remove(value)
            
            target.value = value
            
            
            return 

    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)


# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    mylist = []
    #Open the infile to be read and read the lines to the stack
    with open("English.txt", 'r') as myfile:
        contents = myfile.readlines()
        for i in range(len(contents)):
            mylist.append(contents[i])
    
    
    n_times = [2**n for n in range(3,11)]
        
    #Create Lists
    SLLload = []
    BSTload = []
    AVLload = []
    SLLsearch = []
    BSTsearch = []
    AVLsearch = []
    
    
    for i in range(3,11):
        #Set n
        n = 2 ** i
        #Create elements list
        elements = np.random.choice(mylist, n, replace = False)
        #Choose 5 random elements from the list
        random_items = np.random.choice(elements, 5, replace = False)
        
        
        
        """Singly LInked List"""
        sll = SinglyLinkedList()
        #Loading Singly Linked List
        start = time.perf_counter()
        for i in elements:
            sll.append(i)
        SLLload.append( time.perf_counter() - start)
        #Searching Singly Linked List
        start = time.perf_counter()
        for i in range(5):
            sll.iterative_find(random_items[i])
        SLLsearch.append( time.perf_counter() - start)
        
        """BST"""
        bst = BST()
        #Loading BST
        start = time.perf_counter()
        for i in elements:
            bst.insert(i)
        BSTload.append(time.perf_counter() - start)
        #Searching BST
        start = time.perf_counter()
        for i in range(5):
            bst.find(random_items[i])
        BSTsearch.append(time.perf_counter() - start)
        
        """AVL"""
        avl = AVL()
        #Loading AVL
        start = time.perf_counter()
        for i in elements:
            avl.insert(i)
        AVLload.append(time.perf_counter() - start)
        #Searching AVL
        start = time.perf_counter()
        for i in range(5):
            avl.find(random_items[i])
        AVLsearch.append(time.perf_counter() - start)
        
    """Search Time Plots"""    
    #Linear plot
    ax1 = plt.subplot(221)
    ax1.plot(n_times,SLLsearch, 'b.-', linewidth=1.5, markersize=10, label="Singly Linked List")
    ax1.plot(n_times,BSTsearch, '.-',color="orange", linewidth=1.5, markersize=10, label="BST Search")
    ax1.plot(n_times, AVLsearch, 'k.-', linewidth=1.5, markersize = 10, label = "AVL Search")     
    ax1.legend(loc="upper left", fontsize = "x-small")
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
        
    #Logrithmic Plot
    ax2 = plt.subplot(222)
    ax2.loglog(n_times,SLLsearch, 'b.-', linewidth=1.5, markersize=10)
    ax2.loglog(n_times,BSTsearch, '.-',color="orange", linewidth=1.5, markersize=10)
    ax2.loglog(n_times,AVLsearch, 'k.-', linewidth = 1.5, markersize = 10)
    plt.xlabel("n",fontsize=14)
    
    """Load Time Plots"""
    #Linear plot
    ax3 = plt.subplot(223)
    ax3.plot(n_times,SLLload, 'b.-', linewidth=1.5, markersize=10, label="Singly Linked List")
    ax3.plot(n_times,BSTload, '.-',color="orange", linewidth=1.5, markersize=10, label="BST Load")
    ax3.plot(n_times, AVLload, 'k.-', linewidth=1.5, markersize = 10, label = "AVL Load")     
    ax3.legend(loc="upper left", fontsize = "x-small")
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
        
    #Logrithmic Plot
    ax4 = plt.subplot(224)
    ax4.loglog(n_times,SLLload, 'b.-', linewidth=1.5, markersize=10)
    ax4.loglog(n_times,BSTload, '.-',color="orange", linewidth=1.5, markersize=10)
    ax4.loglog(n_times,AVLload, 'k.-', linewidth = 1.5, markersize = 10)
    plt.xlabel("n",fontsize=14)
    
    
    ax1.set_title("Linear Search", fontsize=8)
    ax2.set_title("Logrithimic Search", fontsize=8)
    ax3.set_title("Linear Load", fontsize=8)
    ax4.set_title("Logrithimic Load", fontsize=8)
    
    plt.suptitle("Time required to load and search different data structures")
    
    plt.tight_layout()
    plt.show()        




if __name__ == "__main__":
    tree = BST()
    for i in [1,2,3,4,5,6]:
        tree.insert(i)
    #print(tree)
    tree.remove(1)
        
    #print(str(tree))
    
    prob4()
    pass