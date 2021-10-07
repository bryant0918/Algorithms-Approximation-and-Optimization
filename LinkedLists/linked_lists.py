# linked_lists.py
"""Volume 2: Linked Lists.
Bryant McArthur
Sep 30, 2021
Math 345 Sec 002
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        
        if type(data) not in [int, str, float]:
            raise TypeError("Must be type int, float, or str")
        
        self.value = data



class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.


# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None               #Set the beginning head, tail and length
        self.tail = None
        self.length = 0

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.length += 1
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node
            self.length += 1
            

    # Problem 2
    def find(self, data,):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        
        length = self.length
        node = self.head
        
        #Raise error if the list is empty
        if node is None:
            raise ValueError("The list is empty")
            
        #Iterate through the list to find the data
        for i in range(length):
            if node.value == data:
                return node
            else:
                node = node.next
                
            #Raise error if the data is not in the list    
            if node is None:
                raise ValueError("The list does not contain the data")               

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        
        #Raise an IndexError if a bad index is given
        if (i < 0) or (i > self.length):
            raise IndexError("i must be from 0 to", self.length)
            
        node = self.head
        
        #retrieve the i-th node
        for k in range(i):
            node = node.next
        
        return node
            
            

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.length

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        
        
        node = self.head
        mystring = []
        #While there exists a node
        while node:
            mystring.append(node.value)
            node = node.next
            
        return repr(mystring)

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        
        target = self.find(data)
        head = self.head
        
        #If we are removing the head
        if target == head:
            self.head = target.next
            head = None
            return
        
        #If we are removing any other node
        else:
            target.prev.next = target.next
            target.next.prev = target.prev
        
        #Always adjust the length
        self.length -= 1
        
        

    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        newnode = LinkedListNode(data)
        oldnode = self.get(index)
        
        #If we are inserting at the head
        if index == 0:
            self.head.prev = newnode
            self.head = newnode
            self.head.next = oldnode
            
        #If we are inserting at the tail
        elif index == len(L):
            self.append(data)
            
        #If we are inserting in the middle
        else:
            oldnode.prev.next = newnode
            oldnode.prev = newnode
            newnode.next = oldnode
        
        #Always adjust the length
        self.length += 1


# Problem 6: Deque class.
class Deque(LinkedList):
    """Deque class that inherits traits from LinkedList class."""
    
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None        #Inherited from the Linked List class
        self.tail = None
        self.length = 0
    
    def pop(self):
        """
        Remove the last node in the list and return its data

        Raises
        ------
        ValueError
            If the deque is empty

        Returns
        -------
        Value of the popped node
        """
        
        #If the deque is empty
        if self.length == 0:
            raise ValueError("The deque is empty")
        
        #We are removing the tail node, and returning it's value
        else:
            poppednode = self.tail            
            self.tail.prev.next = None
            self.tail = self.tail.prev
            self.length -= 1            #Always remember to adjust the length
            
            return poppednode.value
        
    def popleft(self):
        """
        Remove the first node in the deque and return its data.  Raises a 
        ValueError if the deque is empty when we call the remove function.

        Returns
        -------
        Value of the popped node
        """
        
        poppednode = self.head
        LinkedList.remove(self, self.head.value)
        
        return poppednode.value
    
    def appendleft(self, data):
        """
        Insert a new node at the beginning of the deque.

        Parameters
        ----------
        data : 
            The value of the node you would like to insert

        Returns
        -------
        None.

        """
        LinkedList.insert(self,0,data)
        
    def remove(*args, **kwargs):
        """ Disabling the remove function from the Inherited Class"""
        raise NotImplementedError("Use pop(), or popleft() for removal")
        
    def insert(*args, **kwargs):
        """Disabline the insert function from the inherited class"""
        raise NotImplementedError("Use append(), or appendleft() for insertion")
        

# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    from collections import deque
    
    #Create a deque
    mystack = deque()
    
    #Open the infile to be read and read the lines to the stack
    with open(infile, 'r') as myfile:
        contents = myfile.readlines()
        for i in range(len(contents)):
            mystack.appendleft(contents[i])
     
    #Open the outfile to be written and write the lines of the stack in LIFO order.
    with open(outfile, 'w') as myfile:
        myfile.write("\n".join(mystack))
        
    
    





if __name__ == "__main__":
    
    L = LinkedList()
    
    for x in ['a', 'b', 'c', 'd', 'e']:
        L.append(x)
    
    #node = L.find('c')
    #node = L.get(3)
    
    L.remove('b')
    
    #print(str(L))
    
    D = Deque()
    for x in [1]:
        D.append(x)
    D.appendleft(3)
    print(D)
    print(D.pop())
    print(D)
    print(D.popleft())
    print(D)
        
    
    prob7("english.txt","outfile.txt")
    
    pass
        
        
