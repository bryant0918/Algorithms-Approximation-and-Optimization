# object_oriented.py
"""Python Essentials: Object Oriented Programming.
Bryant McArthur
Math 321 Sec 002
September 9
"""

import numpy as np

class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            color (str): the color of the backpack.
            max_size(int): The max capacity of the backpack defaulted to 5.
        """
        
        self.name = name
        self.color = color
        self.max_size = max_size
        self.contents = []

    def put(self, item):
        """Add an item to the backpack's list of contents unless it's full"""
        if len(self.contents) < self.max_size:
            self.contents.append(item)
        else:
            print("No Room!")

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)
    def dump(self):
        """Dumps all the contents of the backpack."""
        self.contents = []
        

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)
    
    def __eq__(self, other):
        """Compares 2 backpacks and returns true if both backpacks have the
        same name, color, and number of contents."""
        
        if (self.name, self.color, len(self.contents)) == (other.name, other.color, len(other.contents)):
            return True
        else:
            return False
    
    def __str__(self):
        """Prints the owner, color, size, max_size, and contents of the backpack."""
        
        print("Owner: \t", self.name, "\n"
              "Color: \t", self.color, "\n"
              "Size: \t", len(self.contents), "\n"
              "Max Size: \t", self.max_size, "\n"
              "Contents: \t", self.contents)
        
        
        

def test_backpack():
    testpack = Backpack("Bryant", "black")
    if testpack.name != "Bryant":
        print("Backpack.name assigned incorrectly")
    for item in ["pencil", "pen", "paper", "computer"]:
        testpack.put(item)
    print("Contents: ", testpack.contents)
    
test_backpack()

# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A Jetpack object class. Inherits from the Backpack class.
    A Jetpack is a backpack that helps you fly.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        fuel (int): The amount of fuel in your jetpack.
    """
    def __init__(self, name, color, max_size=2, fuel=10):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A Jetpack only holds 2 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
            fuel (int): The amount of fuel in your jetpack.
        """
        Backpack.__init__(self, name, color, max_size)
        self.fuel = fuel
        
        
    def fly(self, burn):
        """Accept an amount of fuel you want to burn and decrease your current
        fuel amount by that much. If you do not have enough fuel in your
        backpack print "Not enough fuel!"
        """
        if burn <= self.fuel:
            self.fuel = self.fuel - burn
        else:
            print("Not enough fuel!")
        
    def dump(self):
        """Dumps all the contents of the jetpack including the fuel."""
        self.contents = []
        self.fuel = 0

# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:
    
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag
        
    def conjugate(self):
        
        return ComplexNumber(self.real, self.imag*-1)
    
    def __str__(self):
        if self.imag < 0:
            print(self.real,"-",self.imag,"j")
        else:
            print(self.real, "+", self.imag, "j")
        
    def __abs__(self):
        
        return np.sqrt((self.real*self.real)+(self.imag*self.imag))
    
    def __eq__(self):
        
        if (self.real, self.imag) == (self.real, self.imag):
            return True
        else:
            return False
    
    def __add__(self, other):
        newreal = self.real + other.real
        newimag = self.imag + other.imag
        return ComplexNumber(newreal, newimag)

    def __sub__(self, other):
        newreal = self.real - other.real
        newimag = self.imag - other.imag
        return ComplexNumber(newreal, newimag)
    
    def __mul__(self, other):
        newreal = self.real * self.real
        newimag = (self.imag * self.imag) * -1
        return ComplexNumber(newreal, newimag)
    
    def __truediv__(self,other):
        
        







































