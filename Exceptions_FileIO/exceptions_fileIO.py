# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
Bryant McArthur
September 16, 2021
Math 321 sec 002
"""

from random import choice
import numpy as np


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:
    
    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError("The number must be 3 digits!")
    a1 = int(step_1[0])*100
    b1 = int(step_1[1])*10
    c1 = int(step_1[2])
    first = a1+b1+c1
    
    if abs(int(step_1[0]) - int(step_1[2])) < 2:
        raise ValueError("The first and third digits must differ by 2 or more!")
    
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    a2 = int(step_2[0])*100
    b2 = int(step_2[1])*10
    c2 = int(step_2[2])
    second = a2+b2+c2
    if (step_2[0],step_2[1],step_2[2]) != (step_1[2], step_1[1], step_1[0]):
        raise ValueError("That is not the first number backwards!")
    
    step_3 = input("Enter the positive difference of these numbers: ")
    a3 = int(step_3[0])*100
    b3 = int(step_3[1])*10
    c3 = int(step_3[2])
    third = a3+b3+c3
    if abs(first-second) != third:
        raise ValueError("That is not the positive difference of these numbers!")
    
    step_4 = input("Enter the reverse of the previous result: ") 
    if (step_4[0],step_4[1],step_4[2]) != (step_3[2], step_3[1], step_3[0]):
        raise ValueError("That is not the reverse of the previous number!")
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")

#arithmagic()

# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the 
    program is running, the function should catch the exception and 
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    
    walk = 0
    directions = [1, -1]
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print("Process interrupted at iteration",i)
        return walk
    print("Process Completed")
    return walk

#print(random_walk())


# Problems 3 and 4: Write a 'ContentFilter' class.
"""Class for reading in file
        
    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
        
    """
class ContentFilter(object):   
    # Problem 3
    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        valid = False
        while not valid:
            try:
                with open(filename, 'r') as myfile:
                    self.filename = myfile.name
                    self.contents = myfile.read()
                valid = True
            except:
                filename = input("Please enter a valid file name:")
                
 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""
        if mode not in ['w', 'x', 'a']:
            raise ValueError("the mode is invalid")

    def uniform(self, outfile, mode='w', case='upper'):
        """Write the data to the outfile in uniform case."""
        
        self.check_mode(mode)
        
        data = self.contents
        if case == 'upper':
            data = data.upper()
        elif case == 'lower':
            data = data.lower()
        else:
            raise ValueError("Must be upper or lower case")
        
        with open(outfile, mode) as myfile:
            myfile.write(data)


    def reverse(self, outfile, mode='w', unit='line'):
        """Write the data to the outfile in reverse order."""
        self.check_mode(mode)
        
        data = self.contents
        newdata = []
        
        #Switching all the words on each line
        if unit=="word":
            
            newdata = data.split('\n')
            
            for i in range(len(newdata)):
                superdata = newdata[i].split()
                superdata.reverse()
                newdata[i] = " ".join(superdata)
            
            x = "\n".join(newdata)
        
        #switching just the lines
        elif unit=='line':
            newdata = data.split('\n')
            newdata.reverse()
            x = "\n".join(newdata)
        else:
            raise ValueError("Must be line or word")
        
        #Write the data to the outfile
        with open(outfile, mode) as myfile:
            myfile.write(x)
        
        
    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""
        self.check_mode(mode)
        
        #Open the contents and immediately put it into an array
        with open(self.filename, 'r') as inputfile:
            with open(outfile, mode) as myfile:
                x = np.loadtxt(inputfile, dtype = str)
                xtrans = x.T
        
        #Take the transpose of the data as a list
        data = xtrans.tolist()
        
        #Join the words
        for i in range(len(data)):
            newdata = data[i]
            data[i] = " ".join(newdata)
        #Join the lines
        x = "\n".join(data)
        
        #Write it to the outfile
        with open(outfile, mode) as myfile:
            myfile.write(x)
        
        

    def __str__(self):
        """String representation: info about the contents of the file."""
        
        s = ""
        s = s + "Source file:\t\t\t"+self.filename + "\n"
        s = s + "Total Characters: \t\t" + str(len(self.contents)) + "\n"
        s = s + "Alphabetic characters: \t" + str(sum(k.isalpha() for k in self.contents)) + "\n"
        s = s + "Numerical chcaracters: \t" + str(sum(k.isdigit() for k in self.contents)) + "\n"
        s = s + "Whitespace characters: \t" + str(sum(k.isspace() for k in self.contents)) + "\n"
        s = s + "Number of lines: \t\t" + str(self.contents.count("\n"))
        
        return s
        



if __name__ == "__main__":
    cf = ContentFilter("hello_world.txt")
    print(cf.contents)
    print(cf.filename)
    #cf.uniform("newfile.txt")
    cf.check_mode('w')
    cf.reverse("reverseword.txt", 'w','word')
    #cf.transpose("transpose.txt")
    print(str(cf))
    pass


