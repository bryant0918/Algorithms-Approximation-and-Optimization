# shell2.py
"""Volume 3: Unix Shell 2.
Bryant McArthur
Math 321 Sec 002
11.11.21
"""

import os
from glob import glob
import subprocess

# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    #Find anyting with a certain file pattern
    filenames = glob('**/' + file_pattern, recursive = True)
    matchingfiles = []
    
    #Open them up and see if they have a certain string then append if they do
    for file in filenames:
        with open(file, 'r') as myfile:
            if target_string in myfile.read():
                matchingfiles.append(file)
    
    return matchingfiles


# Problem 4
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    #Find all files in subdirectories
    allfiles = glob("**/*.*", recursive = True)
    
    #Get a list of the files in order by size to return
    sizes = [[f,os.path.getsize(f)] for f in allfiles]
    sizes.sort(key=lambda x:x[1], reverse=True)
    listfiles = [size[0] for size in sizes[:n]]
    smallest = listfiles[-1]
    
    
    lines = subprocess.check_output(["wc","-l", smallest]).decode()
    lines = lines.rstrip().split()
    lines = lines[0]
    
    #Create a new file called smallest.txt and write the lines to it.
    myfile = "smallest.txt"
    with open(myfile, mode='w') as outfile:
        outfile.write(lines)
    
    return listfiles
    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting lists each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (list): list of integers from 0 to the number n
       twoCounter (list): list of integers created by counting down from n by two
       threeCounter (list): list of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter

if __name__ == "__main__":
    x = grep("range", "*.py")
    print(x)
    print(largest_files(4))
    pass
    
    
    