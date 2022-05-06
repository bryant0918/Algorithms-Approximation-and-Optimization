# regular_expressions.py
"""Volume 3: Regular Expressions.
Bryant McArthur
Math 323
February 8, 2022
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile('python')
    
    
# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r'\^\{\@\}\(\?\)\[\%\]\{\.\}\(\*\)\[\_\]\{\&\}\$')

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r'^(Book|Mattress|Grocery) (store|supplier)$')


# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    
    #pyid = re.compile(r'[_a-zA-Z][\w_]*')
    
    return re.compile(r'^[_a-zA-Z]\w*[\b]*\t*(|=[\s]*(\d*(|\.\d*)|\'[^\']*\'|[_a-zA-Z]\w*))$')
   

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    #Find the pattern
    pattern = re.compile(r"(^\s*(if|elif|else|for|while|try|except|finally|with|def|class)[^\n]*)", re.MULTILINE)
    #Add a colon
    code = pattern.sub(r"\1:", code)
    
    return code


# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    #Initialize empty dictionary and regex patterns
    diction = dict()
    Pname = re.compile(r'^[a-zA-Z]+\s[A-Z]?\.?\s?[a-zA-Z]+')
    Pbirthday = re.compile(r'(\d+/\d+/\d+)')
    m_d_y = re.compile(r'(\d+)')
    Pphone = re.compile(r'(\d{3}(-|\))).*(\d{3}-\d{4})')
    Pemail = re.compile(r'[^\s]*\@[^\s]*')
    
    #Write the filelines to a list
    with open(filename, 'r') as myfile:
        contacts = myfile.readlines()
        
    #Read the contacts by line
    for contact in contacts:
        
        #Find the name
        name = Pname.findall(contact)
        
        #Find the Birthday
        birthstring = Pbirthday.findall(contact)
        
        #Check if the list containing the birthday is not empty
        if birthstring:
            
            #Clean the birthday string
            mdy = m_d_y.findall(birthstring[0])
            
            if len(mdy[0]) < 2:
                mdy[0] = '0'+mdy[0]
            if len(mdy[1]) < 2:
                mdy[1] = '0'+mdy[1]
            if len(mdy[2]) == 2:
                mdy[2] = '20'+mdy[2]
            
            birthday = str(mdy[0]+"/"+mdy[1]+"/"+mdy[2])
            
        #If empty set to None
        else:
            birthday = None
        
        #Find the phone string
        phonestring = Pphone.findall(contact)
        
        #Clean the phone string if not empty else set to None
        if phonestring:
            phonestring = phonestring[0]
            p1 = phonestring[0]
            p1 = p1[0:3]
            phone = "("+p1+")"+phonestring[2]
        else:
            phone = None
        
        #Find the email string
        email = Pemail.findall(contact)
        
        #Set email to email or to None if empty
        if email:
            email = email[0]
        else:
            email = None
        
        #Write everything to the given name
        diction[name[0]] = {"birthday":birthday,"email":email,"phone":phone}
   
    return diction

