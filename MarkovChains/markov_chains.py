# markov_chains.py
"""Volume 2: Markov Chains.
Bryant McArthur
Sec 002
November 4
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        self.A: A (n,n) matrix that should be column-stochaastic
        self.labels: A list of the labels
        self.dictionary: A dictionary where the keys are the rows of A
            and the values are the states
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        #make a dictionary
        self.dictionary = {}
        
        #Check if A is column stochastic
        if np.allclose(A.sum(axis=0), np.ones(A.shape[1])) == False:
            raise ValueError("A is not column stochastic")
            
        #Create your attributes
        if states == None:
            m,n = A.shape
            states = [i for i in range(n)]
            
        for i, state in enumerate(states):
            self.dictionary[state] = i
            
        self.A = A
        self.labels = states
            
        

    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        
        index = self.dictionary[state]
        #Take a random available entry
        newcolumn = np.random.multinomial(1, np.array(self.A[:,index]))
        
        newindex = np.argmax(newcolumn)
        
        #print(self.labels[newindex])
        
        return self.labels[newindex]
        

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        
        labels = []
        state = start
        #Walk through the nodes N-1 times calling transition
        for i in range(N-1):
            labels.append(self.transition(state))
            state = labels[i]
            
            
        labels.insert(0,start)
        
        return labels
        

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        
        labels = []
        state = start
        #Walk through the nodes until you reach stop
        while state != stop:
            #print(state)
            labels.append(self.transition(state))
            state = labels[len(labels) - 1]
            
        #Insert start to the paths visited
        labels.insert(0,start)
        
        return labels
        

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        
        m,n = self.A.shape
        
        x = np.random.random(n)
        
        #Normalize x
        x /= sum(x)
        x1 = self.A @ x
        
        # Iterate at least maxiter times
        for i in range(maxiter):
            
            x = x1
            x1 = self.A @ x
            
            dif = []
            for k in range(n):
                dif.append(abs(x[k]-x1[k])) 
            sm = sum(dif)
            
            #If the sum is less than the tolerance it is good enough
            if sm < tol:
                break
            
            #If we never get below the tolerance we say it doesn't converge
            if i == (maxiter - 1):
                raise ValueError("Ak does not converge")
            
            
        return x
    
    
        

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        
        #Open and read the file lines
        with open(filename, 'r') as myfile:
            '''Algorithm 1.1'''
            sentences = myfile.read().split('\n') 
            uniquewords = set()
            # get the set of unique words
            for sentence in sentences:
                words = sentence.split()
                uniquewords.update(words)  
            
            # Add start and stop
            uniquewords = list(uniquewords)
            uniquewords = ['$tart'] + uniquewords + ['$top']  
            n = len(uniquewords)
            
            #Create my dictionary of indices and uniquewords
            dictionary = {word: i for i, word in enumerate(uniquewords)}
            
            # initialize the transition matrix
            M = np.zeros((n, n))  
            
            #Iterate through sentences
            for sentence in sentences:
                words = sentence.split()
                words = ['$tart'] + words + ['$top']
                # add 1 to the entry of the transition matrix that corresponds to transitioning x to y
                for i in range(len(words) - 1):
                    x, y = words[i], words[i + 1]
                    j, k = dictionary[x], dictionary[y]
                    M[k][j] += 1  
                
            #Transition stop to start
            M[-1][-1] = 1  
            #Normalize each column
            M /= np.sum(M, axis=0) 
            #Super impose to make things easier
            super().__init__(M, uniquewords)
        
        

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        #Get a path from start to stop
        sentence = self.path("$tart", "$top")
        
        #Remove start and stop cuz we don't want to read that
        sentence.remove("$tart")
        sentence.remove("$top")
        
        yodatalk = ' '.join(sentence)
        
        
        return yodatalk
      
if __name__ == "__main__":
    A = np.array([[.7,.6],[.3,.4]])
    mc = MarkovChain(A)
    #print(mc.transition(0))
    print(mc.walk(0,10))
    #print(mc.path(0,0))
    #print(mc.steady_state(maxiter=100))
    
    yoda = SentenceGenerator("The Book of Mormon.txt")
    for _ in range(10):
        print(yoda.babble())
    
    
    pass
    
