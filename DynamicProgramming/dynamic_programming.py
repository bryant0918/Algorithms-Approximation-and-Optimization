# dynamic_programming.py
"""Volume 2: Dynamic Programming.
Bryant McArthur
Sec 002
March 31
"""

import numpy as np
from matplotlib import pyplot as plt


def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    T = reversed([i+1 for i in range(N)])
    V = np.zeros(N)
    
    for t in T:
        if t == N:
           pass
        else:
            V[t-1] = t/(t+1)*V[t] + 1/N
        
    return np.max(V), np.argmax(V)+1

#print(calc_stopping(4))


# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    percentage = []
    probability = []
    domain = list(range(3,M+1))
    
    for i in range(3,M+1):
        value, index = calc_stopping(i)
        percentage.append(index/i)
        probability.append(value)
        
    plt.plot(domain, probability)
    plt.plot(domain, percentage)
    plt.show()
    
    return percentage[-1]
    
#print(graph_stopping_times(1000))        


# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    w = np.array([i/N for i in range(N+1)])
    
    C = np.array(u(w))
    c = list(C)
    
    for i in range(len(w)-1):
        c.pop()
        c.insert(0,0)
        C = np.vstack((C,c))
    
    return C.T
    
#print(get_consumption(5))
    


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    w = np.array([i/N for i in range(N+1)])
    
    A = np.ones((N+1,T+1))
    
    C = get_consumption(N, u)
    
    A = C@A
    
    A[:,:-1] = np.zeros((N+1,T))
    
    P = np.zeros_like(A)
    
    #Problem 5
    for t in range(T, -1, -1):
        A[:,-1] = u(w)
        P[:,-1] = w
        
        CV =  np.array([[u(w[i] - w[j]) + B*A[j,t] if j <= i else 0 for j in range(N+1)] for i in range(N+1)])
        
        row_max = np.max(CV,axis=1)
        row_idx_j = np.argmax(CV,axis=1)
        
        P[:,t-1] = np.array(w) - np.array([w[j] for j in row_idx_j] )
        A[:,t-1] = row_max
            
    A[:,-1] = u(w)
    P[:,-1] = w
    
    return A,P

A,P = eat_cake(3,4,.9)

#print(P)
    


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    
    A,P = eat_cake(T,N,B,u)
    
    policy = np.zeros(T+1)
    policy[0] = P[N,0]
    
    for t in range(1,T+1):
        
        remaining = N - (N*sum(policy[:t]))
        #print(remaining)
        policy[t] = P[int(remaining), t]
        
    return policy

#print(find_policy(3,4,.9))
