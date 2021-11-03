# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Bryant McArthur
Sec 002
Halloween
"""

import queue
import networkx as nx
from matplotlib import pyplot as plt

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        
        if n not in self.d.keys():
            self.d[n] = set()
            

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        if u not in self.d.keys():
            self.d[u] = set()
        if v not in self.d.keys():
            self.d[v] = set()
            
        #Add edge on both sides
        self.d[u].add(v)
        self.d[v].add(u)
        

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        if n not in self.d.keys():
            raise KeyError("Node is not in the graph")
            
        else:
            self.d.pop(n)
            for node in self.d:
                self.d[node].discard(n)
        

    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        
        if (u or v) not in self.d.keys():
            raise KeyError("Node not in dictionary")
        if self.d[u] == set():
            raise KeyError("No edge exists between the nodes")
            
        #remove edge from both sides
        self.d[u].remove(v)
        self.d[v].remove(u)
            
        

    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        #Initialize stuff
        Q = queue.Queue()
        V = []
        M = set()
        
        if source not in self.d:
            raise KeyError("The source node is not in the graph")
        
        #Add the first node to Queue and Marked
        Q.put(source)
        M.add(source)
        
        #Traverse as long as Q has a node in it
        while not Q.empty():
            current = Q.get()
            V.append(current)
            for value in self.d[current]:
                if value not in M:
                    Q.put(value)
                    M.add(value)
        
        return V
                    
        #You may want to do M.add(source/current) and/or Q.append(source/current)
        

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        #Call traverse function and truncate it
        V = self.traverse(source)
        i = V.index(target)
        return V[:i+1]
        


# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        #Set Constructors
        self.movies = set()
        self.actors = set()
        self.G = nx.Graph()
        #Open read and close the file
        inputfile = open(filename, 'r', encoding = 'utf-8')
        contents = inputfile.readlines()
        inputfile.close()
        #Iterate through the lines
        for line in contents:
            movie_list = line.strip().split('/')
            #In each line add the first element to the movie list and the rest to the actors
            for i in movie_list:
                if i == movie_list[0]:
                    self.movies.add(i)
                else:
                    self.actors.add(i)
                    self.G.add_edge(movie_list[0], i)
                    

    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        #Only do this if there is a path and then call the nx.shorest_path
        if nx.has_path(self.G, source, target) == True:
            path = nx.shortest_path(self.G, source, target)
            steps = nx.shortest_path_length(self.G, source, target)
            #Take out the movies between every actor
            steps = steps // 2
            return path, steps
        else:
            raise KeyError("There is not a path between source and", str(target))
            
        
        

    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        #Find shortest path between target and everyone else
        lengths = nx.shortest_path_length(self.G, target)
        sums = []
        
        #Iterate through every actor's shortest bath
        for i in lengths.keys():
            
            if lengths[i] % 2 == 0:
                sums.append(lengths[i] // 2)
            
        avg = sum(sums) / len(sums)
        
        #Plot a histogram, looks like most people are connected by 2 or 3 movies
        plt.hist(sums, bins = [i-.5 for i in range(8)])
        plt.title("Path lengths of every actor to", str(target))
        plt.ylabel("Number of actors")
        plt.xlabel("Length of shortest path")
        plt.show()        
        
        return avg
            
        
        
        
if __name__ == "__main__":
    graph = Graph()
    graph.add_edge('u', 'v')
    graph.add_node('n')
    graph.add_edge('u','n')
    #print(str(graph))
    #print(graph.traverse('u'))
    #print(graph.shortest_path('u','n'))
    test = MovieGraph()
    #print(len(test.movies))
    #print(test.average_number("Kevin Bacon"))
    #print(test.path_to_actor("Kevin Bacon", "Tom Holland"))
    
    pass
    
