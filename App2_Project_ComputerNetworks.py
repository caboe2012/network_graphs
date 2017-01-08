'''Algorithmic Thinking 1 - Module 2 : Application 2 '''
"""
Provided code for Application portion of Module 2
"""

# general imports
import urllib2
import random
import time
import math
import matplotlib.pyplot as plt

# CodeSkulptor import
#import simpleplot
#import codeskulptor
#codeskulptor.set_timeout(60)

# Desktop imports
#import matplotlib.pyplot as plt


############################################
# Provided code

def copy_graph(graph):
    """
    Make a copy of a graph
    """
    new_graph = {}
    for node in graph:
        new_graph[node] = set(graph[node])
    return new_graph

def delete_node(ugraph, node):
    """
    Delete a node from an undirected graph
    """
    neighbors = ugraph[node]
    ugraph.pop(node)
    for neighbor in neighbors:
        ugraph[neighbor].remove(node)
    
def targeted_order(ugraph):
    """
    Compute a targeted attack order consisting
    of nodes of maximal degree
    
    Returns:
    A list of nodes
    """
    # copy the graph
    new_graph = copy_graph(ugraph)
    
    order = []    
    while len(new_graph) > 0:
        max_degree = -1
        for node in new_graph:
            if len(new_graph[node]) > max_degree:
                max_degree = len(new_graph[node])
                max_degree_node = node
        
        neighbors = new_graph[max_degree_node]
        new_graph.pop(max_degree_node)
        for neighbor in neighbors:
            new_graph[neighbor].remove(max_degree_node)

        order.append(max_degree_node)
    return order
    


##########################################################
# Code for loading computer network graph

NETWORK_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_rf7.txt"


def load_graph(graph_url):
    """
    Function that loads a graph given the URL
    for a text representation of the graph
    
    Returns a dictionary that models a graph
    """
    graph_file = urllib2.urlopen(graph_url)
    graph_text = graph_file.read()
    graph_lines = graph_text.split('\n')
    graph_lines = graph_lines[ : -1]
    
    print "Loaded graph with", len(graph_lines), "nodes"
    
    answer_graph = {}
    for line in graph_lines:
        neighbors = line.split(' ')
        node = int(neighbors[0])
        answer_graph[node] = set([])
        for neighbor in neighbors[1 : -1]:
            answer_graph[node].add(int(neighbor))

    return answer_graph


##########################################################
# Code for creating UNDIRECTED ER graphs (from Homework 1, Pseudo-Code on Q10)

def undirected_ER(n,p):
	temp = {}
	V = range(0,n)
	for node_i in V:
		temp[node_i] = set()
	for node_i in V:
		for node_j in V:
			a = random.random()
			if node_i == node_j:
				pass
			elif a < p and node_j not in temp[node_i]:
				temp[node_i].add(node_j)
				temp[node_j].add(node_i)
	return temp

#test1 = undirected_ER(5,0.5)
#for k,v in test1.items():
#	print k, ":", v

##########################################################
# Code for creating UPA Graphs

def make_complete_ugraph(num_nodes):
	''' returns a dictionary corresponding 
	to a complete undirected graph with the specified '''
	_undirected_graph = {}
	for _node_row in range(num_nodes):
		_num_edges = set([])
		for _node_col in range(num_nodes):
			if _node_row != _node_col:
				_num_edges.add(_node_col)
		_undirected_graph[_node_row] = _num_edges
	return _undirected_graph

def UPA(n,m):
	'''Compute randomly generated additional nodes from m to n
	starting from a complete graph of m nodes.'''
	V = range(0,m)
	E = make_complete_ugraph(m)
	total_in_degrees = m*(m-1)
	randomly_added_nodes = 0
	for i in range(m,n):
		total_in_degrees += randomly_added_nodes
		V_prime = 0

##########################################################
# Code to determine optimal p for the undirected ER graph

def random_order(ugraph):
	'''Takes a graph and returns a list of the nodes in the graph in some random order.'''
	V = ugraph.keys()
	n = len(V)
	random_node_order = random.sample(V,n)
	return random_node_order 



##########################################################
# Code to create a legend in matplotlib.pyplot

import matplotlib.pyplot as plt

def plot_resilience(er_cc,upa_cc,compnet_cc):
    """
    Plot an example with two curves with legends
    """
    xvals = range(len(upa_cc))
    yvals_er = er_cc
    yvals_upa = upa_cc
    yvals_cn = compnet_cc

    plt.plot(xvals, yvals_er, '-b', label='ER Graph, p = 0.002')
    plt.plot(xvals, yvals_upa, '-r', label='UPA Graph, m = 3')
    plt.plot(xvals, yvals_cn, '-g',label = 'Computer network Graph')
    plt.axis([0,1250,0,1250])
    plt.xlabel('Nodes removed from undirected graph')
    plt.ylabel('Size of largest Connected Component in Graph')
    plt.title("Resilience of targeted attack on ER, UPA and Computer Network Graphs")
    plt.legend(loc='upper right')
    plt.show()

#plot_resilience()

##########################################################
"""
Provided code for application portion of module 2

Helper class for implementing efficient version
of UPA algorithm
"""

import random

n = 1239 # number of nodes from example computer network
m = 3047 # number of edges from example computer network


class UPATrial:
    """
    Simple class to encapsulate optimizated trials for the UPA algorithm
    
    Maintains a list of node numbers with multiple instance of each number.
    The number of instances of each node number are
    in the same proportion as the desired probabilities
    
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a UPATrial object corresponding to a 
        complete graph with num_nodes nodes
        
        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]


    def run_trial(self, num_nodes):
        """
        Conduct num_nodes trials using by applying random.choice()
        to the list of node numbers
        
        Updates the list of node numbers so that each node number
        appears in correct ratio
        
        Returns:
        Set of nodes
        """
        
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for _ in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        
        # update the list of node numbers so that each node number 
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        for dummy_idx in range(len(new_node_neighbors)):
            self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))
        
        #update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors

def make_complete_graph(num_nodes):
    ''' returns a dictionary corresponding 
    to a complete directed graph with the specified '''
    _directed_graph = {}
    for _node_row in range(num_nodes):
        _num_edges = set([])
        for _node_col in range(num_nodes):
            if _node_row != _node_col:
                _num_edges.add(_node_col)
        _directed_graph[_node_row] = _num_edges
    return _directed_graph

def create_UPA_graph(m,n):
    complete_ugraph = make_complete_graph(m)
    UPA_obj = UPATrial(m)
    for _new_node in range(m,n):
        _new_node_edges = UPA_obj.run_trial(m)
        complete_ugraph[_new_node] = _new_node_edges
        for _edge in _new_node_edges:
            complete_ugraph[_edge].add(_new_node)
    return complete_ugraph

def optimal_m(trials, m):
    results = []
    for trial in range(trials):
        total_edges_per_trial = 0
        check = create_UPA_graph(m,1239)
        for node in check.keys():
            total_edges_per_trial += len(check[node])
        ans = total_edges_per_trial/2.
        final = ans/1239.
        results.append(final)
    return results
#print optimal_m(10,3)

#UPA = create_UPA_graph(3,1239)
#print test_UPA

def random_order(ugraph):
    '''Takes a graph and returns a list of the nodes in the graph in some random order.'''
    V = ugraph.keys()
    n = len(V)
    random_node_order = random.sample(V,n)
    return random_node_order 
##########################################################
def FastTargetedOrder(ugraph):
	DegreeSets = [0]*len(ugraph)
	for k in range(len(ugraph)):
		DegreeSets[k] = set()
	for i in range(len(ugraph)):
		d = len(ugraph[i])
		DegreeSets[d].add(i)
	L = ['']*len(ugraph)
	i = 0
	for k in range(len(ugraph)-1,-1,-1):
		while DegreeSets[k] != set():
			u = random.sample(DegreeSets[k],1)
			DegreeSets[k] = DegreeSets[k] - set(u)
			neighbors = ugraph[u[0]]
			for neighbor in neighbors:
				d = len(ugraph[neighbor])
				DegreeSets[d] = DegreeSets[d] - set([neighbor])
				DegreeSets[d-1].add(neighbor)
			L[i] = u[0]
			i += 1
			delete_node(ugraph,u[0])
	return L
#UPA1 = create_UPA_graph(5,1239)
#print FastTargetedOrder(UPA1)
##########################################################
def compute_running_times(upa_network_list,algo):
	results = []
	for network in upa_network_list:
		start = time.time()
		algo(network)
		run_time = time.time()-start
		results.append(run_time)
	return results


##########################################################
def plot_UPA_times(upa_slow,upa_fast):
    """
    Plot an example with two curves with legends
    """
    xvals = range(10,1000,10)
    yvals_slow = upa_slow
    yvals_fast = upa_fast
    plt.plot(xvals, yvals_slow, '-b', label='Targeted Order')
    plt.plot(xvals, yvals_fast, '-r', label='Fast Targeted Order')
#    plt.axis([0,1050,0,1050])
    plt.xlabel('Number of Nodes in the UPA Graph')
    plt.ylabel('Running Time of Functions (in seconds)')
    plt.title("Running time in Desktop Python of targeted order and fast targeted order")
    plt.legend(loc='upper left')
    plt.show()
    