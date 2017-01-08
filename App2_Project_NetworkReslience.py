''' Algorithmic_Thinking 1 - Module 2 '''
from collections import deque
import random
#popleft to remove and return leftmost element(the head of the queue)
#append to add new entry

GRAPH5 = {1: set([2, 4, 6, 8]),
          2: set([1, 3, 5, 7]),
          3: set([4, 6, 8]),
          4: set([3, 5, 7]),
          5: set([6, 8]),
          6: set([5, 7]),
          7: set([8]),
          8: set([7]),
          9: set([10]),
          10: set([9]),
          11: set([])}

def bfs_visited(ugraph,start_node):
	''' Takes the undirected graph ugraph and 
	the node start_node and returns the set consisting 
	of all nodes that are visited by a breadth-first search 
	that starts at start_node'''
	_queue = deque()
	_visited = set([start_node])
	_queue.append(start_node)
	while len(_queue) > 0:
		_current_node = _queue.popleft()
#		print "Node", _current_node, "has the", len(ugraph[_current_node]), "following neighbors"
		for _neighbor in ugraph[_current_node]:
			if _neighbor not in _visited:
				_visited.add(_neighbor)
				_queue.append(_neighbor)
	return _visited

def cc_visited(ugraph):
	'''Takes the undirected graph ugraph and returns 
	a list of sets, where each set consists of all the 
	nodes (and nothing else) in a connected component, 
	and there is exactly one set in the list for each 
	connected component in ugraph and nothing else.'''
	_remaining_nodes = set(ugraph.keys())
	_connected_components = []
	while len(_remaining_nodes) > 0:
		_random_node = random.sample(_remaining_nodes,1)
#		print "RANDOM START NODE", _random_node
		_random_node_connections = bfs_visited(ugraph,_random_node[0])
		_connected_components.append(_random_node_connections)
		_remaining_nodes -= _random_node_connections
#		print _remaining_nodes
	return _connected_components

#print cc_visited(GRAPH5)

def largest_cc_size(ugraph):
	''' Takes the undirected graph ugraph and returns 
	the size (an integer) of the largest connected component 
	in ugraph.'''
	_graph_components = cc_visited(ugraph)
	_largest_component = 0
	for _current_component in _graph_components:
		if len(_current_component) > _largest_component:
			_largest_component = len(_current_component)
#	print "Largest component is", _largest_component
	return _largest_component

def compute_resilience(ugraph, attack_order):
	'''Takes the undirected graph ugraph, a list of nodes attack_order 
	and iterates through the nodes in attack_order. For each node in the list, 
	the function removes the given node and its edges from the graph and then 
	computes the size of the largest connected component for the resulting graph.'''
	_original_largest_component = largest_cc_size(ugraph)
	_resiliency_list = [_original_largest_component]
	_attack_queue  = deque(attack_order)
	while len(_attack_queue) > 0:
#		print "Counter", counter
		_node_to_remove = _attack_queue.popleft()
#		print "Node to remove", _node_to_remove
		_edges_to_remove = ugraph.pop(_node_to_remove)
#		print "Edges to remove", _edges_to_remove
		for _edge in _edges_to_remove:
			if _node_to_remove in ugraph[_edge]:
#				print "Edges before", ugraph[_edge]
				ugraph[_edge] -= set([_node_to_remove])
#				print "Edges after", ugraph[_edge]
#		print "Updated Graph:", ugraph
		_new_largest_component = largest_cc_size(ugraph)
		_resiliency_list.append(_new_largest_component)
	return _resiliency_list

#print compute_resilience(GRAPH5, [1,3,10])

##########################################################
# Code for creating UNDIRECTED ER graphs (from Homework 1, Pseudo-Code on Q10)
import random

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
# code to create random_order -> 

def random_order(ugraph):
	'''Takes a graph and returns a list of the nodes in the graph in some random order.'''
	V = ugraph.keys()
	n = len(V)
	random_node_order = random.sample(V,n)
	return random_node_order 


##########################################################
# Code to determine optimal p for the undirected ER graph
def optimal_p(trials, p):
	'''Takes a number of trials, and a list of p values in order to
	find the optimal value p such that the number of edges in the ER graph
	is approximately the same as the Computer Network and UPA graph'''
	trial_results = {}
	total_edges_per_trial = 0
	for percent in p:
		trial_results[percent] = set()
		for trial in range(trials):
			test = undirected_ER(1239, percent)
			for node in test.keys():
				total_edges_per_trial += len(test[node])
			single_count_edges = total_edges_per_trial/2.
			trial_results[percent].add(single_count_edges)
			total_edges_per_trial = 0
	return trial_results


def calculate_out_degrees(ugraph):
	total_edges_per_trial = 0
	for node in ugraph.keys():
		total_edges_per_trial += len(ugraph[node])
	ans = total_edges_per_trial/2
	return ans

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


##########################################################
# Compute Resilience of ER, UPA and Computer Network when removing one node at a time
def calculate_network_resilience(ugraph):
	ugraph_copy = copy_graph(ugraph)
	attack_nodes = random_order(ugraph_copy)
	cc_sizes = compute_resilience(ugraph_copy,attack_nodes)
	return cc_sizes



#lst = [0.002,0.0025,0.003]
#testing = optimal_p(50,lst)
#for each in lst:
#	print each, ":", sum(testing[each])/len(testing[each])
# RESULTS
#0.002 : 3063.57142857
#0.0025 : 3839.0
#0.003 : 4594.82608696