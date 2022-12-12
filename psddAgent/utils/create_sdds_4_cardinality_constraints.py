"""
implement function that takes in the constraints in the following form:
[[[1','2','3','4'],'eq',3], [['1','4'],'geq',1], [['2','3','4'],'leq',2]]
and returns SDD for it

#find the set of all variables
#construct right-linear vtree
#write in individual constraints as SDDs
#conjoin them
#write out final SDD

"""
from pysdd.sdd import Vtree
import numpy as np
import pandas as pd
import csv
import os
from time import time, localtime

def calculate_sdd_size(n, k, constraint_type):
	no_literal_and_sink_nodes = 2 * n + 1
	if constraint_type == 'geq':
		no_decision_nodes = k * (n - k + 1) - 1
		if (k == n) | (k == 1):
			no_literal_and_sink_nodes -= 1
	elif constraint_type == 'leq':
		no_decision_nodes = (k + 1) * (n - k) - 1
		if (k == n - 1) | (k == 0):
			no_literal_and_sink_nodes -= 1
	elif constraint_type == 'eq':
		no_decision_nodes = (k + 1) * (n - k + 1) - 3
	else:
		raise ValueError('Use onstraint type "eq", "geq" or "leq"')
	size = no_literal_and_sink_nodes + no_decision_nodes
	return size

def create_literals(n, constraint_type):
	literals = ''
	if constraint_type == 'geq':
		for i in range(n):
			literals = literals + 'L ' + str(i) + ' ' + str(i * 2) + ' ' + str(i + 1) + '\n'
			if i != n - 1:
				literals = literals + 'L ' + str(i + n) + ' ' + str(i * 2) + ' ' + str((i + 1) * -1) + '\n'
	elif constraint_type == 'leq':
		for i in range(n):
			literals = literals + 'L ' + str(i) + ' ' + str(i * 2) + ' ' + str((i + 1) * -1) + '\n'
			if i != n - 1:
				literals = literals + 'L ' + str(i + n) + ' ' + str(i * 2) + ' ' + str(i + 1) + '\n'
	elif constraint_type == 'eq':
		for i in range(n):
			literals = literals + 'L ' + str(i) + ' ' + str(i * 2) + ' ' + str(i + 1) + '\n'
			literals = literals + 'L ' + str(i + n) + ' ' + str(i * 2) + ' ' + str((i + 1) * -1) + '\n'
	else:
		raise ValueError('Use onstraint type "eq", "geq" or "leq"')
	return literals

def create_sinks(n, k, constraint_type, sdd_size_str):
	if constraint_type in ['leq', 'geq']:
		if n > 1:
			if ((constraint_type == 'leq') & (k == 0)) | \
			((constraint_type == 'geq') & (k == n)):
				sinks = 'F ' + str(2 * n - 1) + '\n'
			elif ((constraint_type == 'leq') & (k == n - 1)) | \
			((constraint_type == 'geq') & (k == 1)):
				sinks = 'T ' + str(2 * n - 1) + '\n'
			else:
				sinks = 'T ' + str(2 * n - 1) + '\nF ' + str(2 * n) + '\n'
		else:
			sinks = 'no_sinks'
			sdd_size_str = 'sdd 1\n'
	elif constraint_type == 'eq':
		if n > 2:
			sinks = 'F ' + str(2 * n) + '\n'
		else: # case where n = 2 and k = 1
			sinks = 'no_sinks'
			sdd_size_str = 'sdd 5\n'
	else:
		raise ValueError('Use onstraint type "eq", "geq" or "leq"')
	return sinks, sdd_size_str

def np2pd(decision_nodes):
	if decision_nodes.flags['F_CONTIGUOUS']:
		decision_nodes = np.ascontiguousarray(decision_nodes)
	dtype = decision_nodes.dtype
	dtype = [('ID', dtype), ('vtree', dtype), ('no_kids', dtype), ('tprime', dtype), ('tsub', dtype), ('fprime', dtype), ('fsub', dtype), ('level', dtype)]
	decision_nodes.dtype = dtype
	decision_nodes[::-1].sort(0, order = 'level')
	decision_nodes = decision_nodes[['ID', 'vtree', 'no_kids', 'tprime', 'tsub', 'fprime', 'fsub']]
	decision_nodes = pd.DataFrame(decision_nodes.flatten())
	decision_nodes['prefix'] = 'D'
	return decision_nodes

def create_decision_nodes(n, k, constraint_type):
	if constraint_type == 'geq':
		# create decision node array and row+column grid:
		decision_nodes = np.zeros([8, k, n - k + 1], dtype = int)
		grid = np.mgrid[0:k,0:n - k + 1]
		#set level
		decision_nodes[7] = grid[0] + grid[1] + 1
		#set IDs
		decision_nodes[0] = (2 * n + 1) + (n - k + 1) * grid[0] + grid[1]
		if (k == n) | (k == 1):
			decision_nodes[0] -= 1
		#set vtree-nodes:
		decision_nodes[1] = decision_nodes[7] * 2 - 1
		#set number of elements:
		decision_nodes[2] = 2
		#set true child, prime:
		decision_nodes[3] = decision_nodes[7] - 1
		#set false child, prime:
		decision_nodes[5] = decision_nodes[7] - 1 + n
		#set true child, sub:
		decision_nodes[4] = decision_nodes[0] + n - k + 1
		decision_nodes[4][k - 1] = 2 * n - 1
		decision_nodes[4][k - 2][n - k] = n - 1
		#set false child, sub:
		decision_nodes[6] = decision_nodes[0] + 1
		decision_nodes[6][:, n - k] = 2 * n
		if k == n:
			decision_nodes[6][:, n - k] -= 1
		decision_nodes[6][k - 1][n - k - 1] = n - 1
		# reshaping the array so that  info for each decision node is represented by 1 row
		decision_nodes = np.reshape(decision_nodes.T, (k * (n - k + 1), 8), order = 'C')
		# configuring the numpy array into a sorted pandas dataframe:
		decision_nodes = np2pd(decision_nodes)
		#removing the bottom OBDD decision node that is subsumed into a literal in SDD
		decision_nodes.drop(decision_nodes.head(1).index,inplace=True)
		
	elif constraint_type == 'leq':
		# create decision node array and row+column grid:
		decision_nodes = np.zeros([8, k + 1, n - k], dtype = int)
		grid = np.mgrid[0:k + 1, 0:n - k]
		#set level
		decision_nodes[7] = grid[0] + grid[1] + 1
		#set IDs
		decision_nodes[0] = (2 * n + 1) + (n - k) * grid[0] + grid[1]
		if (k == 0) | (k == n - 1):
			decision_nodes[0] -= 1
		#set vtree-nodes:
		decision_nodes[1] = decision_nodes[7] * 2 - 1
		#set number of elements:
		decision_nodes[2] = 2
		#set true child, prime:
		decision_nodes[3] = decision_nodes[7] - 1 + n
		#set false child, prime:
		decision_nodes[5] = decision_nodes[7] - 1
		#set true child, sub:
		decision_nodes[4] = decision_nodes[0] + n - k
		decision_nodes[4][k] = 2 * n
		if k == 0:
			decision_nodes[4][k] -= 1
		decision_nodes[4][k - 1][n - k - 1] = n - 1
		#set false child, sub:
		decision_nodes[6] = decision_nodes[0] + 1
		decision_nodes[6][:, n - k - 1] = 2 * n - 1
		decision_nodes[6][k][n - k - 2] = n - 1
		# reshaping the array so that  info for each decision node is represented by 1 row
		decision_nodes = np.reshape(decision_nodes.T, ((k + 1) * (n - k), 8), order = 'C')
		# configuring the numpy array into a pandas dataframe:
		decision_nodes = np2pd(decision_nodes)
		#removing the bottom OBDD decision node that is subsumed into a literal in SDD
		decision_nodes.drop(decision_nodes.head(1).index,inplace=True)

	elif constraint_type == 'eq':
		# create decision node array and row+column grid:
		decision_nodes = np.zeros([8, k + 1, n - k + 1], dtype = int)
		grid = np.mgrid[0:k + 1,0:n - k + 1]
		#set level
		decision_nodes[7] = grid[0] + grid[1] + 1
		#set IDs
		if (n == 2) & (k == 1):
			decision_nodes[0] = (2 * n) + (n - k + 1) * grid[0] + grid[1]
		else:
			decision_nodes[0] = (2 * n + 1) + (n - k + 1) * grid[0] + grid[1]
		#set vtree-nodes:
		decision_nodes[1] = decision_nodes[7] * 2 - 1
		#set number of elements:
		decision_nodes[2] = 2
		#set true child, prime:
		decision_nodes[3] = decision_nodes[7] - 1
		#set false child, prime:
		decision_nodes[5] = decision_nodes[7] - 1 + n
		#set true child, sub:
		decision_nodes[4] = decision_nodes[0] + n - k + 1
		decision_nodes[4][k] = 2 * n
		decision_nodes[4][k - 1][n - k - 1] = 2 * n - 1
		decision_nodes[4][k - 2][n - k] = n - 1
		#set false child, sub:
		decision_nodes[6] = decision_nodes[0] + 1
		decision_nodes[6][:, n - k] = 2 * n
		decision_nodes[6][k - 1][n - k - 1] = n - 1
		decision_nodes[6][k][n - k - 2] = 2 * n - 1
		# rectify decision node ID shift due to decision node being deleted from second last row in SDD representation
		decision_nodes[0][k] = decision_nodes[0][k] - 1
		decision_nodes[4][k - 1, :n - k - 1] = decision_nodes[4][k - 1, :n - k - 1] - 1
		decision_nodes[6][k, :n - k - 2] = decision_nodes[6][k, :n - k - 2] - 1
		# reshaping the array so that  info for each decision node is represented by 1 row
		decision_nodes = np.reshape(decision_nodes.T, ((k + 1) * (n - k + 1), 8), order = 'C')
		# configuring the numpy array into a pandas dataframe:
		decision_nodes = np2pd(decision_nodes)
		#removing the bottom OBDD decision node that is subsumed into a literal in SDD
		decision_nodes.drop(decision_nodes.head(3).index,inplace=True)

	else:
		raise ValueError('Use onstraint type "eq", "geq" or "leq"')

	return decision_nodes

def check_and_correct_constraint(n, k, constraint_type):
	if (constraint_type == 'geq') & (k == 0):
		raise ValueError('k=0 not permissible for "geq" constraints; trivial case')
	if (constraint_type == 'leq') & (n == k):
		raise ValueError('k=n not permissible for "leq" constraints; trivial case')

	#re-routing corner cases:
	if (constraint_type == 'eq') & (k == 0):
		constraint_type = 'leq'
	if (constraint_type == 'eq') & (k == n):
		constraint_type = 'geq'
	return constraint_type

def build_sdd_filename(n, k, constraint_type, path):
	sddfile = path + str(n) + '-' + str(k) + '-' + constraint_type + '.sdd'
	return sddfile

def create_sdd_from_constraint(constraint, path):
	
	# unfold constraint
	n = len(constraint[0])
	k = constraint[2]
	constraint_type = constraint[1]

	# check whether file exists already
	sddfile = build_sdd_filename(n, k, constraint_type, path)
	try:
		files = os.listdir(path)
	except FileNotFoundError:
		os.mkdir(path)
		files = os.listdir(path)
	if sddfile[len(path):] in files:
		return sddfile

	else:
		#reject trivial cases and change constraints for others:
		constraint_type = check_and_correct_constraint(n, k, constraint_type)

		#create header
		header = 'c ids of sdd nodes start at 0\nc sdd nodes appear bottom-up, children before parents\nc\nc file syntax:\nc sdd count-of-sdd-nodes\nc F id-of-false-sdd-node\nc T id-of-true-sdd-node\nc L id-of-literal-sdd-node id-of-vtree literal\nc D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*\nc\n'

		#create sdd-size row:
		sdd_size = calculate_sdd_size(n, k, constraint_type)
		sdd_size_str = 'sdd ' + str(sdd_size) + '\n'

		#create rows for literals:
		literals = create_literals(n, constraint_type)

		#create rows for sinks:
		sinks, sdd_size_str = create_sinks(n, k, constraint_type, sdd_size_str)

		#create rows for decision nodes:
		decision_nodes = create_decision_nodes(n, k, constraint_type)

		# write header, sdd size and literals to .sdd-file
		file = open(sddfile, 'w')
		if sinks == 'no_sinks':
			file.write(header + sdd_size_str + literals)
		else:
			file.write(header + sdd_size_str + literals + sinks)
		file.close()

		# append decision nodes to .sdd-file
		col_order = ['prefix', 'ID', 'vtree', 'no_kids', 'tprime', 'tsub', 'fprime', 'fsub']
		decision_nodes[col_order].to_csv(sddfile, mode = 'a', sep = ' ', header = False, index = False, quoting=csv.QUOTE_NONE)

		return sddfile

def find_variables(constraints):
	variables = set()
	for constraint in constraints:
		variables.update(set(constraint[0]))
	return sorted(list(variables))

def adjust_sdd(tot_vars, sdd_vars, sdd_file, zones):
	"""this function requires the SDD to follow a right-linear vtree and the \
	variable ordering of the vtree to be ascending, e.g. 1, 3, 5 but not 3, 2, 7"""

	# sort sdd_vars
	sdd_vars = sorted(sdd_vars)

	# skip if the sdd_vars are at the beginning of the tot_vars
	if tot_vars[:sdd_vars[-1]] == sdd_vars:
		return sdd_file
	else:

		# read in sdd
		file = open(sdd_file, 'r')
		lines = file.readlines()
		file.close()

		# make new filename
		sdd_file = sdd_file[:-4] + '-' + '-'.join([str(el) for el in zones]) + '.sdd'

		# decompose it
		header = ''
		size = 0
		literals = []
		sinks = []
		decisions = []
		for line in lines:
			if line[0] == 'c':
				header = header + line
			elif line[0] == 's':
				size = int(line[4:])
			elif line[0] == 'L':
				literals.append(line.split())
			elif line[0] == 'D':
				decisions.append(line.split())
			else:
				sinks.append(line.split())

		# discard unused vars
		last_var = sdd_vars[-1]
		last_var_index = tot_vars.index(last_var)
		tot_vars = tot_vars[:last_var_index + 1]

		# map old vtree nodes to new literals for old sdd
		vtree_node = 0
		vtree_node2var = {}
		for var in sdd_vars:
			vtree_node2var[vtree_node] = var
			vtree_node += 2
		#replace the old literals with new ones
		for literal in literals:
			literal[3] = vtree_node2var[int(literal[2])] * \
			int(abs(int(literal[3])) / int(literal[3]))

		# make dictionary mapping variables to their literal descriptions list
		literal_nodes = {}
		for literal_node in literals:
			var = abs(literal_node[3])
			if var not in literal_nodes.keys():
				literal_nodes[var] = [literal_node]
			else:
				literal_nodes[var].append(literal_node)

		# assign vtree nodes to literals:
		id_new_node = size
		vtree_node = 0
		new_literals = {}
		for var in tot_vars:
			# reassign for existing ones
			if var in literal_nodes.keys():
				for literal in literal_nodes[var]:
					literal[2] = vtree_node
			# create entire literal description for new ones
			else:
				literal_nodes[var] = [['L', id_new_node, vtree_node, var], \
				['L', id_new_node + 1, vtree_node, -var]]
				new_literals[var] = id_new_node
				new_literals[-var] = id_new_node + 1
				id_new_node += 2
			vtree_node += 2
		vtree_node -= 2

		# handle special case of no decision nodes:
		if decisions == []:
			tot_vars = tot_vars[:-1]
			new_decision_nodes = []
			new_layer = []
			decision_node_with_literal = ['D', str(id_new_node), str(vtree_node - 1), \
			str(2), str(new_literals[tot_vars[-1]]), str(0), \
			str(new_literals[-tot_vars[-1]]), str(0)]
			id_new_node += 1
			vtree_node -= 2
			new_layer.append(decision_node_with_literal)
			new_decision_nodes.append(new_layer)
			tot_vars = tot_vars[:-1]

			if len(tot_vars) > 0:
				for i in range(tot_vars[-1]):
					new_layer = []
					new_decision_node = ['D', str(id_new_node), str(vtree_node -1), str(2), \
					str(new_literals[tot_vars[-1 - i]]), str(id_new_node - 1), \
					str(new_literals[-tot_vars[-1 - i]]), str(id_new_node - 1)]
					id_new_node += 1
					vtree_node -= 2
					new_layer.append(new_decision_node)
					new_decision_nodes.append(new_layer)		

		else:
			# normal case:
			# sort decision nodes by vtree node:
			decision_nodes = {}
			for decision_node in decisions:
				if decision_node[2] not in decision_nodes.keys():
					decision_nodes[decision_node[2]] = [decision_node]
				else:
					decision_nodes[decision_node[2]].append(decision_node)

			# build new decision node structure
			tot_vars = tot_vars[:-1]
			start = True
			new_decision_nodes = []
			old_vtree_nodes = [str(el) for el in sorted(\
				[int(el) for el in decision_nodes.keys()])]
			i = -1
			replace_key_with_value = {}
			breaker = 0
			while len(tot_vars) > 0:
				breaker += 1
				if tot_vars[-1] not in sdd_vars:
					if start == True:
						succeeding_node = decision_nodes[old_vtree_nodes[i]][0]
						new_decision_node = ['D', str(id_new_node), str(vtree_node - 1), str(2), \
						str(new_literals[tot_vars[-1]]), str(succeeding_node[5]), \
						str(new_literals[-tot_vars[-1]]), str(succeeding_node[5])]
						new_decision_nodes.append([new_decision_node])
						decision_nodes[old_vtree_nodes[i]][0][5] = str(id_new_node)
						id_new_node += 1
						vtree_node -= 2
						start == False
					else:
						preceeding_node = new_decision_nodes[-1][0]
						new_decision_node = ['D', str(id_new_node), str(vtree_node - 1), str(2), \
						str(new_literals[tot_vars[-1]]), str(preceeding_node[1]), \
						str(new_literals[-tot_vars[-1]]), str(preceeding_node[1])]
						replace_key_with_value[str(preceeding_node[1])] = str(id_new_node)
						new_decision_nodes.append([new_decision_node])
						id_new_node += 1
						vtree_node -= 2
				else:
					start = False
					new_layer = []
					for decision_node in decision_nodes[old_vtree_nodes[i]]:
						new_decision_node = decision_node
						new_decision_node[2] = str(vtree_node - 1)
						loop = True
						while loop:
							loop = False
							for el in [5, 7]:
								if new_decision_node[el] in replace_key_with_value.keys():
									new_decision_node[el] = \
									replace_key_with_value[new_decision_node[el]]
									loop = True
						new_layer.append(new_decision_node)
					new_decision_nodes.append(new_layer)
					i -= 1
					vtree_node -= 2
				tot_vars = tot_vars[:-1]

		# update size
		size = id_new_node

		# create strs:
		size_str = 'sdd ' + str(size) + '\n'
		literals_str = ''
		for var in literal_nodes.keys():
			for lit in literal_nodes[var]:
				literals_str = literals_str + ' '.join([str(el) for el in lit]) + '\n'
		sinks_str = ''
		for sink in sinks:
			sinks_str = sinks_str + ' '.join([str(el) for el in sink]) + '\n'
		decision_nodes_str = ''
		for layer in new_decision_nodes:
			for node in layer:
				decision_nodes_str = decision_nodes_str + \
				' '.join([str(el) for el in node]) + '\n'
		decision_nodes_str = decision_nodes_str[:-1]

		# read out the new sdd
		file = open(sdd_file, 'w')
		file.write(header + size_str + literals_str + sinks_str + decision_nodes_str)
		file.close()

		return sdd_file

def create_sub_sdd(constraint, path, name, variables, zones):
	# make int variables into str:
	constraint[0] = [int(var) for var in constraint[0]]
	variables = [int(var) for var in variables]

	# obtain SDD with already designed function
	path = path + 'sub_sdds/' + name + '/'
	filename = create_sdd_from_constraint(constraint, path)
	
	#adjust SDD for the current vtree
	if variables == constraint[0]:
		adjusted_sdd_file = filename
	else:
		adjusted_sdd_file = adjust_sdd(variables, constraint[0], filename, zones)

	return adjusted_sdd_file

def create_sdds(constraints, path, name):

	#find the set of all variables:
	variables = find_variables(constraints)

	#construct right-linear vtree:
	vtree = Vtree(var_count = len(variables), var_order = variables, vtree_type = 'right')
	vtree_filename = path + name + '.vtree'
	vtree.save(bytes(vtree_filename, 'utf-8'))

	print('vtree constructed')

	#write in individual constraints as SDDs:
	sdd_filenames = []
	for constraint in constraints:
		sdd_filename = create_sub_sdd(constraint, path, name, variables)
		sdd_filenames.append(sdd_filename)

	print('sdds constructed')

	# collect filenames
	files = [vtree_filename, sdd_filenames]

	return files