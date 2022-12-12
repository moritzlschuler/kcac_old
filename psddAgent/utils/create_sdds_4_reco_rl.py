"""
create_psdd(nzones, nresources, constraints)
-> global sum constraint implied
-> constraints in the form [[list_zones1, constraint_type1, rhs1], [list_zones2, constraint_type2, rhs2], ...]
e.g. [[[0], 'geq, 5], [[1], 'leq', 6], ..., [[4, 5], 'eq', 10]]

How can this map to the action space?
Need to be in order of the action space!
"""
from pysdd.sdd import Vtree, SddManager
from psddAgent.utils.create_sdds_4_cardinality_constraints import create_sdd_from_constraint, create_sub_sdd, adjust_sdd
import numpy as np
import pandas as pd
import os
import itertools
import csv
import pickle

def extract_psdd_variables(nzones, nresources, constraints):
	
	# default number of vars: number of resources
	zones2vars = {}
	for i in range(nzones):
		zones2vars[i] = {'no':nresources}

	# reduce number of vars using '<=' and '=' constraints:
	for constraint in constraints:
		if (constraint[1] == 'leq') | (constraint[1] == 'eq'):
			for var in constraint[0]:
				zones2vars[var]['no'] = min(zones2vars[var]['no'], constraint[2])

	# add var numbers:
	counter = 1
	for i in range(nzones):
		zones2vars[i]['vars'] = range(counter, zones2vars[i]['no'] + counter)
		counter += zones2vars[i]['no']

	return zones2vars, counter - 1


def construct_global_sum_constraint_sdd(tot_vars, nresources, path, name):
	# make int variables into str:
	constraint = [range(1 ,tot_vars + 1), 'eq', nresources]

	# obtain SDD with already designed function
	subsdd_dir = path + 'sub_sdds/'
	if not os.path.isdir(subsdd_dir):
		os.mkdir(subsdd_dir)

	path = subsdd_dir + name + '/'
	if not os.path.isdir(path):
		os.mkdir(path)
	filename = create_sdd_from_constraint(constraint, path)
	
	return filename


def calculate_assymsdd_size(no_vars):
	if no_vars == 2:
		size = 5
	else:
		literals = no_vars * 2 - 1
		sinks = 2
		decision_nodes = 2 + 2 * (no_vars - 3) + 1
		size = literals + sinks + decision_nodes
	return size


def create_literals_assymsdd(no_vars):
	literals = ''
	for i in range(no_vars):
		if i != no_vars - 1:
			literals = literals + 'L ' + str(i) + ' ' + str(i * 2) + ' ' + str(i + 1) + '\n'
		literals = literals + 'L ' + str(i + no_vars - 1) + ' ' + str(i * 2) + ' ' + str((i + 1) * -1) + '\n'
	return literals


def create_sinks_assymsdd(no_vars):
	sinks = 'T ' + str(no_vars * 2 - 1) + '\n'
	if no_vars > 2:
		sinks = sinks + 'F ' + str(no_vars * 2) + '\n'
	return sinks


def np2pd4assymsdd(decision_nodes):
	if decision_nodes.flags['F_CONTIGUOUS']:
		decision_nodes = np.ascontiguousarray(decision_nodes)
	dtype = decision_nodes.dtype
	dtype = [('ID', dtype), ('vtree', dtype), ('no_kids', dtype), ('tprime', dtype), ('tsub', dtype), ('fprime', dtype), ('fsub', dtype)]
	decision_nodes.dtype = dtype
	decision_nodes[::-1].sort(0, order = 'vtree')
	decision_nodes = decision_nodes[['ID', 'vtree', 'no_kids', 'tprime', 'tsub', 'fprime', 'fsub']]
	decision_nodes = pd.DataFrame(decision_nodes.flatten())
	decision_nodes['prefix'] = 'D'
	return decision_nodes


def construct_intermediate_decision_nodes(no_vars, count, vtree_node):
	decision_nodes = np.zeros([7, no_vars - 3, 2], dtype = int)
	grid = np.mgrid[0:no_vars - 3,0:2]

	#set id:
	decision_nodes[0] = count + grid[0] * 2 + grid[1]
	#set vtree:
	decision_nodes[1] = vtree_node - grid[0] * 2
	#set number of elements:
	decision_nodes[2] = 2
	#set true prime:
	decision_nodes[3] = no_vars - 3 - grid[0]
	#set true sub:
	decision_nodes[4] = 2 * no_vars + 2 * (grid[0] + 1) - 1
	decision_nodes[4][:, 1] = no_vars * 2
	#set false prime:
	decision_nodes[5] = 2 * no_vars - 4 - grid[0]
	#set false sub:
	decision_nodes[6] = 2 * no_vars + 2 * (grid[0] + 1)

	# reshaping the array so that  info for each decision node is represented by 1 row
	decision_nodes = np.reshape(decision_nodes.T, (2 * (no_vars - 3), 7), order = 'C')
	# configuring the numpy array into a sorted pandas dataframe:
	decision_nodes = np2pd4assymsdd(decision_nodes)

	return decision_nodes


def create_decision_nodes_assymsdd(no_vars):
	if no_vars == 2:
		leaf_layer = 'D 4 1 2 0 3 1 2\n'
		intermediate_layers = 'no_vars is 2'
		root_layer = 'no_vars is 2'


	else:
		# add final decision nodes:
		count = no_vars * 2 + 1
		vtree_node = no_vars * 2 - 3
		leaf_layer = 'D ' + str(count) + ' ' + str(vtree_node) + ' ' + str(2) + \
		' ' + str(no_vars - 2) + ' ' + str(no_vars * 2 - 1) + ' ' + \
		str(2 * no_vars - 3) + ' ' + str(2 * no_vars - 2) + '\n' + \
		'D ' + str(count + 1) + ' ' + str(vtree_node) + ' ' + str(2) + ' ' + \
		str(no_vars - 2) + ' ' + str(no_vars * 2) + ' ' + str(2 * no_vars - 3) + \
		' ' + str(2 * no_vars - 2) + '\n'
		count += 2
		vtree_node -= 2

		# add intermediate layers:
		if no_vars > 3:
			intermediate_layers = construct_intermediate_decision_nodes(no_vars, count, vtree_node)
			count = count + 2 * (no_vars - 3)
		else:
			intermediate_layers = 'no_vars is 3'

		# add root layer:
		root_layer = 'D ' + str(count) + ' 1 2 0 ' + \
		str(count - 2) + ' ' + str(no_vars - 1) + ' ' + str(count - 1)

	return leaf_layer, intermediate_layers, root_layer



def symmetry_breaking_sdd(no_vars, path):

	# check if exists
	sddfile = path + 'symmetry_breaker' + '-' + str(no_vars) + '.sdd'
	files = os.listdir(path)

	if sddfile[len(path):] in files:
		return sddfile

	else:
		#create header
		header = 'c ids of sdd nodes start at 0\nc sdd nodes appear bottom-up, children before parents\nc\nc file syntax:\nc sdd count-of-sdd-nodes\nc F id-of-false-sdd-node\nc T id-of-true-sdd-node\nc L id-of-literal-sdd-node id-of-vtree literal\nc D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*\nc\n'

		#create sdd-size row:
		sdd_size = calculate_assymsdd_size(no_vars)
		sdd_size_str = 'sdd ' + str(sdd_size) + '\n'

		#create rows for literals:
		literals = create_literals_assymsdd(no_vars)

		#create rows for sinks:
		sinks = create_sinks_assymsdd(no_vars)

		#create rows for decision nodes:
		leaf_layer, intermediate_layers, root_layer = create_decision_nodes_assymsdd(no_vars)

		#write everyting to file and returning file name:
		file = open(sddfile, 'w')
		file.write(header + sdd_size_str + literals + sinks + leaf_layer)
		file.close()
		if isinstance(intermediate_layers, str) == False:
			col_order = ['prefix', 'ID', 'vtree', 'no_kids', 'tprime', 'tsub', 'fprime', 'fsub']
			intermediate_layers[col_order].to_csv(sddfile, mode = 'a', sep = ' ', header = False, index = False, quoting=csv.QUOTE_NONE)
		if root_layer != 'no_vars is 2':
			file = open(sddfile, 'a')
			file.write(root_layer)
			file.close()

		return sddfile


def construct_symmetry_breaking_sdd(zone, zones2vars, tot_vars, path, name):
	#construct symmetry breaker
	no_vars = zones2vars[zone]['no']
	if no_vars == 1:
		return "trivial case"
	else:
		subsdd_dir = path + 'sub_sdds/' + name + '/'
		filename = symmetry_breaking_sdd(no_vars, subsdd_dir)
	
		#adjust the sdd
		adjusted_sdd_file = adjust_sdd(range(1, tot_vars + 1), zones2vars[zone]['vars'], filename, [zone])

		return adjusted_sdd_file


def translate_and_check_constraint(constraint, zones2vars):
	variables = list(itertools.chain(*[zones2vars[i]['vars'] for i in constraint[0]]))
	constraint[0] = variables
	if (constraint[1] == 'leq') & (len(constraint[0]) <= constraint[2]):
		constraint = 'trivial case'
	return constraint


def create_sdds(nzones, nresources, constraints, path, name):

	#obtain zone to variable mapping and total number of variables, save the former:
	zones2vars, tot_vars = extract_psdd_variables(nzones, nresources, constraints)
	if not os.path.isdir(path):
		os.mkdir(path)
	zones2vars_dir = path + 'zones2vars/'
	if not os.path.isdir(zones2vars_dir):
		os.mkdir(zones2vars_dir)
	pklfile = open(zones2vars_dir + 'zones2vars-' + name + '.pkl', 'ab')
	pickle.dump(zones2vars, pklfile)
	pklfile.close()

	#construct right-linear vtree:
	variables = list(range(1, tot_vars + 1, 1))
	vtree = Vtree(var_count = tot_vars, var_order = variables, vtree_type = 'right')
	vtree_dir = path + 'vtrees/'
	if not os.path.isdir(vtree_dir):
		os.mkdir(vtree_dir)
	vtree_filename = vtree_dir + name + '.vtree'
	vtree.save(bytes(vtree_filename, 'utf-8'))
	print('vtree constructed')

	#construct global sum constraint sdd:
	global_sum_constraint = construct_global_sum_constraint_sdd(tot_vars, nresources, path, name)
	print('global sum constraint constructed')

	#construct constraint sdds:
	sdd_filenames = [global_sum_constraint]
	for constraint in constraints:
		zones = constraint[0]
		constraint = translate_and_check_constraint(constraint, zones2vars)
		if isinstance(constraint, str) == False:
			sdd_filename = create_sub_sdd(constraint, path, name, variables, zones)
			sdd_filenames.append(sdd_filename)

	#construct symmetry breaking constraints
	for zone in range(nzones):
		assymsdd_filename = construct_symmetry_breaking_sdd(zone, zones2vars, tot_vars, path, name)
		if assymsdd_filename != 'trivial case':
			sdd_filenames.append(assymsdd_filename)
	print('constraints constructed and assymetry enforced')

	# collect filenames
	files = [vtree_filename, sdd_filenames]

	return files