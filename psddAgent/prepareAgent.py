from psddAgent.utils.create_sdds_4_reco_rl import create_sdds
import subprocess
import os

def translate_constraints(node):
	constraints = []
	if 'children' not in node.keys():
	    zone_ids = set([node['zone_id']])
	elif len(node['children']) == 0:
	    zone_ids = set([node['zone_id']])
	else:
		zone_ids = set([])
		for child in node['children']:
			new_zone_ids, new_constraints = translate_constraints(child)
			zone_ids = zone_ids.union(new_zone_ids)
			constraints += new_constraints
	if node['min'] is not None:
		if node['min'] > 0:
			constraints.append([list(zone_ids), 'geq', int(node['min'])])
	if node['max'] is not None:
	    constraints.append([list(zone_ids), 'leq', int(node['max'])])
	if node['equals'] is not None:
	    constraints.append([list(zone_ids), 'eq', int(node['equals'])])
	return zone_ids, constraints

def create_psdd_files(path, env_name, constraints, nzones, nresources):
	_, constraints = translate_constraints(constraints)
	files = create_sdds(nzones, nresources, constraints, path, env_name)
	cur_dir = subprocess.check_output(['pwd']).decode("utf-8").strip() + '/'
	print(cur_dir)
	loc_conjoin_sdds2psdd = cur_dir + '../psddAgent/utils/psdd/mult_sdd2psdd'
	psdd_dir = path + 'psdds/'
	if not os.path.isdir(psdd_dir):
		os.mkdir(psdd_dir)
	loc_final_psdd = psdd_dir + env_name + '.psdd'
	command = [loc_conjoin_sdds2psdd, files[0]]
	for file in files[1]:
		command.append(file)
	command.append(loc_final_psdd)
	print(command)
	subprocess.run(command)
	print("CREATION OF PSDD COMPLETE")
	signal_complete = loc_final_psdd[:-5] + '.done'
	with open(signal_complete, 'w') as fp:
		pass
	print('signal given!')