import numpy as np


def load_motif_spec(filepath):
	"""
	Load motif specification file.

	Args:
		filepath:
			Path to the PDB file for motif specification.

	Returns:
		A dictionary of motif specifications containing
			-	name:
				Name of the motif scaffolding problem
			-	structures:
				A list of dictionaries, each of which defines either 
					-	a motif segment, containing information on the chain and 
						residue index range that the motif structure is coming 
						from, as well as the motif group that this segment belongs
					-	a scaffold segment, containing information on the maximum 
						and minimum number of residues for the segment
			-	min_total_length:
				Minimum number of residues for the generated structure
			-	max_total_length:
				Maximum number of residues for the generated structure.
	"""
	with open(filepath) as file:
		structures = []
		for line in file:
			if line.startswith('REMARK 999 INPUT'):
				if line[18] == ' ':
					structures.append({
						'type': 'scaffold',
						'min_length': int(line[19:23]),
						'max_length': int(line[23:27])
					})
				else:
					structures.append({
						'type': 'motif',
						'chain': line[18],
						'start_index': int(line[19:23]),
						'end_index': int(line[23:27]),
						'group': line[28] if len(line) > 28 and line[28] != ' ' else 'A'
					})
			if line.startswith('REMARK 999 NAME'):
				name = line[18:]
			if line.startswith('REMARK 999 MINIMUM TOTAL LENGTH'):
				min_total_length = int(line[37:])
			if line.startswith('REMARK 999 MAXIMUM TOTAL LENGTH'):
				max_total_length = int(line[37:])
	return {
		'name': name,
		'structures': structures,
		'min_total_length': min_total_length,
		'max_total_length': max_total_length
	}

def sample_motif_mask(spec):
	"""
	Sample a motif configuration from a dictionary of specifications.

	Args:
		spec:
			A dictionary of motif specifications containing
				-	name:
					Name of the motif scaffolding problem
				-	structures:
					A list of dictionaries, each of which defines either 
						-	a motif segment, containing information on the chain and 
							residue index range that the motif structure is coming 
							from, as well as the motif group that this segment belongs
						-	a scaffold segment, containing information on the maximum 
							and minimum number of residues for the segment
				-	min_total_length:
					Minimum number of residues for the generated structure
				-	max_total_length:
					Maximum number of residues for the generated structure.

	Returns:
		A dictionary of masks including
			-	sequence:
				A residue-level mask to indicate which residue contains conditional 
				sequence information
			-	structure: 
				A pair residue-residue mask to indicate which pair of residues contains
				conditional structural information
			-	group:
				Residue-level group indices to indicate which group each residue belongs to 
				(0 indicates scaffold and each positive integer indicates a motif group).
	"""
	success = False
	while not success:

		# Define
		total_length = 0
		motif_sequence_mask = []
		motif_groups = []

		# Generate
		for structure in spec['structures']:
			if structure['type'] == 'scaffold':
				scaffold_length = np.random.randint(structure['min_length'], structure['max_length'] + 1)
				motif_sequence_mask.extend([0] * scaffold_length)
				motif_groups.extend([0] * scaffold_length)
				total_length += scaffold_length
			else:
				motif_length = structure['end_index'] - structure['start_index'] + 1
				motif_sequence_mask.extend([1] * motif_length)
				motif_groups.extend([ord(structure['group']) - ord('A') + 1] * motif_length)
				total_length += motif_length

		# Validate
		if total_length >= spec['min_total_length'] and \
			total_length <= spec['max_total_length']:
			success = True

	# Create motif structure mask
	motif_structure_mask = np.zeros((total_length, total_length))
	num_groups = np.max(motif_groups)
	for i in range(1, 1 + num_groups):
		motif_group_sequence_mask = np.equal(motif_groups, i)
		motif_structure_mask += motif_group_sequence_mask[:, np.newaxis] * motif_group_sequence_mask[np.newaxis, :]

	return {
		'sequence': np.array(motif_sequence_mask).astype(bool),
		'structure': np.array(motif_structure_mask).astype(bool),
		'group': np.array(motif_groups).astype(int)
	}

def save_motif_pdb(spec_filepath, mask, pdb_filepath):
	"""
	Save motif information as a PDB file.

	Args:
		spec_filepath:
			Path to motif specification file.
		mask:
			A residue-level mask to indicate which residue is a motif residue
		pdb_filepath:
			Output PDB filepath.
	"""

	def pad_left(string, length):
		assert len(string) <= length
		return ' ' * (length - len(string)) + string

	# Parse residue index in motif spec file
	spec = load_motif_spec(spec_filepath)
	residue_index_spec = []
	for structure in spec['structures']:
		if structure['type'] == 'motif':
			for i in range(structure['start_index'], structure['end_index'] + 1):
				residue_index_spec.append((
					structure['chain'],
					i,
					structure['group']
				))

	# Parse residue index in motif pdb file
	residue_index_pdb = [i + 1 for i, elt in enumerate(mask) if elt]
	assert len(residue_index_pdb) == len(residue_index_spec)

	# Create residue index map
	residue_index_map = dict([
		(
			'{}_{}'.format(elt[0], elt[1]),
			(residue_index_pdb[i], elt[2])
		)
		for i, elt in enumerate(residue_index_spec)
	])

	# Parse records in motif spec file
	with open(spec_filepath) as file:
		lines = [line for line in file if line.startswith('ATOM')]

	# Update residue index
	updated_lines = []
	for i, line in enumerate(lines):
		chain = line[21]
		residue_index = int(line[22:26])
		key = '{}_{}'.format(chain, residue_index)
		updated_residue_index = residue_index_map[key][0]
		updated_group = residue_index_map[key][1]
		updated_line = line[:21] + 'A' + str(updated_residue_index).rjust(4) + line[26:72] + updated_group.ljust(4) + line[76:]
		updated_lines.append(updated_line)

	# Save
	with open(pdb_filepath, 'w') as file:
		file.write(''.join(updated_lines))
