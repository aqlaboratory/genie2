import gzip
import torch
import numpy as np

from genie.constants.residue import (
	RESTYPES,
	RESTYPE_ORDER,
	RESTYPE_1_TO_3,
	RESTYPE_3_TO_1
)
from genie.utils.motif_utils import (
	load_motif_spec,
	sample_motif_mask
)


def create_empty_np_features(lengths):
	"""
	Create a feature dictionary based on chain lengths.

	Args:
		lengths:
			A list of chain lengths (in a single structure), where each 
			length denotes the sequence length for each chain.
	"""

	# Define
	num_chains = np.array(len(lengths))
	num_residues = np.sum(lengths)
	num_residues_per_chain = np.array(lengths)

	# Generate
	aatype = np.zeros((num_residues, len(RESTYPES)))
	atom_positions = np.zeros((num_residues, 3))
	residue_mask = np.ones(num_residues)
	residue_index = np.concatenate([
		np.arange(length)
		for length in lengths
	])
	chain_index = np.concatenate([
		[idx] * length
		for idx, length in enumerate(lengths)
	])
	fixed_sequence_mask = np.zeros(num_residues)
	fixed_structure_mask = np.zeros((num_residues, num_residues))
	fixed_group = np.zeros(num_residues)
	interface_mask = np.zeros(num_residues)

	# Create
	np_features = {
		'aatype': aatype.astype(int),
		'num_chains': num_chains.astype(int),
		'num_residues': num_residues.astype(int),
		'num_residues_per_chain': num_residues_per_chain.astype(int),
		'atom_positions': atom_positions.astype(float),
		'residue_mask': residue_mask.astype(int),
		'residue_index': residue_index.astype(int),
		'chain_index': chain_index.astype(int),
		'fixed_sequence_mask': fixed_sequence_mask.astype(bool),
		'fixed_structure_mask': fixed_structure_mask.astype(bool),
		'fixed_group': fixed_group.astype(int),
		'interface_mask': interface_mask.astype(bool)
	}

	return np_features

def create_np_features_from_pdb(filepath):
	"""
	Create a feature dictionary from a PDB file.

	Args:
		filepath:
			PDB filepath.
	"""

	# Load
	seqs, coords = parse_pdb(filepath)
	lengths = [len(seq) for seq in seqs]

	# Generate
	np_features = create_empty_np_features(lengths)

	# Generate
	aatype = np.concatenate(seqs)
	aatype = np.eye(len(RESTYPES))[aatype]
	atom_positions = np.concatenate(coords)
	atom_positions = atom_positions - np.mean(atom_positions, axis=0, keepdims=True)

	# Update
	np_features['aatype'] = aatype.astype(int)
	np_features['atom_positions'] = atom_positions.astype(float)

	return np_features

def create_np_features_from_motif_pdb(filepath):
	"""
	Create a feature dictionary from a motif specification file. This involves
	loading the motif specification file and sampling a motif configuration 
	that satisfies the specification.

	Args:
		filepath:
			Path to a motif specification file.
	"""

	# Parse
	spec = load_motif_spec(filepath)
	motif_seqs, motif_coords = parse_pdb(filepath)
	motif_aatype = np.concatenate(motif_seqs)
	motif_aatype = np.eye(len(RESTYPES))[motif_aatype] # one-hot encoding
	motif_atom_positions = np.concatenate(motif_coords)

	# Generate motif mask
	motif_mask = sample_motif_mask(spec)
	fixed_sequence_mask = motif_mask['sequence']
	fixed_structure_mask = motif_mask['structure']
	fixed_group = motif_mask['group']

	# Initialize features
	num_residues = len(fixed_sequence_mask)
	features = create_empty_np_features([num_residues])

	# Update features
	features['aatype'][fixed_sequence_mask] = motif_aatype
	features['atom_positions'][fixed_sequence_mask] = motif_atom_positions
	features['fixed_sequence_mask'] = fixed_sequence_mask
	features['fixed_structure_mask'] = fixed_structure_mask
	features['fixed_group'] = fixed_group

	return features

###############
###   I/O   ###
###############

def save_np_features_to_pdb(np_features, filepath):
	"""
	Save a feature dictionary (padding removed) into a PDB file.

	Args:
		np_features:
			A feature dictionary, where values are numpy arrays.
		filepath:
			Output PDB filepath.
	"""

	def replace(string, index, substring):
		length = len(substring)
		return string[:index] + substring + string[index+length:]

	# Center and round atom positions
	coords = np_features['atom_positions']
	coords = coords - np.mean(coords, axis=0, keepdims=True)
	coords = np.around(coords, decimals=3)

	# Open
	with open(filepath, 'w') as file:

		# Iterate through all residues
		for i in range(coords.shape[0]):

			# Define residue information
			atom_index = i + 1
			residue_name = RESTYPE_1_TO_3[RESTYPES[np.argmax(np_features['aatype'][i])]]
			residue_index = np_features['residue_index'][i] + 1
			chain_name = chr(ord('A') + np_features['chain_index'][i])
			x, y, z = coords[i][0], coords[i][1], coords[i][2]
			group = ' ' if np_features['fixed_group'][i] == 0 else \
				chr(np_features['fixed_group'][i] - 1 + ord('A'))

			# Create line
			line = ' ' * 80
			line = replace(line, 0, 'ATOM')
			line = replace(line, 6, str(atom_index).rjust(5))
			line = replace(line, 13, 'CA')
			line = replace(line, 17, residue_name)
			line = replace(line, 21, chain_name)
			line = replace(line, 22, str(residue_index).rjust(4))
			line = replace(line, 30, str(x).rjust(8))
			line = replace(line, 38, str(y).rjust(8))
			line = replace(line, 46, str(z).rjust(8))
			line = replace(line, 72, group.ljust(4))
			line = replace(line, 77, 'C')

			# Save line
			file.write(line + '\n')

##################
###   Others   ###
##################

def pad_np_features(np_features, max_n_chain, max_n_res):
	"""
	Pad values in a feature dictionary based on maximum number of chains and 
	maximum number of residues.

	Args:
		np_features:
			A feature dictionary, where values are numpy arrays.
		max_n_chain:
			Maximum number of chains.
		max_n_res:
			Maximum number of residues.
	"""
	num_chains = np_features['num_chains']
	num_residues = np_features['num_residues']
	for key in np_features:
		if key == 'num_residues_per_chain':
			np_features[key] = np.concatenate([
				np_features[key],
				np.zeros(max_n_chain - num_chains).astype(np_features[key].dtype)
			])
		elif key == 'fixed_structure_mask':
			np_features[key] = np.pad(
				np_features[key],
				[
					(0, max_n_res - num_residues),
					(0, max_n_res - num_residues)
				],
				'constant',
				constant_values=0
			).astype(np_features[key].dtype)
		elif not key.startswith('num'):
			np_features[key] = np.concatenate([
				np_features[key],
				np.zeros((
					max_n_res - num_residues,
					*np_features[key].shape[1:]
				)).astype(np_features[key].dtype)
			])
	return np_features

def batchify_np_features(list_np_features):
	"""
	Compress a list of feature dictionaries into a batch.

	Args:
		list_np_features:
			A list of feature dictionary, where values in each 
			dictionary are numpy arrays.
	"""

	# Define
	keys = list(list_np_features[0].keys())

	# Pad
	max_n_chain = np.max([
		np_features['num_chains']
		for np_features in list_np_features
	])
	max_n_res = np.max([
		np_features['num_residues']
		for np_features in list_np_features
	])
	list_np_features_padded = [
		pad_np_features(np_features, max_n_chain, max_n_res)
		for np_features in list_np_features
	]

	# Batchify
	np_features = {}
	for key in keys:
		np_features[key] = np.concatenate([
			np.expand_dims(np_features_padded[key], axis=0)
			for np_features_padded in list_np_features_padded
		])

	return np_features

def debatchify_np_features(np_features):
	"""
	Decompress a batch into a list of feature dictionaries.

	Args:
		np_features:
			A batched feature dictionary, where values are numpy arrays.
	"""

	# Define
	num_samples = np_features['aatype'].shape[0]
	list_np_features = []

	# Iterate
	for i in range(num_samples):
		num_chains = np_features['num_chains'][i]
		num_residues = np_features['num_residues'][i]
		list_np_features.append({
			'num_chains': np_features['num_chains'][i],
			'num_residues': np_features['num_residues'][i],
			'num_residues_per_chain': np_features['num_residues_per_chain'][i, :num_chains],
			'aatype': np_features['aatype'][i, :num_residues],
			'atom_positions': np_features['atom_positions'][i, :num_residues],
			'residue_mask': np_features['residue_mask'][i, :num_residues],
			'residue_index': np_features['residue_index'][i, :num_residues],
			'chain_index': np_features['chain_index'][i, :num_residues],
			'fixed_sequence_mask': np_features['fixed_sequence_mask'][i, :num_residues],
			'fixed_structure_mask': np_features['fixed_structure_mask'][i, :num_residues, :num_residues],
			'fixed_group': np_features['fixed_group'][i, :num_residues],
			'interface_mask': np_features['interface_mask'][i, :num_residues]
		})

	return list_np_features

def convert_np_features_to_tensor(features, device):
	"""
	Convert values in a (batched) feature dictionary to tensors.
	"""
	return {
		'num_chains': torch.Tensor(features['num_chains']).int().to(device),
		'num_residues': torch.Tensor(features['num_residues']).int().to(device),
		'num_residues_per_chain': torch.Tensor(features['num_residues_per_chain']).int().to(device),
		'aatype': torch.Tensor(features['aatype']).int().to(device),
		'atom_positions': torch.Tensor(features['atom_positions']).float().to(device),
		'residue_mask': torch.Tensor(features['residue_mask']).int().to(device),
		'residue_index': torch.Tensor(features['residue_index']).int().to(device),
		'chain_index': torch.Tensor(features['chain_index']).int().to(device),
		'fixed_sequence_mask': torch.Tensor(features['fixed_sequence_mask']).bool().to(device),
		'fixed_structure_mask': torch.Tensor(features['fixed_structure_mask']).bool().to(device),
		'fixed_group': torch.Tensor(features['fixed_group']).int().to(device),
		'interface_mask': torch.Tensor(features['interface_mask']).bool().to(device)
	}

def convert_tensor_features_to_numpy(features):
	"""
	Convert values in a (batched) feature dictionary to numpy arrays.
	"""
	return {
		'num_chains': features['num_chains'].detach().cpu().numpy().astype(int),
		'num_residues': features['num_residues'].detach().cpu().numpy().astype(int),
		'num_residues_per_chain': features['num_residues_per_chain'].detach().cpu().numpy().astype(int),
		'aatype': features['aatype'].detach().cpu().numpy().astype(int),
		'atom_positions': features['atom_positions'].detach().cpu().numpy().astype(float),
		'residue_mask': features['residue_mask'].detach().cpu().numpy().astype(int),
		'residue_index': features['residue_index'].detach().cpu().numpy().astype(int),
		'chain_index': features['chain_index'].detach().cpu().numpy().astype(int),
		'fixed_sequence_mask': features['fixed_sequence_mask'].detach().cpu().numpy().astype(bool),
		'fixed_structure_mask': features['fixed_structure_mask'].detach().cpu().numpy().astype(bool),
		'fixed_group': features['fixed_group'].detach().cpu().numpy().astype(int),
		'interface_mask': features['interface_mask'].detach().cpu().numpy().astype(bool)
	}

def prepare_tensor_features(features):
	"""
	Cast tensors in a feature dictionary to the correct type.
	"""
	return {
		'num_chains': features['num_chains'].int(),
		'num_residues': features['num_residues'].int(),
		'num_residues_per_chain': features['num_residues_per_chain'].int(),
		'aatype': features['aatype'].int(),
		'atom_positions': features['atom_positions'].float(),
		'residue_mask': features['residue_mask'].int(),
		'residue_index': features['residue_index'].int(),
		'chain_index': features['chain_index'].int(),
		'fixed_sequence_mask': features['fixed_sequence_mask'].bool(),
		'fixed_structure_mask': features['fixed_structure_mask'].bool(),
		'fixed_group': features['fixed_group'].int(),
		'interface_mask': features['interface_mask'].bool()
	}

###################
###   Helpers   ###
###################

def summarize_pdb(filepath):
	"""
	Get summary statistics for a PDB file.
	"""
	seqs, coords = parse_pdb(filepath)
	num_chains = len(seqs)
	num_residues = np.sum([len(seq) for seq in seqs])
	return {
		'num_residues': num_residues,
		'num_chains': num_chains
	}

def parse_pdb(filepath):
	"""
	Parse a PDB file to extract sequences and Ca coordinates.
	"""

	def _handle(file):
		seqs, coords = [], []
		current_chain = None
		for line in file:
			if line.startswith('ATOM') and line[13:15].strip() == 'CA':
				
				# Parse
				restype_3 = line[17:20]
				restype_1 = RESTYPE_3_TO_1[restype_3]
				restype_order = RESTYPE_ORDER[restype_1]
				chain = line[21]
				x = float(line[30:38])
				y = float(line[38:46])
				z = float(line[46:54])

				# Create data structure
				if current_chain is None or chain != current_chain:
					seqs.append([])
					coords.append([])
					current_chain = chain

				# Update
				seqs[-1].append(restype_order)
				coords[-1].append([x, y, z])

		return seqs, coords

	if filepath.endswith('.gz'):
		with gzip.open(filepath, 'rt') as file:
			seqs, coords = _handle(file)
	else:
		with open(filepath, 'r') as file:
			seqs, coords = _handle(file)

	return seqs, coords