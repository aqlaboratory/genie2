import os
import random
import numpy as np
from enum import Enum
from torch.utils.data import Dataset

from genie.utils.feat_utils import (
	create_np_features_from_pdb,
	pad_np_features
)


class GenieDataset(Dataset):
	"""
	Dataset for Genie 2 training.
	Derived from torch.utils.data.Dataset.
	"""

	def __init__(
		self,
		dataset_info,
		min_n_res,
		max_n_res,
		max_n_chain,
		motif_prob,
		motif_min_pct_res,
		motif_max_pct_res,
		motif_min_n_seg,
		motif_max_n_seg,
	):
		"""
		Initialize dataset.

		Args:
			dataset_info:
				A dictionary on dataset information containing
					-	a key named 'datadir', which stores the data directory
					-	a key named 'names', which store a list of structure names
			min_n_res:
				Minimum number of residues in a structure.
			max_n_res:
				Maximum number of residues in a structure.
			max_n_chain:
				Maximum number of chains in a structure.
			motif_prob:
				Percentage of motif-conditional training tasks.
			motif_min_pct_res:
				Minimum percentage of residues (out of the total sequence length of 
				the input structure) to be defined as motif residues.
			motif_max_pct_res:
				Maximum percentage of residues (out of the total sequence length of 
				the input structure) to be defined as motif residues.
			motif_min_n_seg:
				Minimum number of motif segments.
			motif_max_n_seg:
				Maximum number of motif segments.
		"""
		super(GenieDataset, self).__init__()
		self.min_n_res = min_n_res
		self.max_n_res = max_n_res
		self.max_n_chain = max_n_chain

		# Motif-specific parameters
		self.motif_prob = motif_prob
		self.motif_min_pct_res = motif_min_pct_res
		self.motif_max_pct_res = motif_max_pct_res
		self.motif_min_n_seg = motif_min_n_seg
		self.motif_max_n_seg = motif_max_n_seg

		# Create filepaths
		self.filepaths = self._get_filepaths(dataset_info)
		print('Dataset size: {}'.format(len(self.filepaths)))
  
	def __len__(self):
		"""
		Returns the dataset size.
		"""
		return len(self.filepaths)

	def __getitem__(self, idx):
		"""
		Returns a feature dictionary for a structure with the given index in the
		training dataset. Each value in the feature dictionary is padded accordingly
		(based on the maximum number of residues and the maximum number of chains) to
		ensure the successful construction of a batched feature dictionary.

		Args:
			idx:
				Index of the structure in the training dataset, that is, index in 
				the 'filepaths' parameter.

		Returns:
			np_features:
				A feature dictionary containing information on an input structure 
				(padded to a total sequence length of N), including
					-	aatype: 
							[N, 20] one-hot encoding on amino acid types
					-	num_chains: 
							[1] number of chains in the structure
					-	num_residues: 
							[1] number of residues in the structure
					-	num_residues_per_chain: 
							[1] an array of number of residues by chain
					-	atom_positions: 
							[N, 3] an array of Ca atom positions
					-	residue_mask: 
							[N] residue mask to indicate which residue position is masked
					-	residue_index: 
							[N] residue index (started from 0)
					-	chain_index: 
							[N] chain index (started from 0)
					-	fixed_sequence_mask: 
							[N] mask to indicate which residue contains conditional
							sequence information
					-	fixed_structure_mask: 
							[N, N] mask to indicate which pair of residues contains
							conditional structural information
					-	fixed_group:
							[N] group index to indicate which group the residue belongs to
							(useful for specifying multiple functional motifs)
					-	interface_mask:
							[N] deprecated and set to all zeros.
		"""

		# Load filepath
		filepath = self.filepaths[idx]

		# Load features
		np_features = create_np_features_from_pdb(filepath)

		# Update masks
		if np.random.random() <= self.motif_prob:
			np_features = self._update_motif_masks(np_features)

		# Pad
		np_features = pad_np_features(
			np_features,
			self.max_n_chain,
			self.max_n_res
		)

		return np_features

	############################
	###   Helper Functions   ###
	############################

	def _get_filepaths(self, dataset_info):
		"""
		Load a list of filepaths given dataset information.

		Args:
			dataset_info:
				A dictionary on dataset information containing
					-	a key named 'datadir', which stores the data directory
					-	a key named 'names', which store a list of structure names.

		Returns:
			filepaths:
				A list of filepaths, each of which is the path to the PDB 
				file of a structure.
		"""
		filepaths = [
			os.path.join(dataset_info['datadir'], f'{name}.pdb.gz')
			for name in dataset_info['names']
		]
		random.shuffle(filepaths)
		return filepaths

	def _update_motif_masks(self, np_features):
		"""
		Update fixed sequence and structure mask in the feature dictionary to indicate
		where to provide motif sequence and structure information as conditions. Note 
		that since Genie 2 is trained on single-motif scaffolding tasks, we did not 
		modify fixed_group in the feature dictionary since all motif residues belong to
		the same group (initialized to group 0).

		Implemention of Algorithm 1.

		Args:
			np_features:
				A feature dictionary containing information on an input structure 
				of length N, including
					-	aatype: 
							[N, 20] one-hot encoding on amino acid types
					-	num_chains: 
							[1] number of chains in the structure
					-	num_residues: 
							[1] number of residues in the structure
					-	num_residues_per_chain: 
							[1] an array of number of residues by chain
					-	atom_positions: 
							[N, 3] an array of Ca atom positions
					-	residue_mask: 
							[N] residue mask to indicate which residue position is masked
					-	residue_index: 
							[N] residue index (started from 0)
					-	chain_index: 
							[N] chain index (started from 0)
					-	fixed_sequence_mask: 
							[N] mask to indicate which residue contains conditional
							sequence information
					-	fixed_structure_mask: 
							[N, N] mask to indicate which pair of residues contains
							conditional structural information
					-	fixed_group:
							[N] group index to indicate which group the residue belongs to
							(useful for specifying multiple functional motifs)
					-	interface_mask:
							[N] deprecated and set to all zeros.

		Returns:
			np_features:
				An updated feature dictionary.
		"""

		# Sanity check
		assert np_features['num_chains'] == 1, 'Input must be monomer'

		# Sample number of motif residues
		motif_n_res = np.random.randint(
			np.floor(np_features['num_residues'] * self.motif_min_pct_res),
			np.ceil(np_features['num_residues'] * self.motif_max_pct_res)
		)

		# Sample number of motif segments
		motif_n_seg = np.random.randint(
			self.motif_min_n_seg,
			min(self.motif_max_n_seg, motif_n_res) + 1
		)

		# Sample motif segments
		indices = sorted(np.random.choice(motif_n_res - 1, motif_n_seg - 1, replace=False) + 1)
		indices = [0] + indices + [motif_n_res]
		motif_seg_lens = [indices[i+1] - indices[i] for i in range(motif_n_seg)]

		# Generate motif mask
		segs = [''.join(['1'] * l) for l in motif_seg_lens]
		segs.extend(['0'] * (np_features['num_residues'] - motif_n_res))
		random.shuffle(segs)
		motif_sequence_mask = np.array([int(elt) for elt in ''.join(segs)]).astype(bool)
		motif_structure_mask = motif_sequence_mask[:, np.newaxis] * motif_sequence_mask[np.newaxis, :]
		motif_structure_mask = motif_structure_mask.astype(bool)

		# Update
		np_features['fixed_sequence_mask'] = motif_sequence_mask
		np_features['fixed_structure_mask'] = motif_structure_mask

		return np_features