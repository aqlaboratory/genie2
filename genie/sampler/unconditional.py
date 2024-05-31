import os

from genie.sampler.base import BaseSampler
from genie.utils.feat_utils import (
	create_empty_np_features,
	save_np_features_to_pdb
)


class UnconditionalSampler(BaseSampler):

	def setup(self):
		"""
		Set up by adding additional required parameters.
		"""
		self.add_required_parameter('length')

	def on_sample_start(self, params):
		"""
		Set up an output directory if necessary before sampling starts. The directory 
		is named 'pdbs', where each file stores the generated structure in a PDB format.

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	length: target sequence length.
		"""
		pdbs_dir = os.path.join(params['outdir'], 'pdbs')
		if not os.path.exists(pdbs_dir):
			os.makedirs(pdbs_dir)

	def create_np_features(self, params):
		"""
		Creates a feature dictionary in numpy (without padding or batching operations).

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	length: target sequence length.

		Returns:
			A feature dictionary containing information on an input structure 
			of length N, including
				-	aatype: 
						[N, 20] one-hot encoding on amino acid types. All amino acid
						types are set to 'ALA' since Genie 2 is sequence-agnostic.
				-	num_chains: 
						[1] number of chains in the structure
				-	num_residues: 
						[1] number of residues in the structure
				-	num_residues_per_chain: 
						[1] an array of number of residues by chain
				-	atom_positions: 
						[N, 3] an array of Ca atom positions. Atom positions of 
						all residues are default to the origin. 
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
		return create_empty_np_features([params['length']])

	def on_sample_end(self, params, list_np_features):
		"""
		Save generated structures (in a directory named 'pdbs').

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	length: target sequence length.
			list_np_features:
				A list of feature dictionaries, each of which has padding removed and 
				stores the following information on a generated structure of length N
					-	aatype: 
							[N, 20] one-hot encoding on amino acid types. All amino acid
							types are set to 'ALA' since Genie 2 is sequence-agnostic.
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
		for i, np_features in enumerate(list_np_features):
			name = '{}_{}'.format(params['prefix'], params['offset'] + i)
			output_pdb_filepath = os.path.join(
				params['outdir'], 'pdbs', 
				'{}.pdb'.format(name)
			)
			save_np_features_to_pdb(np_features, output_pdb_filepath)