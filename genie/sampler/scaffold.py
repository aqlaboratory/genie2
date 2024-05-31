import os
import copy
import torch

from genie.sampler.base import BaseSampler
from genie.utils.feat_utils import (
	create_np_features_from_motif_pdb,
	save_np_features_to_pdb
)
from genie.utils.motif_utils import save_motif_pdb


class ScaffoldSampler(BaseSampler):
	"""
	Sampler for motif scaffolding.
	"""

	def setup(self):
		"""
		Set up by adding additional required parameters.
		"""
		self.add_required_parameter('filepath')

	def on_sample_start(self, params):
		"""
		Set up output directories if necessary before sampling starts. This creates
		two output directories:
			-	a directory named 'pdbs', where each file stores the generated 
				structure in a PDB format
			- 	a directory named 'motif_pdbs', where each file - with the same filename 
				as the filename in the 'pdbs' directory - contains the motif structure, 
				aligned in residue indices with the corresponding generated structure and 
				stored in the PDB format.

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	filepath: path to the PDB file of motif configuration.
		"""
		pdbs_dir = os.path.join(params['outdir'], 'pdbs')
		if not os.path.exists(pdbs_dir):
			os.makedirs(pdbs_dir)
		motif_pdbs_dir = os.path.join(params['outdir'], 'motif_pdbs')
		if not os.path.exists(motif_pdbs_dir):
			os.makedirs(motif_pdbs_dir)

	def create_np_features(self, params):
		"""
		Creates a feature dictionary in numpy with updated motif sequence and 
		structure information (without padding or batching operations).

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	filepath: path to the PDB file of motif configuration.

		Returns:
			A feature dictionary containing information on an input structure 
			of length N, including
				-	aatype: 
						[N, 20] one-hot encoding on amino acid types. Amino acid 
						types of scaffold residues are default to 'ALA'.
				-	num_chains: 
						[1] number of chains in the structure
				-	num_residues: 
						[1] number of residues in the structure
				-	num_residues_per_chain: 
						[1] an array of number of residues by chain
				-	atom_positions: 
						[N, 3] an array of Ca atom positions. Atom positions of 
						scaffold residues are default to the origin. 
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
		return create_np_features_from_motif_pdb(params['filepath'])

	def on_sample_end(self, params, list_np_features):
		"""
		Save generated structures (in a directory named 'pdbs'), togther with their 
		corresponding motif information (in a directory named 'motif_pdbs').

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	filepath: path to the PDB file of motif configuration.
			list_np_features:
				A list of feature dictionaries, each of which has padding removed and 
				stores the following information on a generated structure of length N
					-	aatype: 
							[N, 20] one-hot encoding on amino acid types. Amino acid 
							types of scaffold residues are default to 'ALA'.
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
			
			# Define
			name = '{}_{}'.format(params['prefix'], params['offset'] + i)
			
			# Save pdb
			output_pdb_filepath = os.path.join(
				params['outdir'], 'pdbs', 
				'{}.pdb'.format(name)
			)
			save_np_features_to_pdb(np_features, output_pdb_filepath)

			# Save motif pdb
			output_motif_pdb_filepath = os.path.join(
				params['outdir'], 'motif_pdbs',
				'{}.pdb'.format(name)
			)
			save_motif_pdb(
				params['filepath'],
				np_features['fixed_sequence_mask'],
				output_motif_pdb_filepath
			)