import torch
import numpy as np
from abc import ABC, abstractmethod

from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.feat_utils import (
	convert_np_features_to_tensor,
	convert_tensor_features_to_numpy,
	batchify_np_features,
	debatchify_np_features
)


class BaseSampler(ABC):
	"""
	Base sampler for Genie 2.
	"""

	def __init__(self, model):
		"""
		Args:
			model:
				An instance of Genie, which is a derived class 
				defined in diffusion/genie.py.
		"""
		self.model = model
		self.device = model.device

		# Define required parameters
		self.required = ['scale', 'outdir', 'num_samples', 'prefix', 'offset']

		# Set up diffusion schedule parameters
		self.model.setup_schedule()

		# Custom set up
		self.setup()

	@abstractmethod
	def setup(self):
		"""
		Abstract method for custom set up during initialization stage.
		"""
		raise NotImplemented

	@abstractmethod
	def on_sample_start(self, params):
		"""
		Abstract method that run before sampling. Used to set up output
		directories if necessary.

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	user-defined parameters (by calling add_required_parameter).
		"""
		raise NotImplemented

	@abstractmethod
	def create_np_features(self, params):
		"""
		Abstract method that creates a feature dictionary in numpy 
		(without padding or batching operations).

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	user-defined parameters (by calling add_required_parameter).
		"""
		raise NotImplemented

	@abstractmethod
	def on_sample_end(self, params, list_np_features):
		"""
		Abstract methods that run after sampling. Used to save generation
		outputs such as generated structures and conditional information.

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	user-defined parameters (by calling add_required_parameter).
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
		raise NotImplemented

	def sample(self, params):
		"""
		Main function for sampling, which runs in the following steps
			-	validate if required parameters are provided
			-	prepare for sampling
			- 	sample structures
			-	postprocessing.

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	user-defined parameters (by calling add_required_parameter).
		"""
		self.validate_parameters(params)
		self.on_sample_start(params)
		list_np_features = self._sample(params)
		self.on_sample_end(params, list_np_features)

	def _sample(self, params):
		"""
		Sampling structures given input sampling parameters.

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	user-defined parameters (by calling add_required_parameter).

		Returns:
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

		# Create features
		features = convert_np_features_to_tensor(
			batchify_np_features([
				self.create_np_features(params)
				for _ in range(params['num_samples'])
			]),
			self.device
		)

		# Create frames
		trans = torch.randn_like(features['atom_positions'])
		rots = compute_frenet_frames(
			trans,
			features['chain_index'],
			features['residue_mask']
		)
		ts = T(rots, trans)

		# Define steps
		steps = reversed(np.arange(1, self.model.config.diffusion['n_timestep'] + 1))

		# Iterate
		for step in steps:

			# Define current diffusion timestep
			timesteps = torch.Tensor([step] * params['num_samples']).int().to(self.device)

			# Compute noise
			with torch.no_grad():
				z_pred = self.model.model(ts, timesteps, features)['z']

			# Compute posterior
			w_z = (1. - self.model.alphas[timesteps]) / self.model.sqrt_one_minus_alphas_cumprod[timesteps]
			trans_mean = (1. / self.model.sqrt_alphas[timesteps]).view(-1, 1, 1) * (ts.trans - w_z.view(-1, 1, 1) * z_pred)
			trans_mean = trans_mean * features['residue_mask'].unsqueeze(-1)

			# Sample
			if step == 1:

				# Compute rotations
				rots_mean = compute_frenet_frames(
					trans_mean,
					features['chain_index'],
					features['residue_mask']
				)

				# Compute frames
				ts = T(rots_mean.detach(), trans_mean.detach())

			else:

				# Compute translations
				trans_z = torch.randn_like(ts.trans)
				trans_sigma = self.model.sqrt_betas[timesteps].view(-1, 1, 1)
				trans = trans_mean + params['scale'] * trans_sigma * trans_z
				trans = trans * features['residue_mask'].unsqueeze(-1)

				# Compute rotations
				rots = compute_frenet_frames(
					trans,
					features['chain_index'],
					features['residue_mask']
				)

				# Compute frames
				ts = T(rots.detach(), trans.detach())

		# Postprocess
		features['atom_positions'] = ts.trans.detach().cpu()
		np_features = convert_tensor_features_to_numpy(features)
		list_np_features = debatchify_np_features(np_features)

		return list_np_features

	###############################
	###   Required Parameters   ###
	###############################

	def add_required_parameter(self, name):
		"""
		Add an additional required parameter.

		Args:
			name:
				Name of the required parameter to be added.
		"""
		self.required.append(name)

	def validate_parameters(self, params):
		"""
		Validate if all required parameters are present in the dictionary
		of sampling parameters.

		Args:
			params:
				A dictionary of sampling parameters. Required parameters include
					-	scale: sampling noise scale
					-	outdir: output directory
					-	num_samples: number of samples to generate (in a batch)
					-	prefix: prefix for filenames of generated structures
					-	offset: offset for distinguishing between batches
					-	user-defined parameters (by calling add_required_parameter).
		"""
		for name in self.required:
			if name not in params:
				return False
		return True
