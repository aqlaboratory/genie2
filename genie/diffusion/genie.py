import torch

from genie.diffusion.ddpm import DDPM
from genie.utils.affine_utils import T
from genie.utils.geo_utils import compute_frenet_frames
from genie.utils.loss import mse
from genie.utils.feat_utils import prepare_tensor_features


class Genie(DDPM):
	"""
	An instantiation of DDPM for unconditional diffusion.
	"""

	def training_step(self, batch, batch_idx):
		"""
		Training iteration.

		Args:
			batch:
				A batched feature dictionary with a batch size B, where each 
				structure is padded to the maximum sequence length N. It contains 
				the following information
					-	aatype: 
							[B, N, 20] one-hot encoding on amino acid types
					-	num_chains: 
							[B, 1] number of chains in the structure
					-	num_residues: 
							[B, 1] number of residues in the structure
					-	num_residues_per_chain: 
							[B, 1] an array of number of residues by chain
					-	atom_positions: 
							[B, N, 3] an array of Ca atom positions
					-	residue_mask: 
							[B, N] residue mask to indicate which residue position is masked
					-	residue_index: 
							[B, N] residue index (started from 0)
					-	chain_index: 
							[B, N] chain index (started from 0)
					-	fixed_sequence_mask: 
							[B, N] mask to indicate which residue contains conditional
							sequence information
					-	fixed_structure_mask: 
							[B, N, N] mask to indicate which pair of residues contains
							conditional structural information
					-	fixed_group:
							[B, N] group index to indicate which group the residue belongs to
							(useful for specifying multiple functional motifs)
					-	interface_mask:
							[B, N] deprecated and set to all zeros.
			batch_idx:
				[1] Index of this training batch.

		Returns:
			loss:
				[1] Motif-weighted mean of per-residue mean squared error between the predicted 
				noise and the groundtruth noise, averaged across all structures in the batch
		"""

		# Perform setup in the first run
		if not self.setup:
			self.setup_schedule()
			self.setup = True

		# Define features
		features = prepare_tensor_features(batch)

		# Sample time step
		s = torch.randint(
			self.config.diffusion['n_timestep'],
			size=(features['atom_positions'].shape[0],)
		).to(self.device) + 1

		# Sample noise
		z = torch.randn_like(features['atom_positions']) * features['residue_mask'].unsqueeze(-1)

		# Apply noise
		trans_s = self.sqrt_alphas_cumprod[s].view(-1, 1, 1) * features['atom_positions'] + \
			self.sqrt_one_minus_alphas_cumprod[s].view(-1, 1, 1) * z
		rots_s = compute_frenet_frames(
			trans_s,
			features['chain_index'],
			features['residue_mask']
		)
		ts = T(rots_s, trans_s)

		# Predict noise
		output = self.model(ts, s, features)

		# Compute masks
		condition_mask = features['residue_mask'] * features['fixed_sequence_mask']
		infill_mask = features['residue_mask'] * ~features['fixed_sequence_mask']

		# Compute condition and infill losses
		condition_losses = mse(output['z'], z, condition_mask, aggregate='sum')
		infill_losses = mse(output['z'], z, infill_mask, aggregate='sum')

		# Compute weighted losses
		unweighted_losses = (condition_losses + infill_losses) / features['num_residues']
		weighted_losses = (self.config.training['condition_loss_weight'] * condition_losses + infill_losses) / \
			(self.config.training['condition_loss_weight'] * torch.sum(condition_mask, dim=-1) + torch.sum(infill_mask, dim=-1))

		# Aggregate
		unweighted_loss = torch.mean(unweighted_losses)
		weighted_loss = torch.mean(weighted_losses)
		self.log('unweighted_loss', unweighted_loss, on_step=True, on_epoch=True)
		self.log('weighted_loss', weighted_loss, on_step=True, on_epoch=True)

		# Log
		batch_mask = torch.sum(condition_mask, dim=-1) > 0
		condition_losses = condition_losses / torch.sum(condition_mask, dim=-1)
		infill_losses = infill_losses / torch.sum(infill_mask, dim=-1)
		for i in range(batch_mask.shape[0]):
			if batch_mask[i]:
				self.log('motif_mse_loss', condition_losses[i], on_step=True, on_epoch=True)
				self.log('scaffold_mse_loss', infill_losses[i], on_step=True, on_epoch=True)
			else:
				self.log('unconditional_mse_loss', infill_losses[i], on_step=True, on_epoch=True)
			
		return weighted_loss
