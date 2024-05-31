import torch
from torch import nn

from genie.model.single_feature_net import SingleFeatureNet
from genie.model.pair_feature_net import PairFeatureNet
from genie.model.pair_transform_net import PairTransformNet
from genie.model.structure_net import StructureNet


class Denoiser(nn.Module):
	"""
	SE(3)-Equivariant Denoiser.

	Given a noisy structure at timestep t, the model predicts the noise that
	is added at timestep t. For further details, please refer to our Genie
	paper at https://arxiv.org/abs/2301.12485 and our Genie 2 paper at
	https://arxiv.org/abs/2405.15489.
	"""

	def __init__(
		self,
		c_s,
		c_p,
		n_timestep,
		rescale,

		# Parameters for single feature network
		c_pos_emb,
		c_chain_emb,
		c_timestep_emb,
		max_n_res,
		max_n_chain,

		# Parameters for pair feature network
		relpos_k,
		template_dist_min,
		template_dist_step,
		template_dist_n_bin,

		# Parameters for pair transform network
		n_pair_transform_layer,
		include_mul_update,
		include_tri_att,
		c_hidden_mul, 
		c_hidden_tri_att, 
		n_head_tri, 
		tri_dropout, 
		pair_transition_n,

		# Parameters for structure network
		n_structure_layer, 
		n_structure_block,
		c_hidden_ipa, 
		n_head_ipa, 
		n_qk_point, 
		n_v_point, 
		ipa_dropout,
		n_structure_transition_layer, 
		structure_transition_dropout

	):
		"""
		Args:
			c_s:
				Dimension of per-residue (single) representation.
			c_p:
				Dimension of paired residue-residue (pair) representation.
			n_timestep:
				Total number of diffusion timesteps.
			rescale:
				Rescale factor for coordinate space.
			*:
				Module-specfic parameters. Refer to its corresponding module
				for further details.
		"""
		super(Denoiser, self).__init__()
		self.rescale = rescale

		self.single_feature_net = SingleFeatureNet(
			c_s=c_s,
			n_timestep=n_timestep,
			c_pos_emb=c_pos_emb,
			c_chain_emb=c_chain_emb,
			c_timestep_emb=c_timestep_emb,
			max_n_res=max_n_res,
			max_n_chain=max_n_chain
		)
		
		self.pair_feature_net = PairFeatureNet(
			c_s=c_s,
			c_p=c_p,
			n_timestep=n_timestep,
			relpos_k=relpos_k,
			template_dist_min=template_dist_min,
			template_dist_step=template_dist_step,
			template_dist_n_bin=template_dist_n_bin
		)

		self.pair_transform_net = PairTransformNet(
			c_p=c_p,
			n_pair_transform_layer=n_pair_transform_layer,
			include_mul_update=include_mul_update,
			include_tri_att=include_tri_att,
			c_hidden_mul=c_hidden_mul,
			c_hidden_tri_att=c_hidden_tri_att,
			n_head_tri=n_head_tri,
			tri_dropout=tri_dropout,
			pair_transition_n=pair_transition_n
		) if n_pair_transform_layer > 0 else None

		self.structure_net = StructureNet(
			c_s=c_s,
			c_p=c_p,
			n_structure_layer=n_structure_layer,
			n_structure_block=n_structure_block,
			c_hidden_ipa=c_hidden_ipa,
			n_head_ipa=n_head_ipa,
			n_qk_point=n_qk_point,
			n_v_point=n_v_point,
			ipa_dropout=ipa_dropout,
			n_structure_transition_layer=n_structure_transition_layer,
			structure_transition_dropout=structure_transition_dropout
		)

	def forward(self, ts, timesteps, features):
		"""
		Args:
			ts:
				[B, n_res] Frames at a given diffusion timestep.
			timesteps:
				[B] Diffusion timesteps.
			features:
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

		Returns:
			A dictionary containing
				z:
					[B, N, 3] Predicted noise vector.
		"""

		# Initialize
		trans = ts.trans

		# Rescale
		ts = ts.scale_translation(self.rescale)

		# Predict
		s = self.single_feature_net(ts, timesteps, features)
		p = self.pair_feature_net(s, ts, timesteps, features)
		if self.pair_transform_net is not None:
			p = self.pair_transform_net(p, features)
		states, ts = self.structure_net(s, p, ts, features)

		# Descale
		ts = ts.scale_translation(1./self.rescale)

		return {
			'z': trans - ts.trans,
		}
