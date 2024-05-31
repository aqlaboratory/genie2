import torch
from torch.optim import Adam
from abc import ABC, abstractmethod
from pytorch_lightning.core import LightningModule

from genie.model.model import Denoiser
from genie.diffusion.schedule import get_betas


class DDPM(LightningModule, ABC):
	"""
	Base denoising diffusion probabilistic model.
	"""

	def __init__(self, config):
		"""
		Args:
			config:
				A Config object (defined in config.py) containing parameters to 
				set up training dataset, model and training process.
		"""
		super(DDPM, self).__init__()
		self.config = config

		# Model for noise prediction at each diffusion timestep
		self.model = Denoiser(
			**self.config.model,
			n_timestep=self.config.diffusion['n_timestep'],
			max_n_res=self.config.io['max_n_res'],
			max_n_chain=self.config.io['max_n_chain']
		)

		# Flag for lazy setup
		self.setup = False

	def setup_schedule(self):
		"""
		Set up variance schedule and precompute its corresponding terms.
		"""
		self.betas = get_betas(
			self.config.diffusion['n_timestep'],
			self.config.diffusion['schedule']
		).to(self.device)

		self.alphas = 1. - self.betas
		self.alphas_cumprod = torch.cumprod(self.alphas, 0)
		self.alphas_cumprod_prev = torch.cat([
			torch.Tensor([1.]).to(self.device),
			self.alphas_cumprod[:-1]
		])
		self.one_minus_alphas_cumprod = 1. - self.alphas_cumprod
		
		self.sqrt_betas = torch.sqrt(self.betas)
		self.sqrt_alphas = torch.sqrt(self.alphas)
		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
		self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1. - self.alphas_cumprod_prev)
		self.sqrt_recip_alphas_cumprod = 1. / self.sqrt_alphas_cumprod
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

	@abstractmethod
	def training_step(self, batch, batch_idx):
		raise NotImplemented

	def configure_optimizers(self):
		return Adam(
			self.model.parameters(),
			lr=self.config.optimization['lr']
		)