import math
import torch


def get_betas(n_timestep, schedule):
	"""
	Set up a variance schedule.

	Args:
		n_timestep:
			Number of diffusion timesteps (denoted as N).
		schedule:
			Name of variance schedule. Currently support 'cosine'.

	Returns:
		A sequence of variances (with a length of N + 1), where the
		i-th element denotes the variance at diffusion step i. Note 
		that diffusion step is one-indexed and i = 0 indicates the 
		un-noised stage.
	"""
	if schedule == 'cosine':
		return cosine_beta_schedule(n_timestep)
	else:
		print('Invalid schedule: {}'.format(schedule))
		exit(0)

def cosine_beta_schedule(n_timestep):
	"""
	Set up a cosine variance schedule.

	Args:
		n_timestep:
			Number of diffusion timesteps (denoted as N).

	Returns:
		A sequence of variances (with a length of N + 1), where the
		i-th element denotes the variance at diffusion step i. Note 
		that diffusion step is one-indexed and i = 0 indicates the 
		un-noised stage.
	"""
	steps = n_timestep + 1
	x = torch.linspace(0, n_timestep, steps)
	alphas_cumprod = torch.cos((x / steps) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	return torch.concat([
		torch.zeros((1,)),
		torch.clip(betas, 0, 0.999)
	])