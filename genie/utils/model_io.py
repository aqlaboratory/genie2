import os
import glob
import numpy as np

from genie.config import Config
from genie.diffusion.genie import Genie


def get_versions(rootdir, name):
	"""
	Fetch model version, sorted in ascending order.

	Args:
		rootdir:
			Root directory.
		name:
			Model name.

	Returns:
		A list of model versions, sorted in ascending order.
	"""
	basedir = os.path.join(rootdir, name)
	return sorted([
		int(version_dir.split('_')[-1])
		for version_dir in glob.glob(os.path.join(basedir, 'version_*'), recursive=False)
	])

def get_epochs(rootdir, name, version):
	"""
	Fetch checkpoint epochs, sorted in ascending order.

	Args:
		rootdir:
			Root directory.
		name:
			Model name.
		version:
			Model version.

	Returns:
		A list of checkpoint epochs, sorted in ascending order.
	"""
	basedir = os.path.join(rootdir, name)
	return sorted([
		int(epoch_filepath.split('=')[-1].split('.')[0])
		for epoch_filepath in glob.glob(os.path.join(basedir, 'version_{}'.format(version), 'checkpoints', '*.ckpt'))
	])

def load_config(rootdir, name):
	"""
	Load configuration file.

	Args:
		rootdir:
			Root directory.
		name:
			Model name.

	Returns:
		An instance of Config (defined in config.py).
	"""
	return Config(os.path.join(rootdir, name, 'configuration'))

def load_default_model(rootdir, name):
	"""
	Load default model without any pretrained weight.

	Args:
		rootdir:
			Root directory.
		name:
			Model name.

	Returns:
		An instance of Genie (defined in diffusion/genie.py).
	"""
	return Genie(load_config(rootdir, name))

def load_model(rootdir, name, version=None, epoch=None):
	"""
	Load model from training directory. By default, the latest model version 
	and checkpoint epoch are used. If no pretrained weight is available, a
	default Genie is created without any pretrained weight.

	Args:
		rootdir:
			Root directory.
		name:
			Model name.
		version:
			Model version. Default to None.
		epoch:
			Checkpoint epoch. Default to None.

	Returns:
		An instance of Genie (defined in diffusion/genie.py).
	"""

	# Check for latest version if needed
	available_versions = get_versions(rootdir, name)
	if version is None:
		if len(available_versions) == 0:
			print('No checkpoint available (version)')
			print('Using default untrained model')
			return load_default_model(rootdir, name)
		else:
			version = np.max(available_versions)
	else:
		assert version in available_versions, 'Missing checkpoint version: {}'.format(version)

	# Check for latest epoch if needed
	available_epochs = get_epochs(rootdir, name, version)
	if epoch is None:
		if len(available_epochs) == 0:
			print('No checkpoint available (epoch)')
			print('Using default untrained model')
			return load_default_model(rootdir, name)
		else:
			epoch = np.max(available_epochs)
	else:
		assert epoch in available_epochs, 'Missing checkpoint epoch: {}'.format(epoch)

	# Load configuration
	config = load_config(rootdir, name)

	# Define checkpoint
	ckpt_filepath = os.path.join(
		rootdir,
		name,
		'version_{}'.format(version), 
		'checkpoints',
		'epoch={}.ckpt'.format(epoch)
	)

	# Load model
	print('Loading checkpoint: {}'.format(ckpt_filepath))
	return Genie.load_from_checkpoint(ckpt_filepath, config=config)

def load_pretrained_model(rootdir, name, epoch):
	"""
	Load pretrained Genie.

	Args:
		rootdir:
			Root directory.
		name:
			Model name.
		epoch:
			Checkpoint epoch.

	Returns:
		An instance of Genie (defined in diffusion/genie.py).
	"""

	# load configuration
	config = load_config(rootdir, name)

	# define checkpoint
	ckpt_filepath = os.path.join(
		rootdir,
		name,
		'checkpoints',
		'epoch={}.ckpt'.format(epoch)
	)

	# check
	if not os.path.exists(ckpt_filepath):
		print('Missing checkpoint: {}'.format(ckpt_filepath))
		exit(0)

	# load model
	print('Loading checkpoint: {}'.format(ckpt_filepath))
	return Genie.load_from_checkpoint(ckpt_filepath, config=config)