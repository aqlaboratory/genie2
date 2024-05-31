import os
import glob
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from genie.data.dataset import GenieDataset
from genie.utils.feat_utils import summarize_pdb


class GenieDataModule(LightningDataModule):
	"""
	Pytorch Lightning data module for Genie 2 training.
	Derived from pytorch_lightning.LightningDataModule.

	Key functions:
		-	filter input dataset based on the number of residues
		-	randomly split input dataset into a training and a validation dataset
		-	create a training dataset and a validation dataset by writing the pdb 
			names into train.txt and validation.txt respectively
		-	initialize datasets and dataloaders based on pdb names.
	"""

	def __init__(
		self,
		name,
		rootdir,
		datadir,
		min_n_res,
		max_n_res,
		max_n_chain,
		validation_split,
		batch_size,
		motif_prob,
		motif_min_pct_res,
		motif_max_pct_res,
		motif_min_n_seg,
		motif_max_n_seg
	):
		"""
		Initialize data module.

		Args:
			name:
				Name of the training run.
			rootdir:
				Root directory.
			datadir:
				Data directory.
			min_n_res:
				Minimum number of residues in a structure.
			max_n_res:
				Maximum number of residues in a structure.
			max_n_chain:
				Maximum number of chains in a structure.
			validation_split:
				Either the number of structures in the validation dataset (by
				specifying a positive integer), or the percentage of structures 
				from the dataset to be used for validation (by specifying a 
				float between 0 and 1).
			batch_size:
				Number of structures in a training batch.
			motif_prob:
				Percentage of motif-conditional training tasks.
			motif_min_pct_res:
				Minimum percentage of residues (out of the total sequence length 
				of the input structure) to be defined as motif residues.
			motif_max_pct_res:
				Maximum percentage of residues (out of the total sequence length 
				of the input structure) to be defined as motif residues.
			motif_min_n_seg:
				Minimum number of motif segments.
			motif_max_n_seg:
				Maximum number of motif segments.
		"""
		super(GenieDataModule, self).__init__()

		# Base parameters
		self.name = name
		self.rootdir = rootdir
		self.min_n_res = min_n_res
		self.max_n_res = max_n_res
		self.max_n_chain = max_n_chain
		self.validation_split = validation_split
		self.batch_size = batch_size

		# Dataset parameters
		self.datadir = datadir
  
		# Additional parameters related to motif-conditional training task
		self.motif_prob = motif_prob
		self.motif_min_pct_res = motif_min_pct_res
		self.motif_max_pct_res = motif_max_pct_res
		self.motif_min_n_seg = motif_min_n_seg
		self.motif_max_n_seg = motif_max_n_seg


	def setup(self, stage=None):
		"""
		Set up data module before training.

		Args:
			stage:
				Stage name required by LightningDataModule. Default to None.
		"""

		# Define filepaths for training and validation dataset
		# These files keep track of the set of structures used in the training
		# and validation process respectively, by recording their names.
		train_filepath = os.path.join(self.rootdir, self.name, 'train.txt')
		validation_filepath = os.path.join(self.rootdir, self.name, 'validation.txt')

		# Check if the training dataset and validation dataset are created
		# This ensures that subsequent training runs utilize the same training-
		# validation split on the dataset.
		if os.path.exists(train_filepath):
			if self.validation_split is not None:
				# Sanity check for existence of validation file
				assert os.path.exists(validation_filepath)
		else:
			print(f'INFO: creating dataset...')

			# Load names of all filtered structures
			# Filtering for input structures is customizable via the function _validate.
			names = self._fetch_names(self.datadir)

			# Check if validation dataset is required
			if self.validation_split is not None:

				# Split structures into a training set and a validation set
				# Splitting of input structures is customizable via the function _split.
				train_names, validation_names = self._split(names)

				# Save structure names into files
				self._save_names(train_names, train_filepath)
				self._save_names(validation_names, validation_filepath)

			else:

				# Save structure names into files
				train_names = names
				self._save_names(train_names, train_filepath)

	def train_dataloader(self):
		"""
		Set up dataloader for training.

		Returns:
			An instance of torch.utils.data.DataLoader.
		"""

		# Create dataset information
		dataset_info = {
			'datadir': self.datadir,
			'names': self._load_names(
				os.path.join(self.rootdir, self.name, 'train.txt')
			)
		}

		# Create dataset
		dataset = GenieDataset(
			dataset_info,
			self.min_n_res,
			self.max_n_res,
			self.max_n_chain,
			self.motif_prob,
			self.motif_min_pct_res,
			self.motif_max_pct_res,
			self.motif_min_n_seg,
			self.motif_max_n_seg,
		)

		# Create dataloader
		return DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=True
		)

	############################
	###   Helper Functions   ###
	############################

	def _load_names(self, filepath):
		"""
		Load structure names.

		Args:
			filepath:
				Path to the file containing a list of structure names.

		Returns:
			names:
				A list of structure names.
		"""
		with open(filepath) as file:
			names = [line.strip() for line in file]
		return names

	def _save_names(self, names, filepath):
		"""
		Save structure names.

		Args:
			names:
				A list of structure names.
			filepath:
				Path to output file that stores this list of structure names.
		"""
		with open(filepath, 'w') as file:
			file.write('\n'.join(names))

	def _fetch_names(self, datadir):
		"""
		Fetch names for structures in the data directory, that pass the set of 
		filters defined in the function _validate.

		Args:
			datadir:
				Data directory.

		Returns:
			A list of names for all passed structures.
		"""
		names = []
		for filepath in tqdm(glob.glob(os.path.join(datadir, '*.pdb.gz'))):
			if self._validate(filepath):
				names.append(filepath.strip().split('/')[-1].split('.')[0])
		return names

	def _split(self, names):
		"""
		Split structures into a training dataset and a validation dataset. 

		By default, the splitting is based on random selection and the parameter 
		validation_split specifies either the number of validation data points or 
		the percentage of the total dataset used for validation.

		Args:
			names:
				A list of structure names to be split.

		Returns:
			train_names:
				A list of names for structures used at the training stage.
			validation_names:
				A list of names for structures used at the valiadtion stage.
		"""
		split_idx = int(len(names) * self.validation_split) \
			if self.validation_split < 1 else int(self.validation_split)
		train_names = names[:-split_idx]
		validation_names = names[-split_idx:]
		return train_names, validation_names

	def _validate(self, filepath):
		"""
		Filter input structure based on the minimum and maximum number of residues.

		Args:
			filepath:
				Path to the PDB file of a structure.

		Returns:
			A boolean indicating whether the structure passes the set of filters.
		"""
		summary = summarize_pdb(filepath)
		return summary['num_residues'] >= self.min_n_res and \
			summary['num_residues'] <= self.max_n_res 