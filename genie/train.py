import os
import wandb
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.trainer import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from genie.config import Config
from genie.utils.model_io import load_model

from genie.data.data_module import GenieDataModule


def main(args):

	# Load configuration
	config = Config(filename=args.config)

	# Create logger
	loggers = []
	if not args.test:
		loggers.append(
			TensorBoardLogger(
				save_dir=config.io['rootdir'],
				name=config.io['name']
			)
		)
		loggers.append(
			WandbLogger(
				project=f'[Genie] {config.io["name"]}'
			)
		)

	# Set up checkpoint callback
	checkpoint_callback = ModelCheckpoint(
		every_n_epochs=config.training['checkpoint_every_n_epoch'],
		filename='{epoch}',
		save_top_k=-1
	)

	# Initial random seeds
	seed_everything(config.training['seed'], workers=True)

	# Data module
	dm = GenieDataModule(
		**config.io,
		batch_size=config.training['batch_size']
	)

	# Model
	model = load_model(config.io['rootdir'], config.io['name'])

	# Trainer
	trainer = Trainer(
		devices=args.devices,
		num_nodes=args.num_nodes,
		accelerator='gpu',
		logger=loggers,
		strategy='ddp',
		deterministic=True,
		enable_progress_bar=args.test,
		log_every_n_steps=config.training['log_every_n_step'],
		max_epochs=config.training['n_epoch'],
		callbacks=[checkpoint_callback]
	)

	# Run
	trainer.fit(model, dm)


if __name__ == '__main__':

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--devices', type=int, help='Number of GPU devices to use')
	parser.add_argument('-n', '--num_nodes', type=int, help='Number of nodes')
	parser.add_argument('-c', '--config', type=str, help='Path for configuration file', required=True)
	parser.add_argument('-t', '--test', action='store_true', help='Enable test mode', default=False)
	args = parser.parse_args()

	# Run
	main(args)