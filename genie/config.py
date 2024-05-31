import os

int_or_none = lambda x: int(x) if x is not None else None
float_or_none = lambda x: float(x) if x is not None else None
str_list_or_none = lambda x: x.strip().split(',') if x is not None else None
int_list_or_none = lambda x: [int(y) for y in x.strip().split(',')] if x is not None else None
eval_if_str = lambda x: literal_eval(x) if isinstance(x, str) else x

class Config:

	def __init__(self, filename=None):
		config = {} if filename is None else self._load_config(filename)
		self._create_config(config)

	def _create_config(self, config):

		self.io = {

			'name':                               config.get('name',                                    None),
			'rootdir':                            config.get('rootDirectory',                           'runs'),
			'datadir':                            config.get('dataDirectory',                           'data/afdbreps_l-256_plddt_80/pdbs'),
			'min_n_res':              int_or_none(config.get('minimumNumResidues',                      20)),
			'max_n_res':              int_or_none(config.get('maximumNumResidues',                      256)),
			'max_n_chain':            int_or_none(config.get('maximumNumChains',                        1)),
			'validation_split':     float_or_none(config.get('validationSplit',                         None)),     

			# Motif conditioning
			'motif_prob':                   float(config.get('motifProbability',                        0.8)),
			'motif_min_pct_res':            float(config.get('motifMinimumPercentageResidues',          0.05)),
			'motif_max_pct_res':            float(config.get('motifMaximumPercentageResidues',          0.5)),
			'motif_min_n_seg':                int(config.get('motifMinimumNumberSegments',              1)),
			'motif_max_n_seg':                int(config.get('motifMaximumNumberSegments',              4)),

		}

		self.diffusion = {
			'n_timestep':                     int(config.get('numTimesteps',                            1000)),
			'schedule':                           config.get('schedule',                                'cosine'),
		}

		self.model = {

			# General
			'c_s':                            int(config.get('singleFeatureDimension',                  384)),
			'c_p':                            int(config.get('pairFeatureDimension',                    128)),
			'rescale':                      float(config.get('rescale',                                 1)),

			# Single feature network
			'c_pos_emb':                      int(config.get('positionalEmbeddingDimension',            256)),
			'c_chain_emb':                    int(config.get('chainEmbeddingDimension',                 64)),
			'c_timestep_emb':                 int(config.get('timestepEmbeddingDimension',              512)),

			# Pair feature network
			'relpos_k':                       int(config.get('relativePositionK',                       32)),
			'template_dist_min':            float(config.get('templateDistanceMinimum',                 2)),
			'template_dist_step':           float(config.get('templateDistanceStep',                    0.5)),
			'template_dist_n_bin':            int(config.get('templateDistanceNumBins',                 37)),

			# Pair transform network
			'n_pair_transform_layer':         int(config.get('numPairTransformLayers',                  5)),
			'include_mul_update':                 config.get('includeTriangularMultiplicativeUpdate',   True),
			'include_tri_att':                    config.get('includeTriangularAttention',              False),
			'c_hidden_mul':                   int(config.get('triangularMultiplicativeHiddenDimension', 128)),
			'c_hidden_tri_att':               int(config.get('triangularAttentionHiddenDimension',      32)),
			'n_head_tri':                     int(config.get('triangularAttentionNumHeads',             4)),
			'tri_dropout':                  float(config.get('triangularDropout',                       0.25)),
			'pair_transition_n':              int(config.get('pairTransitionN',                         4)),

			# Structure network
			'n_structure_layer':              int(config.get('numStructureLayers',                      8)),
			'n_structure_block':              int(config.get('numStructureBlocks',                      1)),
			'c_hidden_ipa':                   int(config.get('ipaHiddenDimension',                      16)),
			'n_head_ipa':                     int(config.get('ipaNumHeads',                             12)),
			'n_qk_point':                     int(config.get('ipaNumQkPoints',                          4)),
			'n_v_point':                      int(config.get('ipaNumVPoints',                           8)),
			'ipa_dropout':                  float(config.get('ipaDropout',                              0.1)),
			'n_structure_transition_layer':   int(config.get('numStructureTransitionLayers',            1)),
			'structure_transition_dropout': float(config.get('structureTransitionDropout',              0.1)),

		}

		self.training = {
			'seed':                           int(config.get('seed',                                    100)),
			'n_epoch':                        int(config.get('numEpoches',                              1)),
			'batch_size':                     int(config.get('batchSize',                               1)),
			'log_every_n_step':               int(config.get('logEverySteps',                           1000)),
			'checkpoint_every_n_epoch':       int(config.get('checkpointEveryEpoches',                  500)),
			'condition_loss_weight':          int(config.get('conditionLossWeight',                     1)),
		}

		self.optimization = {
			'lr':                           float(config.get('learningRate',                            1e-4))
		}

	def _load_config(self, filename):
		config = {}
		with open(filename) as file:
			for line in file:
				elts = line.split()
				if len(elts) == 2:
					if elts[1] == 'True':
						config[elts[0]] = True
					elif elts[1] == 'False':
						config[elts[0]] = False
					else:
						config[elts[0]] = elts[1]
		return config