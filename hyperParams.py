import numpy as np
import re

BATCH_SIZE = 128
NUM_EPOCHS = 128
WEIGHT_DECAY = 1e-5
LR=2e-4
BETAS = (0.9, 0.999)


SIZE_IMAGE = 128
SIZE_KERNEL = 5
NUM_CONV_LAYERS = int(np.log2(SIZE_IMAGE)) - SIZE_KERNEL // 2       # = 5
SIZE_MINI_MAP = SIZE_IMAGE // (2 ** NUM_CONV_LAYERS)                # = 4

NUM_ENCODER_CHANNELS = 64 # First encoder conv layer output channel

NUM_GEN_CHANNELS = 2 ** (NUM_CONV_LAYERS + np.log2(NUM_ENCODER_CHANNELS) - 1) 
                    # = 1024 First generator convT layer input channel
NUM_FC_CHANNELS = int(NUM_GEN_CHANNELS * SIZE_MINI_MAP ** 2)
                    # = 1024 * 4 * 4 First generator fc layer output size

LENGTH_Z = 50

NUM_AGES = 10
NUM_GENDERS = 2
NUM_GENDERS_EXPANDED = NUM_GENDERS * (NUM_AGES // NUM_GENDERS)

LENGTH_LABEL = NUM_AGES + NUM_GENDERS
LENGTH_L = NUM_AGES + NUM_GENDERS_EXPANDED

# loss function
def loss_weights(epoch):
    WEIGHT = { 'eg': 1,
               'tv': 0.05,
               'ed': 0.001 + 0.00001 * epoch,
               'gd': 0.002 + 0.00003 * epoch,
               'di_gp': 10,}
    return WEIGHT
    # if 1 <= epoch < 101:
    # 	WEIGHT = {  'eg': 1,
    #                'tv': 0.05,
    #                'ed': 0.00001 * epoch,
    #                'gd': 0.00002 * epoch,
    #                'di_gp': 20,}
    # 	return WEIGHT
    # else:
    # 	WEIGHT = {  'eg': 1,
    #                'tv': 0.05,
    #                'ed': 0.001,
    #                'gd': 0.002,
    #                'di_gp': 10,}
    # 	return WEIGHT
        
MALE = 0
FEMALE = 1

UTKFACE_DEFAULT_PATH = './data/UTKFace'
UTKFACE_ORIGINAL_IMAGE_FORMAT = re.compile('^(\d+)_(\d+)_\d+_(\d+)\.jpg\.chip\.jpg$')

TRAINED_MODEL_EXT = '.dat'
TRAINED_MODEL_FORMAT = "{}" + TRAINED_MODEL_EXT


