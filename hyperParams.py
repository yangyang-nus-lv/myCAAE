import numpy as np
import re

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
    if 1 <= epoch < 31:
        WEIGHT = {  'eg': 1,
                    'tv': 0.1 - 0.08 / 30 * epoch, #0.01 at epoch10
                    'ez': 0.00001 * epoch, # 0.0001 at epoch10
                    'gd': 0.00002 * epoch, # 0.0002 at epoch10
		 }
        return WEIGHT
    elif:
        WEIGHT = {  'eg': 1,
                    'tv': 0.02,
                    'ez': 0.0003 + 0.0002 / 50 * (epoch - 30),
                    'gd': 0.0006 + 0.0004 / 50 * (epoch - 30), }
        return WEIGHT

MALE = 0
FEMALE = 1

UTKFACE_DEFAULT_PATH = './data/UTKFace'
UTKFACE_ORIGINAL_IMAGE_FORMAT = re.compile('^(\d+)_(\d+)_\d+_(\d+)\.jpg\.chip\.jpg$')

TRAINED_MODEL_EXT = '.dat'
TRAINED_MODEL_FORMAT = "{}" + TRAINED_MODEL_EXT


