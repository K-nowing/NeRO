import argparse
import os
import sys

import options
from train.trainer import Trainer
from utils.base_utils import load_cfg
from utils.logger import logger

logger.process(os.getpid())
logger.title("[{}] (PyTorch code for training NeRF/BARF)".format(sys.argv[0]))
opt_path = sys.argv[1]
opt_cmd = options.parse_arguments(sys.argv[2:])
opt = options.set(opt_path, opt_cmd=opt_cmd)
options.save_options_file(opt)
Trainer(opt).run()

# parser = argparse.ArgumentParser()
# parser.add_argument('--cfg', type=str)
# flags = parser.parse_args()

# Trainer(load_cfg(flags.cfg)).run()
