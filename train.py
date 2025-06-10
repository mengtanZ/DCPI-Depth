from __future__ import absolute_import, division, print_function

from options import Options
from trainer import Trainer
import warnings

warnings.filterwarnings('ignore')

options = Options()
opts = options.parse()

import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.vis_device_id)

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()