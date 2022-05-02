experiment_folder = "./experiment_results/run11"

import coloredlogs, logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')

logger.debug("this is a debugging message")
logger.info("this is an informational message")
logger.warning("this is a warning message")
logger.error("this is an error message")
logger.critical("this is a critical message")

import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


import ide.run_from_path as rfp

experiment_folder = "./experiment_results/run_noise_de"
rfp.run_experiments_from_folder(experiment_folder, parallel=True)