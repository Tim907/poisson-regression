from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Synthetic, Covertype
from efficient_probit_regression.experiments import LewisSamplingExperiment

MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 1


import numpy as np


epsilon = 0
epsilon = 0.1

#dataset = Synthetic(p=2, variant=1)
dataset = Synthetic(p=1, variant=1)
#dataset = Synthetic(p=2, variant=2)
#dataset = Synthetic(p=1, variant=1)

dataset = Covertype()
dataset.p = 1

experiment = LewisSamplingExperiment(
    p=dataset.p,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    dataset=dataset,
    results_filename=settings.get_results_dir_p(dataset.p)
                     / f"{dataset.get_name()}_lewis-fast_p_{dataset.p}.csv",
)
experiment.run(parallel=True)
