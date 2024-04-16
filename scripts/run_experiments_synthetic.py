from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Synthetic, Covertype
from efficient_probit_regression.experiments import LeverageScoreSamplingConvexHullExperiment
from efficient_probit_regression.settings import get_logger

MIN_SIZE = 15000
MAX_SIZE = 40000
STEP_SIZE = 1000
NUM_RUNS = 51


logger = get_logger()
import numpy as np


#dataset = Synthetic(n=100000, d=10, p=1, variant=1, seed=1)
#dataset = Synthetic(n=100000, d=10, p=2, variant=1, seed=1)
#dataset = Synthetic(n=100000, d=10, p=1, variant=2, seed=1)
dataset = Synthetic(n=100000, d=10, p=2, variant=2, seed=1)


logger.info("Starting leverage score sampling experiment")
experiment = LeverageScoreSamplingConvexHullExperiment(
    p=dataset.p,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    dataset=dataset,
    results_filename=settings.get_results_dir_p(dataset.p)
    / f"{dataset.get_name()}_leverage_p_{dataset.p}.csv",
)
experiment.run(parallel=True)
