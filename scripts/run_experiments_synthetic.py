from efficient_probit_regression import settings
from efficient_probit_regression.datasets import Synthetic, Covertype, OnlineRetail, Diabetes
from efficient_probit_regression.experiments import LeverageScoreSamplingConvexHullExperiment
from efficient_probit_regression.settings import get_logger
from efficient_probit_regression.utils import run_experiments

MIN_SIZE = 15000
MAX_SIZE = 40000
STEP_SIZE = 1000
NUM_RUNS = 51

#dataset = Synthetic(n=100000, d=6, p=1, variant=1, seed=2)
#dataset = Synthetic(n=100000, d=6, p=2, variant=1, seed=2)
#dataset = Synthetic(n=100000, d=6, p=1, variant=2, seed=2)
dataset = Synthetic(n=100000, d=6, p=2, variant=2, seed=2)

run_experiments(dataset, min_size=MIN_SIZE, max_size=MAX_SIZE, step_size=STEP_SIZE, num_runs=NUM_RUNS)

