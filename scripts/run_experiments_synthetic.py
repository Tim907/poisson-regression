from efficient_poisson_regression.datasets import Synthetic, Covertype, OnlineRetail, Diabetes, SyntheticSimplexGaussian
from efficient_poisson_regression.utils import run_experiments

MIN_SIZE = 15000
MAX_SIZE = 40000
STEP_SIZE = 1000
NUM_RUNS = 201

MIN_SIZE = 50
MAX_SIZE = 600
STEP_SIZE = 50
dataset = SyntheticSimplexGaussian(n=100000, d=6, p=1, variant=1, seed=3)
#dataset = SyntheticSimplexGaussian(n=100000, d=6, p=2, variant=1, seed=3)

run_experiments(dataset, min_size=MIN_SIZE, max_size=MAX_SIZE, step_size=STEP_SIZE, num_runs=NUM_RUNS)

