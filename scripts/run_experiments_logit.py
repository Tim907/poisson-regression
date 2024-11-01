from efficient_poisson_regression import settings
from efficient_poisson_regression.datasets import Covertype, KDDCup, Webspam  # noqa
from efficient_poisson_regression.experiments import LogitSamplingExperiment

MIN_SIZE = 500
MAX_SIZE = 15000
STEP_SIZE = 500
NUM_RUNS = 21

# MIN_SIZE = 1000
# MAX_SIZE = 30000
# STEP_SIZE = 1000
# NUM_RUNS = 21

P = 1

dataset = Covertype()
# dataset = KDDCup()
# dataset = Webspam()

experiment = LogitSamplingExperiment(
    p=P,
    min_size=MIN_SIZE,
    max_size=MAX_SIZE,
    step_size=STEP_SIZE,
    num_runs=NUM_RUNS,
    dataset=dataset,
    results_filename=settings.get_results_dir_p(P)
    / f"{dataset.get_name()}_logit_p_{P}.csv",
)
experiment.run(parallel=True)
