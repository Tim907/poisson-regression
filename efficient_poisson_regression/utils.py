from . import settings
from .datasets import BaseDataset
from .experiments import LeverageScoreSamplingExperiment, UniformSamplingExperiment, \
    LeverageScoreSamplingConvexHullExperiment

_logger = settings.get_logger()


def run_experiments(dataset: BaseDataset, min_size, max_size, step_size, num_runs):
    # check if results directory exists
    if not settings.RESULTS_DIR.exists():
        settings.RESULTS_DIR.mkdir()

    _logger.info("Starting leverage score sampling experiment")
    experiment = LeverageScoreSamplingConvexHullExperiment(
        p=dataset.p,
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        dataset=dataset,
        results_filename=settings.get_results_dir_p(dataset.p) / f"{dataset.get_name()}_leverage.csv",
    )
    experiment.run(parallel=False)

    _logger.info("Starting uniform sampling experiment")
    experiment_uniform = UniformSamplingExperiment(
        p=dataset.p,
        min_size=min_size,
        max_size=max_size,
        step_size=step_size,
        num_runs=num_runs,
        dataset=dataset,
        results_filename=settings.get_results_dir_p(dataset.p) / f"{dataset.get_name()}_uniform.csv",
    )
    experiment_uniform.run(parallel=False)

