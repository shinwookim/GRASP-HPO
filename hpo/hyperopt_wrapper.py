from typing import List, Dict, Optional
import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import *
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

def hyperopt_tune(trainable, search_space, scheduler, concurrency, num_samples, resources_per_trial,
                  space: Optional[Dict] = None, metric: Optional[str] = None, 
                  mode: Optional[str] = None, points_to_evaluate: Optional[List[Dict]] = None, 
                  n_initial_points: int = 20, random_state_seed: Optional[int] = None, 
                  gamma: float = 0.25):
    """
    Wrapper for hyperopt search algorithm.
    """
    algo = HyperOptSearch(space=search_space, metric=metric, mode=mode, points_to_evaluate=points_to_evaluate, 
                          n_initial_points=n_initial_points, random_state_seed=random_state_seed, gamma=gamma)
    algo = ConcurrencyLimiter(algo, max_concurrent=concurrency)
    scheduler = scheduler
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            metric=metric,
            mode=mode,
            search_alg=algo,
            scheduler=scheduler,
            num_samples=num_samples,
            resources_per_trial=resources_per_trial,
        ),
    )
    results = tuner.fit()
    return results