from sklearn.datasets import load_breast_cancer
from src.hpo.hpo_factory import HPOFactory
import logging

HYPERPARAMETER_RANGES = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0)
}

if __name__ == "__main__":
    data, labels = load_breast_cancer(return_X_y=True)

    # grasp_hpo = HPOFactory.create_hpo_strategy('GraspHpo')
    # grasp_best_trial_config, grasp_best_trial_score = grasp_hpo.hyperparameter_optimization(data, labels, HYPERPARAMETER_RANGES)
    # print('GRASP_HPO: ')
    # print('configuration: ', grasp_best_trial_config)
    # print('f1_score: ', grasp_best_trial_score)

    print()
    logger = logging.getLogger('ray"')
    logger.setLevel(logging.CRITICAL)
    hyperband_hpo = HPOFactory.create_hpo_strategy('Hyperband')
    hyperbanbd_best_trial_config, hyperbanbd_best_trial_score = hyperband_hpo.hyperparameter_optimization(data, labels, HYPERPARAMETER_RANGES)
    logger.setLevel(logging.NOTSET)
    print('Hyperband: ')
    print('configuration: ', hyperbanbd_best_trial_config)
    print('f1_score: ', hyperbanbd_best_trial_score)
