from functools import partial

from src.hpo.hpo_strategy import HPOStrategy
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import f1_score
import xgboost as xgb
from sklearn.model_selection import train_test_split


search_space = {
    'n_estimators': (50, 500),
    'max_depth': (3, 10),
    'colsample_bytree': (0.5, 1),
    'reg_lambda': (0.01, 1.0),
    'subsample': (0.5, 1.0)
}


class Hyperband(HPOStrategy):
    def hyperparameter_optimization(self, data, labels, search_space):
        ray.init(local_mode=False, log_to_driver=False)  # Initialize Ray (or specify your cluster configuration)

        # Define the hyperparameter search space
        config = {
            "max_depth": tune.randint(search_space['max_depth'][0], search_space['max_depth'][1]),
            "subsample": tune.uniform(search_space['subsample'][0], search_space['subsample'][1]),
            "colsample_bytree": tune.uniform(search_space['colsample_bytree'][0], search_space['colsample_bytree'][1]),
            "n_estimators": tune.randint(search_space['n_estimators'][0], search_space['n_estimators'][1]),
            "reg_lambda": tune.uniform(search_space['reg_lambda'][0], search_space['reg_lambda'][1]),
            "log_level": "ERROR"
        }

        # Define the ASHA scheduler for hyperparameter optimization
        scheduler = ASHAScheduler(
            max_t=100,  # Maximum number of training iterations
            grace_period=10,  # Minimum number of iterations for each trial
            reduction_factor=2,  # Factor by which trials are pruned
        )

        partial_train_xgboost = partial(train_xgboost, data=data, labels=labels)

        # Run the hyperparameter search
        analysis = tune.run(
            partial_train_xgboost,
            config=config,
            num_samples=10,  # Number of trials to run
            metric="f1_score",  # Metric to optimize
            mode="max",  # Maximize the F1 score
            scheduler=scheduler
        )

        # Get the best configuration
        best_trial = analysis.get_best_trial(metric="f1_score", mode="max")
        best_f1_score = best_trial.last_result["f1_score"]

        # print("Best hyperparameters:", best_trial.config)
        # print("Best F1 score:", best_f1_score)

        # Close Ray
        ray.shutdown()
        return best_trial.config, best_f1_score


# Define your XGBoost training function
def train_xgboost(config, data, labels):
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    params = {
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        'num_class': 10,
        "max_depth": config["max_depth"],
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"],
        #"n_estimators": config["n_estimators"],
        "reg_lambda": config["reg_lambda"],
        "seed": 42,
    }

    # Train XGBoost model
    evals = [(dtrain, "train"),(dtest, "eval")]
    bst = xgb.train(
        params, dtrain, num_boost_round = config["n_estimators"], evals=evals, verbose_eval=False
    )

    # Predict
    preds = bst.predict(dtest)
    #threshold = 0.5  # You can adjust this threshold as needed
    #binary_preds = [1 if p > threshold else 0 for p in preds]

    # Calculate F1 score
    f1 = f1_score(y_test, preds, average='weighted')

    return {"f1_score": f1}
