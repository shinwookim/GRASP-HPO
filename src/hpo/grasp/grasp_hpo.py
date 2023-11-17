from sklearn.model_selection import train_test_split
#from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import f1_score

from src.hpo.hpo_strategy import HPOStrategy
from src.hpo.grasp.phase1grasp import Construction
from src.hpo.grasp.phase2grasp import LocalSearch


LOCAL_SEARCH_ITERATIONS = 50
BUILDING_PHASE_ITERATIONS = 20


class GraspHpo(HPOStrategy):

    def __init__(self) -> None:
        self.phase1 = Construction(self.evaluate_solution, BUILDING_PHASE_ITERATIONS)
        self.phase2 = LocalSearch(self.evaluate_solution, LOCAL_SEARCH_ITERATIONS)


    def hyperparameter_optimization(self, data, labels, search_space):
        x_train, x_test, y_train, y_test = self.prepare_dataset(data, labels)
        dtrain, dtest = self.format_dataset(x_train, x_test, y_train, y_test)

        best_intermediate_combinations = self.phase1.building_phase(dtrain, dtest, y_test, search_space)

        return self.phase2.local_search(best_intermediate_combinations, dtrain, dtest, y_test, search_space)


    def prepare_dataset(self, data, labels):
        return train_test_split(data, labels, test_size=0.2, random_state=1)
    
    def format_dataset(self, xtrain, xtest, ytrain, ytest):
        dtrain = xgb.DMatrix(xtrain, label=ytrain)
        dtest = xgb.DMatrix(xtest, label=ytest)
        return dtrain, dtest


    #def evaluate_solution(self, params, x_train, x_test, y_train, y_test):
    #    xgboost_classifier = XGBClassifier(**params)
    #    xgboost_classifier.fit(x_train, y_train)
    #    y_pred = xgboost_classifier.predict(x_test)
    #    return f1_score(y_test, y_pred, average='weighted')
    
    def evaluate_solution(self, config, dtrain, dtest, y_test):
        params = {
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "num_class": 10,
            "max_depth": config["max_depth"],
            #"min_child_weight": params["min_child_weight"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            #"eta": config["learning_rate"],
            #"gamma": config["gamma"],
            "lambda": config["reg_lambda"],
            #"n_estimators": config["n_estimators"],
            "seed": 42,
        }

        evals = [(dtrain, "train"),(dtest, "eval")]
        bst = xgb.train(
            params, dtrain, num_boost_round = config["n_estimators"], evals=evals, verbose_eval=False
        )

        # Predict
        preds = bst.predict(dtest)
        #threshold = 0.5  # You can adjust this threshold as needed
        #print(preds)
        #binary_preds = [1 if p > threshold else 0 for p in preds]

        # Calculate F1 score
        f1 = f1_score(y_test, preds,average='weighted')
        return f1
