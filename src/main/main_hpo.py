from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier

from sklearn.datasets import load_breast_cancer, load_digits
from grasp_core import GRASP_HPO
import sys



class HPO_SIM:

    def __init__(self) -> None:
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None


    def prepare_dataset(self, dataset):
        x = dataset.data
        y = dataset.target
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,random_state=1)
        

    
    def evaluate_solution(self, params) -> float:
        xgboost_classifier = XGBClassifier(**params)

        # f1_score
        xgboost_classifier.fit(self.x_train, self.y_train)
        y_pred = xgboost_classifier.predict(self.x_test)
        return f1_score(self.y_test, y_pred, average='weighted')
        # k fold below, comment out 3 lines up
        scoring=make_scorer(f1_score,average='weighted')
        scores = cross_val_score(xgboost_classifier,self.x_train,self.y_train,cv=5,error_score='raise',scoring=scoring)
        return scores.mean()
    

if __name__ == '__main__':

    sim = HPO_SIM()
    sim.prepare_dataset(load_digits())

    hp_ranges = {
        'n_estimators': (50, 500),
        'max_depth': (3, 10),
        'colsample_bytree': (0.5, 1),
        'reg_lambda': (0.01, 1.0),
        'subsample': (0.5, 1.0)
    }

    ghpo = GRASP_HPO(sim.evaluate_solution)
    ghpo.configure_ranges(hp_ranges)
    try: ghpo.set_gen_n(int(sys.argv[1]))
    except: pass

    score, solution = ghpo.optimize()

    print('Best score: {}\nFrom HPs: {}'.format(score, solution))