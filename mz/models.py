from sklearn.model_selection import StratifiedKFold
from utils import print_eval_metrics


def train_stratified_KFold_validation(clf, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
    n = 0
    for train_idx, val_idx in cv.split(X_train, y_train):
        n += 1
        x_tr = X_train.iloc[train_idx]
        x_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        clf.fit(x_tr, y_tr)
        y_pred = [1 if y > 0.5 else 0 for y in clf.predict(x_val)]
        print(f"{n} fold: ")
        print_eval_metrics(y_val, y_pred)
