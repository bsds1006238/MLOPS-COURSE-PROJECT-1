from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import loguniform

MODEL_PARAMS = [
    
    {
        "clf": [RandomForestClassifier(random_state=42)],
        "clf__n_estimators": randint(200, 1000),
        "clf__max_depth": [None] + list(range(5, 41)),
        "clf__min_samples_split": randint(2, 50),
        "clf__min_samples_leaf": randint(1, 20),
        "clf__max_features": ["sqrt", "log2", None, 0.3, 0.5, 0.7],
        "clf__bootstrap": [True, False],
    },
     {
        "clf": [SVC(probability=True)],
        "clf__C": loguniform(1e-3, 1e3),
        "clf__gamma": loguniform(1e-4, 1e0),
        "clf__kernel": ["rbf"],
        "clf__class_weight": [None, "balanced"],
    },
     {
        "clf": [LogisticRegression(max_iter=5000)],
        "clf__C": loguniform(1e-3, 1e3),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs", "liblinear"],
        "clf__class_weight": [None, "balanced"],
    }
]



RANDOM_SEARCH_PARAMS = {
    'n_iter' : 50,
    'cv': 5,
    'n_jobs' : -1,
    'random_state' : 42,
    'scoring' : 'accuracy'
}
