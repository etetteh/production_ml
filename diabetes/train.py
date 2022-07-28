import os
import fire
import json
import joblib
import logging
import optuna

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import plotly.express as px


def create_processing_pipeline(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Standardizes the numerical features, and onehot encodes the categorical fields
    Returns:
        data processing pipeline
    """
    # numerical column processor
    numeric_features = numeric_cols
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # categorical column processor
    categorical_features = categorical_cols
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # combined processor
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer', numeric_transformer, numeric_features),
            ('categorical_transformer', categorical_transformer, categorical_features)])

    return preprocessor


def get_classifier_params(hparams: dict, classifier_name: str) -> dict:
    """ Get the tuned hparams of a classifier.

    Args:
        hparams: tuned hyperparameters of the classifier
        classifier_name: name of classifier whose hyperparameters have been tuned

    Returns:
        A dict of hyperparameters
    """
    params = hparams[classifier_name]
    data_random_state = params.pop('data_random_state')
    return params, data_random_state


def compute_metrics(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> (np.ndarray, dict):
    """
    Computes model performance metrics on test data split.
    Specifically, it computes the `confusion matrix`, `accuracy`, `precision`, `recall`,
    and `auc`.
    Returns the y_scores from .predict_proba, and a dict of the metrics

    Args:
        model: classification model pipeline
        X_test: ndarray of train labels
        y_test: ndarray of test labels

    Returns:
        y_scores: predicted probabilities on X_test
        metrics: computed performance metrics
    """
    # get predictions from test data
    predictions = model.predict(X_test)
    y_scores = model.predict_proba(X_test)

    # compute performance metrics
    cm = confusion_matrix(y_test, predictions)
    acc = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    auc_ = roc_auc_score(y_test, y_scores[:, 1])

    metrics = {
        'confusion_matrix': cm,
        'accuracy': round(acc, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'auc': round(auc_, 4)
    }

    return y_scores, metrics


def plot_roc_curve(y_test: np.ndarray, y_scores: np.ndarray, output_dir: str, classifier_name: str):
    """
    Plots ROC curve. Saves resulting plot as .html (interactive copy), and .png (static copy)

    Args:
        y_test: array of true labels
        y_scores: array of predicted labels probabilities
        classifier_name: string of classification algorithm name, e.g 'Random Forest'
        output_dir: directory to save performance metric (ROC-AUC curve) figures
    """
    # compute false-positive rate and true-positive rate
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])
    title = classifier_name + ' ROC Curve'

    # plot ROC curve
    fig = px.area(
        x=fpr, y=tpr,
        title=f"{title} (AUC={auc(fpr, tpr):.4f})",
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=450, height=400
    )

    # plot threshold line (main diagonal)
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    # set figure margins
    fig.update_layout(
        margin=dict(l=10, r=15, t=40, b=30)
    )
    # save plots as .png and .html files
    fig.write_html(output_dir + title + '.html')
    fig.write_image(output_dir + title + '.png')
    # fig.show()


def save_results(model: Pipeline, metrics: dict, output_dir: str, filename: str):
    """
    Saves model as a pickle file, and model performance metrics in
    json format.

    Args:
        model: classifier to save
        metrics: metrics to save (excludes confusion matrix)
        filename: name of the classifier
        output_dir: output directory to store files
    """

    # pickle model
    classifier_filename = filename + '_model.pkl'
    joblib.dump(model, output_dir + classifier_filename)

    # save metrics
    classifier_metrics_filename = filename + '_metrics.json'
    metrics.pop('confusion_matrix')
    with open(output_dir + classifier_metrics_filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved {filename} classifier as: {classifier_filename}")
    logger.info(f"Saved {filename} classifier metrics: {classifier_metrics_filename}")


class Objective(object):
    """
    Objective for running hyperparameter tuning for each classifier

    Args:
        dataset: a pandas.DataFrame file
        classifier: the name of the classifier whose hparams are to be tuned (e.g. RandomForest, GradientBoosting etc.)
        dataset_attr: features and label names and their indexes
    """

    def __init__(self, dataset: pd.DataFrame, dataset_attr: dict, classifier: str):
        self.dataset = dataset
        self.classifier = classifier
        self.dataset_attr = dataset_attr


    def __call__(self, trial):
        features, label, numeric_cols, categorical_cols = self.dataset_attr["features"], self.dataset_attr["label"], self.dataset_attr["numeric_cols"], self.dataset_attr["categorical_cols"]

        X, y = self.dataset[features].values, self.dataset[label].values

        # split data into train set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=trial.suggest_int('data_random_state', 0, 99,
                                                                                                                 step=1))

        preprocessor = create_processing_pipeline(numeric_cols, categorical_cols)
        pipeline = None

        if self.classifier == 'RandomForest':
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'n_estimators': trial.suggest_int('n_estimators', 10, 100, step=1),
                'random_state': trial.suggest_int('random_state', 0, 99, step=1),
            }
            # Create preprocessing and training pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('random_forest', RandomForestClassifier(**params))]
                                )
        elif self.classifier == 'AdaBoost':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 10, 100, step=1),
                'algorithm': trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R']),
                'random_state': trial.suggest_int('random_state', 0, 99, step=1),
            }
            # Create preprocessing and training pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('ada_boost', AdaBoostClassifier(**params))]
                                )
        elif self.classifier == 'GradientBoosting':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1.0, log=True),
                'loss': trial.suggest_categorical('loss', ['log_loss', 'exponential']),
                'n_estimators': trial.suggest_int('n_estimators', 10, 100, step=1),
                'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
                'random_state': trial.suggest_int('random_state', 0, 99, step=1),
            }
            # Create preprocessing and training pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('gradient_boosting', GradientBoostingClassifier(**params))]
                                )
        elif self.classifier == 'XGBClassifier':
            params = {
                'use_label_encoder': False,
                'verbosity': 0,
                'silent':True,
                'n_estimators': trial.suggest_int('n_estimators', 10, 100, step=1),
                'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                'random_state': trial.suggest_int('random_state', 0, 99, step=1),
                'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
            }

            if params['booster'] in ['gbtree', 'dart']:
                params['max_depth'] = trial.suggest_int('max_depth', 3, 9, step=2)
                params['min_child_weight'] = trial.suggest_int('min_child_weight', 2, 10)
                params['eta'] = trial.suggest_float('eta', 1e-8, 1.0, log=True)
                params['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
                params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])

            if params['booster'] == 'dart':
                params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
                params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
            # Create preprocessing and training pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('xgb_classifier', XGBClassifier(**params))]
                                )
        elif self.classifier == 'CatboostClassifier':
            params = {
                'verbose': 0,
                'objective': trial.suggest_categorical('objective', ['Logloss', 'CrossEntropy']),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.01, 0.1),
                'depth': trial.suggest_int('depth', 1, 12),
                'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'random_state': trial.suggest_int('random_state', 0, 99, step=1),
            }
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.1, 1)
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('xgb_classifier', CatBoostClassifier(**params))]
                                )

        model = pipeline.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)

        auc_ = roc_auc_score(y_test, y_scores[:, 1])

        return auc_


def train_base_classifiers(dataset: pd.DataFrame, dataset_attr: dict, classifiers: dict, output_dir: str) -> dict:
    """
    Fits a classifier on the given dataframe.

    Args:
        dataset: pandas dataframe
        dataset_attr: features and label names and their indexes
        classifiers: dict of base classifiers with dataset split random seed
        output_dir: directory to save model results, and figures
    Returns:
        dict of performance metrics
    """
    # Separate features and labels
    features, label, numeric_cols, categorical_cols = dataset_attr["features"], dataset_attr["label"], dataset_attr["numeric_cols"], dataset_attr["categorical_cols"]
    X, y = dataset[features].values, dataset[label].values

    preprocessor = create_processing_pipeline(numeric_cols, categorical_cols)

    results = dict()
    # Create preprocessing and training pipeline
    for classifier_name, classifier_attr in classifiers.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=classifier_attr['random_state'])
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            (classifier_name, classifier_attr['classifier'])
        ]
        )

        # fit the pipeline to train a random forest model on the training set
        model = pipeline.fit(X_train, y_train)

        # compute performance metrics
        y_scores, metrics = compute_metrics(model, X_test, y_test)

        # save model performance metrics
        results[classifier_name] = metrics

        # plot classifier ROC curve
        plot_roc_curve(y_test, y_scores, output_dir, classifier_name)

        print()
        logger.info(f"Finished training {classifier_name} classifier...")
        logger.info(f"{classifier_name} classifier performance metrics: \n{metrics}")
        save_results(model=model, metrics=metrics, output_dir=output_dir, filename=classifier_name)

    return results


def train_voting_classifiers(dataset: pd.DataFrame, dataset_attr: dict, classifiers: dict, output_dir: str) -> (dict, Pipeline):
    """
    Fits a voting classifier on the given dataframe.

    Args:
        dataset: pandas dataframe of dataset
        dataset_attr: features and label names and their indexes
        classifiers: dict of base classifiers with dataset split random seed
        output_dir: directory to save model results, and figures
    Returns:
        dict of performance metrics
    """
    features, label, numeric_cols, categorical_cols = dataset_attr["features"], dataset_attr["label"], dataset_attr["numeric_cols"], dataset_attr["categorical_cols"]

    # Separate features and label
    X, y = dataset[features].values, dataset[label].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=classifiers['AdaBoost']['random_state'], stratify=y)

    preprocessor = create_processing_pipeline(numeric_cols, categorical_cols)

    top_classifiers = ['CatboostClassifier', 'XGBClassifier', 'GradientBoosting']

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('voting_classifier', VotingClassifier(estimators=[
                                   (cls_name, cls['classifier']) for cls_name, cls in classifiers.items() if
                                   cls_name in top_classifiers],
                                   voting='soft',
                                   # weights=[5, 1]
                               ))
                               ]
                        )

    # fit the pipeline to train a random forest model on the training set
    model = pipeline.fit(X_train, y_train)

    # compute performance metrics
    y_scores, metrics = compute_metrics(model, X_test, y_test)

    # plot classifier ROC curve
    plot_roc_curve(y_test, y_scores, output_dir, 'Voting Classifier')

    # output results
    print()
    logger.info(
        f"Voting Classifier Performance Metrics: Confusion Matrix: {metrics['confusion_matrix']} \
        | Accuracy: {metrics['accuracy']} | Overall Precision {metrics['precision']} | Overall Recall {metrics['recall']} | AUC {metrics['auc']}"
    )
    return metrics, model


def main(dataset: str , col_names: json, n_trials: int = 200):
    """
    1. Runs hparams optimization for base classifiers
    2. trains base classifiers
    3. trains a voting classifier using the best performing base classifiers

    Args:
        dataset: csv file to be used
        col_names: a json file containing feature and label names
        n_trials: number of trials for hparams optimization
    """
    dataset = pd.read_csv(dataset)

    with open(col_names, 'r') as f:
        dataset_attr = json.load(f)

    X, y = dataset[dataset_attr['features']].values, dataset[dataset_attr['label']].values

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.20, stratify=y)

    # log dataset info
    logger.info(f"Dataset feature columns: {dataset_attr['features']}")
    logger.info(f"Dataset label column: {dataset_attr['label']}")
    logger.info(f"Shape of training dataset: {X_train.shape}")
    logger.info(f"Shape of test dataset: {X_test.shape}")

    logger.info(f"Number of hyperparameter tuning trials: {n_trials}")

    # tune hparams
    tuned_hparams = dict()
    classifier_names = {'ada': 'AdaBoost', 'gb': 'GradientBoosting', 'rf': 'RandomForest', 'xgb': 'XGBClassifier', 'catb': 'CatboostClassifier'}
    for classifier in classifier_names.values():
        objective = Objective(dataset=dataset, dataset_attr=dataset_attr, classifier=classifier)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        tuned_hparams[classifier] = study.best_trial.params
        print()
        print(f"{classifier} best trial results: {study.best_trial}")
        logger.info(f"{classifier} best trial params: {study.best_trial.params}")
        print()

    # get tuned hparams
    ada_boost_params, ada_boost_data_random_state = get_classifier_params(tuned_hparams, classifier_names['ada'])
    gradient_boosting_params, gradient_boosting_data_random_state = get_classifier_params(tuned_hparams, classifier_names['gb'])
    random_forest_params, random_forest_data_random_state = get_classifier_params(tuned_hparams, classifier_names['rf'])
    xgb_classifier_params, xgb_classifier_data_random_state = get_classifier_params(tuned_hparams, classifier_names['xgb'])
    catb_classifier_params, catb_classifier_data_random_state = get_classifier_params(tuned_hparams, classifier_names['catb'])

    # load tuned hparams into their respective classifiers
    xgb_classifier_params['use_label_encoder'] = False
    xgb_classifier_params['verbosity'] = 0
    xgb_classifier_params['silent'] = True

    catb_classifier_params['verbose'] = 0

    classifiers_keys = {'cls': 'classifier', 'rands': 'random_state'}
    classifiers = {
        classifier_names['ada']: {classifiers_keys['cls']: AdaBoostClassifier(**ada_boost_params),
                                  classifiers_keys['rands']: ada_boost_data_random_state},
        classifier_names['gb']: {classifiers_keys['cls']: GradientBoostingClassifier(**gradient_boosting_params),
                                 classifiers_keys['rands']: gradient_boosting_data_random_state},
        classifier_names['rf']: {classifiers_keys['cls']: RandomForestClassifier(**random_forest_params),
                                 classifiers_keys['rands']: random_forest_data_random_state},
        classifier_names['xgb']: {classifiers_keys['cls']: XGBClassifier(**xgb_classifier_params),
                                  classifiers_keys['rands']: xgb_classifier_data_random_state},
        classifier_names['catb']: {classifiers_keys['cls']: CatBoostClassifier(**catb_classifier_params),
                                   classifiers_keys['rands']: catb_classifier_data_random_state}
    }

    # train classifiers and a voting classifiers
    base_classifiers_metrics = train_base_classifiers(dataset=dataset, dataset_attr=dataset_attr, classifiers=classifiers,
                                                      output_dir=output_dir)
    voting_classifier_metrics, model = train_voting_classifiers(dataset=dataset, dataset_attr=dataset_attr,
                                                                classifiers=classifiers,
                                                                output_dir=output_dir)

    # save model and performance metrics
    save_results(model=model, metrics=voting_classifier_metrics, output_dir=output_dir, filename='votingClassifier')


if __name__ == "__main__":
    output_dir = './outputs/'
    os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('Diabetes Classification: Training...')
    logger.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    # file handler
    file_handler = logging.FileHandler(output_dir+'diabetes.log', mode='w')
    file_handler.setFormatter(formatter)

    # stream handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Tuned hyperparameters, performance metrics, plots, and logs are saved in {output_dir}")

    fire.Fire(main)
