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
from sklearn.metrics import auc, accuracy_score, confusion_matrix, precision_score, \
    recall_score, roc_auc_score, \
    roc_curve

import plotly.express as px


def create_processing_pipeline() -> ColumnTransformer:
    """
    Standardizes the numerical features, and onehot encodes the categorical fields
    Returns:
        data processing pipeline
    """
    # numerical column processor
    numeric_features = slice(0, 7)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    # categorical column processor
    categorical_features = slice(7, 8)
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


def compute_metrics(model: Pipeline, x_test: np.ndarray, y_test: np.ndarray) -> (np.ndarray, dict):
    """
    Computes model performance metrics on test data split.
    Specifically, it computes the `confusion matrix`, `accuracy`, `precision`, `recall`,
    and `auc`.
    Returns the y_scores from .predict_proba, and a dict of the metrics

    Args:
        model: classification model pipeline
        x_test: ndarray of train labels
        y_test: ndarray of test labels

    Returns:
        y_scores: predicted probabilities on X_test
        metrics: computed performance metrics
    """
    # get predictions from test data
    predictions = model.predict(x_test)
    y_scores = model.predict_proba(x_test)

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
        data: a pandas.DataFrame file
        classifier: the name of the classifier whose hparams are to be tuned (e.g. RandomForest, GradientBoosting etc.)
    """

    def __init__(self, data, classifier, features):
        self.data = data
        self.classifier = classifier
        self.features = features

    def __call__(self, trial):
        # get features and labels
        label = 'Diabetic'
        x, y = self.data[self.features].values, self.data[label].values

        # split data into train set and test set
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=trial.suggest_int('data_random_state', 0, 99,
                                                                                           step=1))

        preprocessor = create_processing_pipeline()
        pipeline = None

        if self.classifier == 'RandomForest':
            params = {
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
                'loss': trial.suggest_categorical('loss', ['log_loss', 'exponential']),
                'n_estimators': trial.suggest_int('n_estimators', 10, 100, step=1),
                'criterion': trial.suggest_categorical('criterion', ['friedman_mse', 'squared_error']),
                'random_state': trial.suggest_int('random_state', 0, 99, step=1),
            }
            # Create preprocessing and training pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('gradient_boosting', GradientBoostingClassifier(**params))]
                                )

        model = pipeline.fit(x_train, y_train)
        y_scores = model.predict_proba(x_test)

        auc_ = roc_auc_score(y_test, y_scores[:, 1])

        return auc_


def train_base_classifiers(dataset: pd.DataFrame, features: list, classifiers: dict, output_dir: str) -> dict:
    """
    Fits a classifier on the given dataframe.

    Args:
        dataset: pandas dataframe
        features: a list of feature names
        classifiers: dict of base classifiers with dataset split random seed
        output_dir: directory to save model results, and figures
    Returns:
        dict of performance metrics
    """
    # Separate features and labels
    label = 'Diabetic'
    x, y = dataset[features].values, dataset[label].values

    preprocessor = create_processing_pipeline()

    results = dict()
    # Create preprocessing and training pipeline
    for classifier_name, classifier_attr in classifiers.items():
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=classifier_attr['random_state'])
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            (classifier_name, classifier_attr['classifier'])
        ]
        )

        # fit the pipeline to train a random forest model on the training set
        model = pipeline.fit(x_train, y_train)

        # compute performance metrics
        y_scores, metrics = compute_metrics(model, x_test, y_test)

        # save model performance metrics
        results[classifier_name] = metrics

        # plot classifier ROC curve
        plot_roc_curve(y_test, y_scores, output_dir, classifier_name)

        print()
        logger.info(f"Finished training {classifier_name} classifier...")
        logger.info(f"{classifier_name} classifier performance metrics: \n{metrics}")
        save_results(model=model, metrics=metrics, output_dir=output_dir, filename=classifier_name)

    return results


def train_voting_classifiers(dataset: pd.DataFrame, features: list, classifiers: dict, base_classifiers_metrics: dict,
                             output_dir: str) -> (dict, Pipeline):
    """
    Fits a voting classifier on the given dataframe, using the best top 2 base classifiers.

    Args:
        dataset: pandas dataframe of dataset
        features: a list of feature names
        classifiers: dict of base classifiers with dataset split random seed
        base_classifiers_metrics: dict of base classifiers' performance metrics
        output_dir: directory to save model results, and figures
    Returns:
        dict of performance metrics
    """

    # Separate features and labels
    label = 'Diabetic'
    x, y = dataset[features].values, dataset[label].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=classifiers['AdaBoost']['random_state'])

    preprocessor = create_processing_pipeline()

    # get top performing base classifiers using their auc scores
    sorted_dict = dict(sorted(base_classifiers_metrics.items(), key=lambda item: item[1]['auc'], reverse=True))
    sorted_dict_keys = list(sorted_dict.keys())
    top2_classifiers = sorted_dict_keys[:int(np.ceil(len(sorted_dict_keys) * 0.5))]

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('voting_classifier', VotingClassifier(estimators=[
                                   (cls_name, cls['classifier']) for cls_name, cls in classifiers.items() if
                                   cls_name in top2_classifiers],
                                   voting='soft',
                                   weights=[5, 1]))
                               ]
                        )

    # fit the pipeline to train a random forest model on the training set
    model = pipeline.fit(x_train, y_train)

    # compute performance metrics
    y_scores, metrics = compute_metrics(model, x_test, y_test)

    # plot classifier ROC curve
    plot_roc_curve(y_test, y_scores, output_dir, 'Voting Classifier')

    # output results
    print()
    logger.info(
        f"Voting Classifier Performance Metrics: Confusion Matrix: {metrics['confusion_matrix']} \
        | Accuracy: {metrics['accuracy']} | Overall Precision {metrics['precision']} | Overall Recall {metrics['recall']} | AUC {metrics['auc']}"
    )
    return metrics, model


def main(dataset: str = 'diabetes.csv', n_trials: int = 100):
    """
    1. Runs hparams optimization for base classifiers
    2. trains base classifiers
    3. trains a voting classifier using the best performing base classifiers

    Args:
        dataset: csv file to be used
        n_trials: number of trials for hparams optimization
    """
    # create output directory to store results and plots
    # output_dir = 'outputs/'
    # os.makedirs(output_dir, exist_ok=True)

    dataset = pd.read_csv(dataset)
    features = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI',
                'DiabetesPedigree', 'Age']

    # tune hparams
    tuned_hparams = dict()
    classifier_names = ['AdaBoost', 'GradientBoosting', 'RandomForest']
    for classifier in classifier_names:
        objective = Objective(dataset, classifier, features)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        tuned_hparams[classifier] = study.best_trial.params
        print()
        print(f"{classifier} best trial results: {study.best_trial}")
        logger.info(f"{classifier} best trial params: {study.best_trial.params}")
        print()

    # get tuned hparams
    ada_boost_params, ada_boost_data_random_state = get_classifier_params(tuned_hparams, classifier_names[0])
    gradient_boosting_params, gradient_boosting_data_random_state = get_classifier_params(tuned_hparams, classifier_names[1])
    random_forest_params, random_forest_data_random_state = get_classifier_params(tuned_hparams, classifier_names[2])

    # load tuned hparams into their respective classifiers
    classifiers_keys = ['classifier', 'random_state']
    classifiers = {
        classifier_names[0]: {classifiers_keys[0]: AdaBoostClassifier(**ada_boost_params),
                              classifiers_keys[1]: ada_boost_data_random_state},
        classifier_names[1]: {classifiers_keys[0]: GradientBoostingClassifier(**gradient_boosting_params),
                              classifiers_keys[1]: gradient_boosting_data_random_state},
        classifier_names[2]: {classifiers_keys[0]: RandomForestClassifier(**random_forest_params),
                              classifiers_keys[1]: random_forest_data_random_state}
    }

    # train classifiers and a voting classifiers
    base_classifiers_metrics = train_base_classifiers(dataset=dataset, features=features, classifiers=classifiers,
                                                      output_dir=output_dir)
    voting_classifier_metrics, model = train_voting_classifiers(dataset=dataset, features=features,
                                                                classifiers=classifiers,
                                                                base_classifiers_metrics=base_classifiers_metrics,
                                                                output_dir=output_dir)

    # save model and performance metrics
    save_results(model=model, metrics=voting_classifier_metrics, output_dir=output_dir, filename='votingClassifier')


if __name__ == "__main__":
    output_dir = './outputs/'
    os.makedirs(output_dir, exist_ok=True)

    global seed
    seed = 1

    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    # file handler
    file_handler = logging.FileHandler(output_dir+'diabetes.log')
    file_handler.setFormatter(formatter)

    # stream handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    fire.Fire(main)
