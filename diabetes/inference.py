
import fire
import json
import joblib
import logging
import numpy as np


def inference(x: json, model: str = None):
    """
    Args:
        model: name of the inference model to use
        x: diabetes instance to be classified
    """
    with open(x, 'r') as f:
        test_data = json.load(f)
        test_data = test_data['instance']

    models = {
        'AdaBoost': './outputs/AdaBoost_model.pkl',
        'GradientBoosting': './outputs/GradientBoosting_model.pkl',
        'Random Forest': './outputs/RandomForest_model.pkl',
        'Voting Classifier': './outputs/votingClassifier_model.pkl'
    }

    if model == 'adaboost':
        models = {'AdaBoost': './outputs/AdaBoost_model.pkl'}
    if model == 'gradientboost':
        models = {'GradientBoosting': './outputs/GradientBoosting_model.pkl'}
    if model == 'randomforest':
        models = {'Random Forest': './outputs/RandomForest_model.pkl'}
    if model == 'vclassifier':
        models = {'Voting Classifier': './outputs/votingClassifier_model.pkl'}

    for cls_name, saved_model in models.items():
        model = joblib.load(saved_model)

        x = np.array(test_data)
    
        pred = model.predict(x)
        pred_proba = model.predict_proba(x)

        logger.info(f"{cls_name}: Predicted class is {pred[0]}, with {pred_proba.max()*100:.2f}% confidence.")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    # file handler
    file_handler = logging.FileHandler('./outputs/inference.log')
    file_handler.setFormatter(formatter)

    # stream handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # run inference
    fire.Fire(inference)
    
