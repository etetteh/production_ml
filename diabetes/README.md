# Diabetes Classification
In this project, we train an AdaBoost, Gradient Boosting, and Random Forest classifiers from the Scikit-Learn machine learning library,
on a diabetes dataset, to classify whether a patient has diabetes or not.

## Stand alone Implementation
### installation
`pip install -r requirements.txt`

### training
1. Initial step involves hyper-parameter optimization for each of the classifier used.\
2. Next, the tuned hyper-parameters are passed to each classifier accordingly, and the classifiers are trained/fitted.\
3. Finally, a voting classifier is trained using two of the three classifiers based on the AUC performance metric of the classifiers.

The train.py script saves each classifier as a pickle file, the classifier's corresponding performance metric as a json file,
and AUC plot as a png (static) and html (interactive) files.

Example: to run 100 trials of hyper-parameter optimization, and train the classifiers:\
`python train.py --dataset diabetes.csv --n_trials 100`

### inference
The default mode of inference.py runs inference using all the classifiers. However, users can pass a specific classifier to use.\
Users can choose one of the following models: adaboost, gradientboost, randomforest, vclassifier.\
The inference data must be in a json format.\

Example: running inference with all classifiers\
`python inference.py --x test_data.json`

Example: running inference with Gradient Boosting\
`python inference.py --x test_data.json --model gradientboost` 


## Docker Implementation
For users with no experience of docker, you might want to refer to the [docker documentation](https://docs.docker.com/get-started/overview/).

### build image
`docker build -f Dockerfile -t <image-name> .`

### run training
`docker run -v <absolute-path-of-local-dir>:/diabetes -t -d --rm --name diabetes-train  diabetes python diabetes/train.py --dataset diabetes.csv --n_trials 100`

### run inference
`docker run -v <absolute-path-of-local-dir>:/diabetes -it --rm --name diabetes-inference diabetes python diabetes/inference.py --x test_data.json`

### run inference with a specific model
You may choose one of ['adaboost', 'gradientboost', 'randomforest', 'vclassifier'].\
`docker run -v <absolute-path-of-local-dir>:/diabetes -it --rm --name diabetes-inference diabetes python diabetes/inference.py --x test_data.json --model gradientboost`
