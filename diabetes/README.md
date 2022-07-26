# Diabetes Classification
In this project, we train an AdaBoost, Gradient Boosting, and Random Forest classifiers from the [scikit-learn](https://scikit-learn.org/stable/) machine learning library, and additional train an XGBoost and CatBoost classifiers from [xgboost](https://xgboost.readthedocs.io) and [catboost](https://catboost.ai) respectively,  
on a diabetes dataset, to classify whether a patient has diabetes or not.

## Stand alone Implementation
### installation
```
pip install -r requirements.txt
```
In case of issues regarding the installation of any of the library, refer to its documentation.

### training
1. Initial step involves hyper-parameter optimization for each of the classifier used.
2. Next, the tuned hyper-parameters are passed to each classifier accordingly, and the classifiers are trained/fitted.
3. Finally, a voting classifier is trained using two of the three classifiers based on the AUC performance metric of the classifiers.

The train.py script saves each classifier as a pickle file, the classifier's corresponding performance metric as a json file,
and AUC plot as a png (static) and html (interactive) files.
The columns.json file is similar to:
```
{"features": ["Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age"],
"label": "Diabetic",
"numeric_cols": [0,1,2,3,4,5,6],
"categorical_cols": [7]
}
```

Example: to run 100 trials of hyper-parameter optimization, and train the classifiers:\
```
python train.py --dataset diabetes.csv --col_names columns.json --n_trials 100
```

### inference
The default mode of inference.py runs inference using all the classifiers. However, users can pass a specific classifier to use.\
Users can choose one of the following models: adaboost, gradientboost, randomforest, xgboost, catboost, vclassifier.\
The inference data must be in a json format.

Example: running inference with all classifiers\
```
python inference.py --test_data test_data.json
```

Example: running inference with Gradient Boosting\
```
python inference.py --test_data test_data.json --model gradientboost
``` 


## Docker Implementation
For users with no experience of docker, you might want to refer to the [docker documentation](https://docs.docker.com/get-started/overview/).

**NOTE:** CatBoost currently requires 64-bit version of Python, so you might have to change the `Dockerfile` to use `From amd64/python:3.9.13-slim`.
Also, `plotly` and `kaleido` don't currently work when using `amd64/...`, so you might have to comment out the plot function and calls.

### Build image
```
docker build -f Dockerfile -t <image-name> .
```

### Train model
```
docker run -v <absolute-path-of-local-dir>:/diabetes -t -d --rm --name diabetes-train  diabetes python diabetes/train.py --dataset diabetes.csv --col_names columns.json --n_trials 100
```

### Perform inference
```
docker run -v <absolute-path-of-local-dir>:/diabetes -it --rm --name diabetes-inference diabetes python diabetes/inference.py --test_data test_data.json
```

### Perform inference with a specific model
You may choose one of ['adaboost', 'gradientboost', 'randomforest', 'xgboost', 'catboost', 'vclassifier'].\
```
docker run -v <absolute-path-of-local-dir>:/diabetes -it --rm --name diabetes-inference diabetes python diabetes/inference.py --test_data test_data.json --model gradientboost
```
