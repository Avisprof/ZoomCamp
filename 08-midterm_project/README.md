# Predict future sales

The purpose of this project is to create a model which will predict total sales for every product and city in the next week with daily historical sales data

## 2. About the dataset

The growth of supermarkets in most populated cities are increasing and market competitions are also high. The dataset is one of the historical sales of supermarket company which has recorded in 3 different branches for 3 months data.

You can get this dataset from [kaggle](https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales)

## 3. About files and folders in this repo

|  File name |      Description       |
|:--------:|:-----------------------------------:|
|    README.md   |Details about this project| 
|    supermarket_sales.csv   |Dataset with supermarket sales from the kaggle |
|    notebook.ipynb   |Jupyter notebook file with EDA, training different models and choose the best model |
|    train.py   |Python script for training xgboost model |
|    xgb_model.bin   |Saved model from the script `train.py` |
|    predict.py   |Python script that loads the model and puts into a flask webservice|
|    Pipfile   |The concrete requirements for a Python Application|
|    Pipfile.lock   |The details of the environment|
|    Dockerfile   |Text document that contains all the commands to create docker container image|
|    predict-test.py   |Python script that sends a request to the host on virtual machine |
 
## 4. Model training
In time series data we wish to predict some variable given only a trailing window of its previous values. 
In this model used [lag_1, lag_2, lag_3, lag_4, lag_5] features, which contains sales data for previous weeks

## 5. How to reproduce the project


## 6. Local deployment of the model
Thist model can be deployed locally using Docker and the following commands
```
docker build -t kaggle_superstore .
docker run -it --rm -p 9696:9696 kaggle_superstore
```

For the predict 
python predict_test.py
