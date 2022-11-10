import pandas as pd
import itertools
import pickle

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error



input_file = 'supermarket_sales.csv' 
output_file = 'xgb_model.bin'
lags = [1,2,3,4,5]
xgb_params = {
        'eval_metric': 'mae',
        'max_depth': 1,
        'eta':0.1,
        'nthread': 8,  
        'seed': 1,
        'verbosity': 1,
}


# read the data
print(f'read input file: {input_file}')
data = pd.read_csv(input_file, parse_dates=['Date'])

# prepare the data
print('prepare the data with lags')
data.columns = data.columns.str.lower().str.replace(' ', '_')

cat_columns = data.columns[data.dtypes == 'object']
for col in cat_columns:
    data[col] = data[col].str.lower().str.replace(' ', '_')

# sort values by date
data.sort_values(by='date', inplace=True)

# add column "week", and get aggregate data
data['week'] = data.date.dt.isocalendar().week
df_week = data.groupby(['city', 'product_line','week'], as_index=False).agg({'quantity':'sum'})

products = df_week.product_line.unique()
cities = df_week.city.unique()
weeks = range(1,14)

# get the full dataframe with all sales by the product_line and city in each week
df_full = pd.DataFrame(itertools.product(products, cities, weeks), columns=['product_line', 'city', 'week'])
df_full = df_full.merge(df_week, how='left', on=['product_line', 'city','week'])
df_full.fillna(0, inplace=True)


def get_df_lags(df, lags=[1,2,3]):
    """
    add lags to dataframe
    return new dataframe
    """
    
    df_lag = df.copy()
    
    for lag in lags:
        df_lag[f'lag_{lag}'] = df_lag.quantity.shift(lag)
            
    return df_lag

def train(df_train, y_train):

    dv = DictVectorizer(sparse=False)
    train_dicts = df_train[columns].to_dict(orient='records')

    X_train = dv.fit_transform(train_dicts)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(xgb_params, dtrain)

    return dv, model

def predict(df, dv, model):

    dicts = df[columns].to_dict(orient='records')
    X = dv.transform(dicts)
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest).round(0)

    return y_pred


df_full_lags = get_df_lags(df_full, lags).dropna()
columns = ['product_line', 'city'] + list(df_full_lags.columns[df_full_lags.columns.str.startswith('lag')])

# validation
print(f'doing evaluation of xgb model with params: {xgb_params}')

df_train = df_full_lags.query('week < 12')
df_val = df_full_lags.query('week == 12')

df_train_full = df_full_lags.query('week < 13')
df_test = df_full_lags.query('week == 13')

y_train = df_train['quantity'].values
y_val = df_val['quantity'].values
y_test = df_test['quantity'].values

dv, model = train(df_train, y_train)
y_pred = predict(df_val, dv, model)
mae = mean_absolute_error(y_val, y_pred)
print(f'MAE: {mae:.3f}')

print('trainig the final model')
dv, model = train(df_train_full, df_train_full['quantity'].values)
y_pred = predict(df_test, dv, model)
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae:.3f}')

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'xgb model is saved to {output_file}')


