import requests

url = 'http://localhost:9696/predict'


lag_sales = {
    "product_line": "electronic_accessories",
    "city": "mandalay",
    "lag_1": 17,
    "lag_2": 24,
    "lag_3": 52,
    "lag_4": 36,
    "lag_5": 40
}


response = requests.post(url, json=lag_sales).json()
print(f'for the product "{response["product_line"]}" and city "{response["city"]}" the next week forecast of sales is {response["predict_sales"]} items')






