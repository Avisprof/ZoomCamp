import bentoml
from bentoml.io import NumpyNdarray

model_ref = bentoml.sklearn.get('mlzoomcamp_homework:qtzdz3slg6mwwdu5')
#model_ref = bentoml.sklearn.get('mlzoomcamp_homework:jsi67fslz6txydu5')
model_runner = model_ref.to_runner()

svc = bentoml.Service('homework', runners=[model_runner])
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(vector):
    return model_runner.predict.run(vector)

