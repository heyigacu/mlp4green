# mlp4green



## 1. How to train your model:
### 1.1.1 prepare yout datasets like our datasets, you can clean data as follow:
```
python data_clean.py
```
### 1.1.2 run models comparing and train predictor:
```
python main.py
```

## 2. How to predict the green odor of molecules:
```
python predictor.py -i example.smi -o result.csv
```
you can also go to webserver for a prediction at https://hwwlab.com/webserver/mlp4green
