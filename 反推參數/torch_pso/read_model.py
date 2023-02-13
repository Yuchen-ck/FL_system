from uv_model import *
import math
import torch
import pandas as pd

model =  small_ANN()
load_model_path = './2023-02-11_0.92114.pt'
model.load_state_dict(torch.load(load_model_path))

def change_to_tensor(x):

    x1 = x[0]
    x2 = x[1]
    x3 = x[2]

    scaled_x1 = min_max_scaler_x1(x1) 

    scaled_x2 = min_max_scaler_x2(x2) 

    scaled_x3 = min_max_scaler_x3(x3) 

    df1 = pd.DataFrame([[scaled_x1,scaled_x2,scaled_x3]])
    Test_data = df1.values
    predict_tensor = torch.from_numpy(Test_data).float()

    return predict_tensor

def min_max_scaler_x1(data, min_value =1.26047156 , max_value=7.99761224):
    scaled_data = (data - min_value) / (max_value - min_value)
    return scaled_data

def min_max_scaler_x2(data, min_value = 150 , max_value = 300):
    scaled_data = (data - min_value) / (max_value - min_value)
    return scaled_data

def min_max_scaler_x3(data, min_value = 50 , max_value = 70):
    scaled_data = (data - min_value) / (max_value - min_value)
    return scaled_data

def torch_model_function(x):
    predict_tensor = change_to_tensor(x)
    result = model(predict_tensor)
    result = result.detach().numpy()
    y = result[0][0] 
    y = float(str(y))
    y_func =  math.ceil(y)
    
    return y_func
