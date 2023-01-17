from uv_preprocessing_ import *
from torch.optim.lr_scheduler import StepLR
from torch import nn
from sklearn.metrics import r2_score

def test(model):
    #這裡要用測試集
    model.eval()
    Batch_size = 50
    test_dataset = load_test_dataloader(Batch_size)
    test_r2_score = []

    with torch.no_grad():
        for (X_test, y_true,index) in test_dataset:
        
            #seq = seq.to(device)
            y_pred = model(X_test)
            test_r2_score.append(r2_score(y_pred,y_true))

    avg_r2 = sum(test_r2_score) /len(test_r2_score)
    print(round(avg_r2,4))

    return round(avg_r2,5)