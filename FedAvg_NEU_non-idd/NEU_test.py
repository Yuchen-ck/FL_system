from NEU_preprocessing import *

def test_NUE(model):
    acc = 0
    model.eval()
    test_loader = load_test_datalaoder(val_bz=1)
    test_correct = []
    with torch.no_grad():
        for b, (X_test, gt) in enumerate(test_loader):
            # Limit the number of batches
            (X_test, gt) = (X_test.to('cuda'), gt.to('cuda'))
            # Apply the model
            y_val = model(X_test)

            # Tally the number of correct predictions
            predicted = torch.max(y_val.data, 1)[1] 
            
            # 設定batch size = 1 的時候，才能使用!!!
            if predicted[0] ==  gt.data[0] :
                acc += 1
        # print(acc)       

    # if i % 10 == 0 :
    #     torch.save(net.state_dict(), f'PNA{i}.pt')
    
    acc_2 = acc / 360
    test_correct.append(acc_2)
    print(acc_2)

    return test_correct[-1]
    