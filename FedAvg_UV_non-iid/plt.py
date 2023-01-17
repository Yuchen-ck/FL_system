import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_losses =[0.68591, 0.95148, 0.93362, 0.94388, 0.96555, 0.96195, 0.96581, 0.93192, 0.96611, 0.96518, 0.96377, 0.96617, 0.96563, 0.96026, 0.96493, 0.96439, 0.96512, 0.96503, 0.96472, 0.95635, 0.96584, 0.9666, 0.96427, 0.96526, 0.96597, 0.96502, 0.96326, 0.96519, 0.96448, 0.96354]
    y =  list(range(len(train_losses)))
    fig = plt.figure()
    plt.plot(y, train_losses,label='training loss')

    plt.title('Non-IID_1 without attack')
    plt.xlabel('communication rounds')
    plt.ylabel('R2_score')
    plt.legend()
    
    #限制刻度的範圍
    plt.ylim(0.2,1)

    plt.savefig('plot_with_cround.png')
    plt.show()