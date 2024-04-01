import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def train_val_loss(model_history):
    tr_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]

    Epochs = [i+1 for i in range(len(tr_loss))]
    loss_label = f'best epoch= {str(index_loss + 1)}'

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')

    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout
    plt.show()



def mae_loss_epoch(model_history):
    mae = model_history.history['mean_absolute_error']
    tr_loss = model_history.history['loss']
    Epochs = [i+1 for i in range(len(tr_loss))]

    acc_loss_df = pd.DataFrame({"Mean Absolute error" : mae,
                            "Loss" : tr_loss,
                            "Epoch" : Epochs})

    acc_loss_df.style.bar(color = '#84A9AC',
                    subset = ['Mean Absolute error','Loss'])
    


def acc_loss(model_history):
    #------------------------- Grafik 1 Accuracy -------------------------

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['mean_squared_error'], color='b', label='Training Mean Squared Error')
    plt.plot(model_history.history['val_mean_squared_error'], color='r', label='Validation Mean Squared Error')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.title('Eğitim ve Test Ortalama Kare Hata Grafiği', fontsize=16)

    #------------------------- Grafik 2 Loss -------------------------

    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['loss'], color='b', label='Training Loss')
    plt.plot(model_history.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Eğitim ve Test Kayıp Grafiği', fontsize=16)
    plt.show()



def residual_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test.values - y_pred

    print("ytest",y_test.values)
    print("residual",residuals)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.values, residuals, color='blue')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()
