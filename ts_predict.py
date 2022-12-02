from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

def arima_predict (S_train, S_test,lag):
    
    history = list(S_train.values) 

    predictions = list()

    date_future = pd.date_range(start=S_test.index[-1], periods=S_test.shape[0]) 
    for t in range(S_test.shape[0]):
        model = ARIMA(history, order=(lag,2,1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = S_test.iloc[t]
        history.append(obs)
     
    new_index = list(S_test.index)+list(date_future) 

    for t in range(S_test.shape[0]):
        model = ARIMA(history, order=(lag,2,1))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        history.append (yhat)

    P = pd.Series (index=new_index,data=predictions)    

    return P

