from statsmodels.tsa.seasonal import seasonal_decompose

def outliers_with_quantile (X, threshold):
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    Xout = X[((X < (Q1 - 1.5 * IQR)) |(X > (Q3 + 1.5 * IQR)))]
    return  Xout


def get_stl_outliers (data, threshold, period):
    result = seasonal_decompose(data, model='additive', extrapolate_trend='freq', period=period)
    resid = result.resid

    Xout = resid [ abs(resid)  > threshold ]
    data_out = data.iloc[Xout.index]
    return data_out







