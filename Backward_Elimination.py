import numpy as np
import pandas as pd
from scipy import stats as sc
from sklearn.linear_model import LinearRegression

def calcFullModel(df):
    Y=df.iloc[:,0].values
    X=df.iloc[:,1:].values
    model = LinearRegression()
    model.fit(X, Y)
    y_pred = model.predict(X)
    SSR_full = np.sum((np.mean(Y) - y_pred)**2)
    SSE_full = np.sum((Y - y_pred)**2)
    S_full=SSE_full/(len(Y)-X.shape[1]-1)
    print(f"SSR_full: {SSR_full:.4f} , S_full: {S_full:.4f}\n");
    return SSR_full, S_full

def calcReducedModel(cols,df,SSR_full):
    SSR_red={}
    SSR_Cond={}
    for col in cols:
        df1=df.drop(col,axis=1)
        Y=df1.iloc[:,0].values
        X=df1.iloc[:,1:].values
        model = LinearRegression()
        model.fit(X, Y)
        y_pred = model.predict(X)
        SSR_red[col] = np.sum((y_pred - np.mean(Y))**2)
        SSR_Cond[col] = (SSR_full - SSR_red[col]);
        
    for col in cols:
        print(f"{col} -> SSR_red: {SSR_red[col]:.4f} , SSR_Cond: {SSR_Cond[col]:.4f}\n")   

    minCol = min(SSR_Cond, key=SSR_Cond.get)
    SSR_min = SSR_Cond[minCol]
    return SSR_red, minCol, SSR_min

def HypothesisTest(df):
    SSR_full, S_full=calcFullModel(df)
    cols=df.columns[1:]
    SSR_red, minCol, SSR_min=calcReducedModel(cols,df,SSR_full)
    F_Calc = SSR_min / S_full
    v2 = len(df) - len(cols) - 1
    F_Crit = sc.f.ppf(0.95, 1, v2)
    if F_Calc > F_Crit:
        print(f"F_Calc: {F_Calc:.4f} is greater than F_Crit: {F_Crit:.4f}.")
        print(f"The variable {minCol} is significant. Keeping it in the model.\n")
        return 0, minCol
    else:
        print(f"F_Calc: {F_Calc:.4f} is not greater than F_Crit: {F_Crit:.4f}.")
        print(f"The variable {minCol} is not significant. Removing it from the model.")
        return 1, minCol

def processRun():
    print("Model Started")
    fileName = input("Enter the file name: ")
    df= pd.read_csv(fileName,header=None);
    cols=df.columns[1:]
    len1= len(cols)
    df2=df;
    for i in range(len1):
        print(f"\nStage {i+1}:")
        result, minCol = HypothesisTest(df2)
        if result == 1:
            df2 = df2.drop(minCol, axis=1)
        else:
            print("No more variables to remove. Ending the process.")
            print(f"Final model includes the variables: {df2.columns[1:].tolist()}\n")
            return

processRun()