import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# (replace with your actual data)
X = [
    [5000, 3000, 35, 1, 5000],
    [6000, 4000, 40, 0, 10000]
]

y = np.array([1000, 2000])

models = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor())
]

ensembleModel = VotingRegressor(models)

X_df = pd.DataFrame(X, columns=['Income', 'Expenses', 'Age', 'Gender', 'Investments'])
if X_df.shape[0] != y.shape[0]:
    raise ValueError("Number of samples in X and y are not consistent.")


X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
ensembleModel.fit(X_train, y_train)

y_pred = ensembleModel.predict(X_test)

# Performance measuring metrics
#mse = mean_squared_error(y_test, y_pred)
#mae = mean_absolute_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)

#p redicted savings with the features compared to the original
print("Original Savings \t Predicted Savings \t Features")
for i in range(len(y_test)):
    print(f"{y_test[i]} \t\t\t {y_pred[i]} \t\t {', '.join([f'{X_df.columns[j]}: {X_test.iloc[i,j]}' for j in range(X_test.shape[1])])}")
