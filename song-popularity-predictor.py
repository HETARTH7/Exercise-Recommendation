from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('spotify_songs.csv')
dataset.describe()

X = dataset.iloc[:, 11:].values
y = dataset.iloc[:, 3].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(X_train, y_train)

random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(X_train, y_train)

svr = SVR()
svr.fit(X_train, y_train)

y_pred_linear_reg = linear_reg.predict(X_test)
y_pred_decision_tree_reg = decision_tree_reg.predict(X_test)
y_pred_random_forest_reg = random_forest_reg.predict(X_test)
y_pred_svr = svr.predict(X_test)


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Metrics:")
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R sq Score:", r2)
    print()


evaluate_model(linear_reg, X_test, y_test, "Linear Regression")

evaluate_model(decision_tree_reg, X_test, y_test, "Decision Tree Regression")

evaluate_model(random_forest_reg, X_test, y_test, "Random Forest Regression")

evaluate_model(svr, X_test, y_test, "SVR")
