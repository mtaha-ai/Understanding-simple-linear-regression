from linear_model import LinearRegressor
import matplotlib.pyplot as plt
from utils import Scaler
import pandas as pd

def plot(X, y, y_pred):
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, y_pred, color='red', label='Best Fit Line')
    plt.xlabel('Area in sqft')
    plt.ylabel('Price')
    plt.title('Best Fit Line')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    df = pd.read_csv('data.csv')
    X = df['Area_sqft'].values
    y = df['Price'].values
    X = X.reshape(-1, 1)
    scaler = Scaler()
    X_scaled = scaler.MinMaxScaler(X)
    y_scaled = scaler.MinMaxScaler(y)
    model = LinearRegressor()
    model.fit(X_scaled, y_scaled)

    y_pred = model.predict(X_scaled)
    plot(X_scaled, y_scaled, y_pred)

main()