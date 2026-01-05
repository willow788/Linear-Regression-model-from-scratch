model = LinearRegression(learn_rate=0.0001, iter=1000)
model.fit(X_b, y)

predictions = model.predict(X_b)

mse = np.mean((predictions - y) ** 2)
print("Mean Squared Error:", mse)

rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)


mae = np.mean(np.abs(predictions - y))
print("Mean Absolute Error:", mae)

ss_res = np.sum((y - predictions) ** 2)
sst_tot = np.sum((y - np.mean(y))** 2 )
r2_score = 1 - (ss_res / sst_tot)
print("R^2 Score:", r2_score)
