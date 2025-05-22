import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import xgboost as xgb
import torch
import torch.nn as nn

# Load the dataset
data = pd.read_csv(r"C:\Users\Mi\Downloads\solar_energy_test_2024.csv")

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
data['DayOfYear'] = data['Date'].dt.dayofyear
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# Features and target variable
X_raw = data[['DayOfYear', 'Month', 'Year']]
y = data['Solar Energy (kWh)']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
X = pd.DataFrame(X_scaled, columns=X_raw.columns)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest-XGBoost Hybrid Model with Hyperparameter Tuning
rf_model = RandomForestRegressor(random_state=42)

# Hyperparameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Grid Search for Random Forest
grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)

# Best Random Forest model
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf_xgb = best_rf_model.predict(X_test)

# Evaluation for Random Forest-XGBoost Hybrid
rmse_rf_xgb = np.sqrt(mean_squared_error(y_test, y_pred_rf_xgb))
r2_rf_xgb = r2_score(y_test, y_pred_rf_xgb)
mae_rf_xgb = mean_absolute_error(y_test, y_pred_rf_xgb)

# Calculate accuracy percentage for Random Forest-XGBoost Hybrid
mean_actual = np.mean(y_test)
accuracy_percentage_rf_xgb = (1 - (mae_rf_xgb / mean_actual)) * 100 if mean_actual > 0 else 0

# Print results for Random Forest-XGBoost Hybrid
print(f"Random Forest-XGBoost Hybrid: RMSE = {rmse_rf_xgb:.4f}, R² = {r2_rf_xgb:.4f}, MAE = {mae_rf_xgb:.4f}, Accuracy (%) = {accuracy_percentage_rf_xgb:.2f}")

# KMeans-XGBoost Hybrid Model with Hyperparameter Tuning
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# Add cluster labels to the training data
X_train['Cluster'] = kmeans.labels_
X_test['Cluster'] = kmeans.predict(X_test)

# Hyperparameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Grid Search for XGBoost
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='neg_mean_squared_error')
grid_search_xgb.fit(X_train, y_train)

# Best XGBoost model
best_xgb_model = grid_search_xgb.best_estimator_
y_pred_kmeans_xgb = best_xgb_model.predict(X_test)

# Evaluation for KMeans-XGBoost Hybrid
rmse_kmeans_xgb = np.sqrt(mean_squared_error(y_test, y_pred_kmeans_xgb))
r2_kmeans_xgb = r2_score(y_test, y_pred_kmeans_xgb)
mae_kmeans_xgb = mean_absolute_error(y_test, y_pred_kmeans_xgb)

# Calculate accuracy percentage for KMeans-XGBoost Hybrid
accuracy_percentage_kmeans_xgb = (1 - (mae_kmeans_xgb / mean_actual)) * 100 if mean_actual > 0 else 0

# Print results for KMeans-XGBoost Hybrid
print(f"KMeans-XGBoost Hybrid: RMSE = {rmse_kmeans_xgb:.4f}, R² = {r2_kmeans_xgb:.4f}, MAE = {mae_kmeans_xgb:.4f}, Accuracy (%) = {accuracy_percentage_kmeans_xgb:.2f}")

# CNN-XGBoost Hybrid Model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return torch.exp(x)

# Hyperparameters for CNN-LSTM
input_size = 3
hidden_size = 50
num_epochs = 100
learning_rate = 0.001

# Prepare data for CNN-LSTM
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Convert data to tensors for CNN-LSTM
X_train_tensor = torch.FloatTensor(X_train_reshaped)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_reshaped)

# Initialize and train the CNN-LSTM model
cnn_xgb_model = CNNLSTMModel(input_size=input_size, hidden_size=hidden_size)
cnn_xgb_model.eval()
with torch.no_grad():
    y_pred_cnn_xgb = cnn_xgb_model(X_test_tensor.permute(0, 2, 1)).numpy()

# Train XGBoost on CNN features
y_pred_cnn_xgb = np.clip(y_pred_cnn_xgb, a_min=0, a_max=None)
xgb_cnn_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_cnn_model.fit(X_train, y_train)

# Predictions from CNN-XGBoost
y_pred_cnn_xgb_final = xgb_cnn_model.predict(X_test)

# Evaluation for CNN-XGBoost Hybrid
rmse_cnn_xgb = np.sqrt(mean_squared_error(y_test, y_pred_cnn_xgb_final))
r2_cnn_xgb = r2_score(y_test, y_pred_cnn_xgb_final)
mae_cnn_xgb = mean_absolute_error(y_test, y_pred_cnn_xgb_final)

# Calculate accuracy percentage for CNN-XGBoost Hybrid
accuracy_percentage_cnn_xgb = (1 - (mae_cnn_xgb / mean_actual)) * 100 if mean_actual > 0 else 0

# Print results for CNN-XGBoost Hybrid
print(f"CNN-XGBoost Hybrid: RMSE = {rmse_cnn_xgb:.4f}, R² = {r2_cnn_xgb:.4f}, MAE = {mae_cnn_xgb:.4f}, Accuracy (%) = {accuracy_percentage_cnn_xgb:.2f}")

# Prepare DataFrame for all predictions
predictions_all_df = pd.DataFrame({
    'Date': data['Date'].iloc[X_test.index],  # Assuming the original dates are available
    'Random Forest-XGBoost Hybrid': y_pred_rf_xgb,
    'KMeans-XGBoost Hybrid': y_pred_kmeans_xgb,
    'CNN-XGBoost Hybrid': y_pred_cnn_xgb_final
})

# Save predictions to CSV
predictions_all_df.to_csv(r"C:\Users\Mi\Downloads\predicted_values.csv", index=False)
print("Predicted values saved to 'predicted_values.csv'")

# Save results to CSV
results_df = pd.DataFrame({
    'Model': ['Random Forest-XGBoost Hybrid', 'KMeans-XGBoost Hybrid', 'CNN-XGBoost Hybrid'],
    'RMSE': [rmse_rf_xgb, rmse_kmeans_xgb, rmse_cnn_xgb],
    'R²': [r2_rf_xgb, r2_kmeans_xgb, r2_cnn_xgb],
    'MAE': [mae_rf_xgb, mae_kmeans_xgb, mae_cnn_xgb],
    'Accuracy (%)': [accuracy_percentage_rf_xgb, accuracy_percentage_kmeans_xgb, accuracy_percentage_cnn_xgb]
})

results_df.to_csv(r"C:\Users\Mi\Downloads\model_results.csv", index=False)
print(results_df)

# Prepare DataFrame for all predictions
predictions_all_df = pd.DataFrame({
    'Date': data['Date'].iloc[X_test.index],  # Assuming the original dates are available
    'Random Forest-XGBoost Hybrid': y_pred_rf_xgb,
    'KMeans-XGBoost Hybrid': y_pred_kmeans_xgb,
    'CNN-XGBoost Hybrid': y_pred_cnn_xgb_final
})

# Plotting Random Forest-XGBoost Hybrid Predictions
plt.figure(figsize=(14, 7))
plt.plot(predictions_all_df['Date'], predictions_all_df['Random Forest-XGBoost Hybrid'], label='Random Forest-XGBoost Hybrid', color='blue')
plt.title('Predicted Solar Energy (kWh) - Random Forest-XGBoost Hybrid')
plt.xlabel('Date')
plt.ylabel('Solar Energy (kWh)')
plt.legend()
plt.grid()
plt.savefig('random_forest_xgboost_predictions.png')
plt.show()

# Plotting KMeans-XGBoost Hybrid Predictions
plt.figure(figsize=(14, 7))
plt.plot(predictions_all_df['Date'], predictions_all_df['KMeans-XGBoost Hybrid'], label='KMeans-XGBoost Hybrid', color='green')
plt.title('Predicted Solar Energy (kWh) - KMeans-XGBoost Hybrid')
plt.xlabel('Date')
plt.ylabel('Solar Energy (kWh)')
plt.legend()
plt.grid()
plt.savefig('kmeans_xgboost_predictions.png')
plt.show()

# Plotting CNN-XGBoost Hybrid Predictions
plt.figure(figsize=(14, 7))
plt.plot(predictions_all_df['Date'], predictions_all_df['CNN-XGBoost Hybrid'], label='CNN-XGBoost Hybrid', color='red')
plt.title('Predicted Solar Energy (kWh) - CNN-XGBoost Hybrid')
plt.xlabel('Date')
plt.ylabel('Solar Energy (kWh)')
plt.legend()
plt.grid()
plt.savefig('cnn_xgboost_predictions.png')
plt.show()

# Create a DataFrame for the metrics excluding accuracy
metrics_df = pd.DataFrame({
    'RMSE': [results_df['RMSE'].iloc[0], results_df['RMSE'].iloc[1], results_df['RMSE'].iloc[2]],
    'R²': [results_df['R²'].iloc[0], results_df['R²'].iloc[1], results_df['R²'].iloc[2]]
}, index=['Random Forest-XGBoost Hybrid', 'KMeans-XGBoost Hybrid', 'CNN-XGBoost Hybrid'])

# Plotting the bar graph without accuracy
plt.figure(figsize=(10, 6))
metrics_df.plot(kind='bar', color=['red', 'green'], legend=False)

# Adding labels and title
plt.ylabel('Metrics')
plt.title('Model Comparison: RMSE and R²')
plt.xticks(rotation=0)
plt.grid(axis='y')

# Adding a legend
plt.legend(['RMSE', 'R²'], loc='upper right')

# Save the figure
plt.tight_layout()
plt.savefig('model_comparison_bar_plot_rmse_r2.png')
plt.show()
