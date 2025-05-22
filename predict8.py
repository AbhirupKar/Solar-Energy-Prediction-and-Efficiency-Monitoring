import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
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

# Initialize models with tuned SVM
models = {
    'Random Forest': RandomForestRegressor(),
    'SVM': SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale'),
    'KNN': KNeighborsRegressor(n_neighbors=5),
}

# Prepare data for CNN and GRU
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train non-CNN models
results = {}
accuracy_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Ensure positive predictions
    y_pred = np.clip(y_pred, a_min=0, a_max=None)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mean_actual = np.mean(y_test)

    # Ensure R² is non-negative
    if r2 < 0:
        r2 = 0  # Set R² to 0 if it is negative

    accuracy_percentage = (1 - (mae / mean_actual)) * 100 if mean_actual > 0 else 0

    results[name] = {'RMSE': rmse, 'R²': r2, 'MAE': mae}
    accuracy_results[name] = {'Accuracy (%)': accuracy_percentage}

# Define ANN Model
class ANNModel(nn.Module):
    def __init__(self, input_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # Hidden layer with 50 neurons
        self.fc2 = nn.Linear(50, 1)            # Output layer

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return torch.exp(x)  # Ensures positive output

# Initialize and train the ANN model
ann_model = ANNModel(input_size=X_train.shape[1])
criterion_ann = nn.MSELoss()
optimizer_ann = torch.optim.Adam(ann_model.parameters(), lr=0.001)

# Convert data to tensors for ANN
X_train_tensor_ann = torch.FloatTensor(X_train.values)
y_train_tensor_ann = torch.FloatTensor(y_train.values).view(-1, 1)
X_test_tensor_ann = torch.FloatTensor(X_test.values)

# ANN Training
num_epochs = 100
for epoch in range(num_epochs):
    ann_model.train()
    optimizer_ann.zero_grad()
    outputs = ann_model(X_train_tensor_ann)
    loss = criterion_ann(outputs, y_train_tensor_ann)
    loss.backward()
    optimizer_ann.step()

# ANN Prediction
ann_model.eval()
with torch.no_grad():
    y_pred_ann = ann_model(X_test_tensor_ann)

# Ensure positive predictions
y_pred_ann = y_pred_ann.numpy()
y_pred_ann = np.clip(y_pred_ann, a_min=0, a_max=None)

# ANN Evaluation
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
r2_ann = r2_score(y_test, y_pred_ann)
mae_ann = mean_absolute_error(y_test, y_pred_ann)

# Ensure R² is non-negative
if r2_ann < 0:
    r2_ann = 0  # Set R² to 0 if it is negative

accuracy_percentage_ann = (1 - (mae_ann / mean_actual)) * 100 if mean_actual > 0 else 0

results['ANN'] = {'RMSE': rmse_ann, 'R²': r2_ann, 'MAE': mae_ann}
accuracy_results['ANN'] = {'Accuracy (%)': accuracy_percentage_ann}

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
        conv_output_size = input_size - 2  # After two conv layers
        self.fc1 = nn.Linear(32 * conv_output_size, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return torch.exp(x)  # Ensures positive output

# Define GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return torch.exp(out)  # Ensures positive output

# Hyperparameters for CNN and GRU
input_size = 3
hidden_size = 50
num_epochs = 100
learning_rate = 0.001

# Convert data to tensors for CNN and GRU
X_train_tensor = torch.FloatTensor(X_train_reshaped)
y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_reshaped)

# CNN Training
cnn_model = CNNModel(input_size)
criterion = nn.MSELoss()
optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    cnn_model.train()
    optimizer_cnn.zero_grad()
    cnn_input = X_train_tensor.permute(0, 2, 1)  # [batch, channels=1, sequence_length]
    outputs = cnn_model(cnn_input)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer_cnn.step()

# CNN Prediction
cnn_model.eval()
with torch.no_grad():
    cnn_input_test = X_test_tensor.permute(0, 2, 1)
    y_pred_cnn = cnn_model(cnn_input_test)

# Ensure positive predictions
y_pred_cnn = y_pred_cnn.numpy()
y_pred_cnn = np.clip(y_pred_cnn, a_min=0, a_max=None)

# CNN Evaluation
rmse_cnn = np.sqrt(mean_squared_error(y_test, y_pred_cnn))
r2_cnn = r2_score(y_test, y_pred_cnn)
mae_cnn = mean_absolute_error(y_test, y_pred_cnn)

# Ensure R² is non-negative
if r2_cnn < 0:
    r2_cnn = 0  # Set R² to 0 if it is negative

accuracy_percentage_cnn = (1 - (mae_cnn / mean_actual)) * 100 if mean_actual > 0 else 0

results['CNN'] = {'RMSE': rmse_cnn, 'R²': r2_cnn, 'MAE': mae_cnn}
accuracy_results['CNN'] = {'Accuracy (%)': accuracy_percentage_cnn}

# GRU Training
gru_model = GRUModel(input_size=1, hidden_size=hidden_size)
optimizer_gru = torch.optim.Adam(gru_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    gru_model.train()
    optimizer_gru.zero_grad()
    outputs = gru_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer_gru.step()

# GRU Prediction
gru_model.eval()
with torch.no_grad():
    y_pred_gru = gru_model(X_test_tensor)

# Ensure positive predictions
y_pred_gru = y_pred_gru.numpy()
y_pred_gru = np.clip(y_pred_gru, a_min=0, a_max=None)

# GRU Evaluation
rmse_gru = np.sqrt(mean_squared_error(y_test, y_pred_gru))
r2_gru = r2_score(y_test, y_pred_gru)
mae_gru = mean_absolute_error(y_test, y_pred_gru)

# Ensure R² is non-negative
if r2_gru < 0:
    r2_gru = 0  # Set R² to 0 if it is negative

accuracy_percentage_gru = (1 - (mae_gru / mean_actual)) * 100 if mean_actual > 0 else 0

results['GRU'] = {'RMSE': rmse_gru, 'R²': r2_gru, 'MAE': mae_gru}
accuracy_results['GRU'] = {'Accuracy (%)': accuracy_percentage_gru}

# Print results
for model_name, metrics in results.items():
    accuracy = accuracy_results[model_name]['Accuracy (%)']
    print(f"{model_name}: RMSE = {metrics['RMSE']:.4f}, R² = {metrics['R²']:.4f}, MAE = {metrics['MAE']:.4f}, Accuracy (%) = {accuracy:.2f}")

# Save results to CSV
results_df = pd.DataFrame(results).T
accuracy_df = pd.DataFrame(accuracy_results).T
final_results_df = pd.concat([results_df, accuracy_df], axis=1)
final_results_df.to_csv(r"C:\Users\Mi\Downloads\model_results_with_accuracy1.csv", index=True)
print(final_results_df)

# Future Predictions (2025 - 2027)
future_dates = pd.date_range(start='2025-01-01', end='2027-12-31', freq='D')
future_data = pd.DataFrame({
    'Date': future_dates,
    'DayOfYear': future_dates.dayofyear,
    'Month': future_dates.month,
    'Year': future_dates.year
})

# Scale future data
future_features_scaled = scaler.transform(future_data[['DayOfYear', 'Month', 'Year']])
future_data_scaled = pd.DataFrame(future_features_scaled, columns=['DayOfYear', 'Month', 'Year'])

# Predictions from classical models
predictions = {}
for name, model in models.items():
    future_pred = model.predict(future_data_scaled[['DayOfYear', 'Month', 'Year']])
    future_pred = np.clip(future_pred, a_min=0, a_max=None)  # Ensure positive predictions
    predictions[name] = future_pred.flatten()

# CNN Predictions
future_data_reshaped = future_data_scaled.values.reshape((future_data_scaled.shape[0], 3, 1))
future_data_tensor = torch.FloatTensor(future_data_reshaped)

cnn_model.eval()
with torch.no_grad():
    future_input_cnn = future_data_tensor.permute(0, 2, 1)
    future_pred_cnn = cnn_model(future_input_cnn)

predictions['CNN'] = np.clip(future_pred_cnn.numpy().flatten(), a_min=0, a_max=None)  # Ensure positive predictions

# GRU Predictions
gru_model.eval()
with torch.no_grad():
    future_pred_gru = gru_model(future_data_tensor)

predictions['GRU'] = np.clip(future_pred_gru.numpy().flatten(), a_min=0, a_max=None)  # Ensure positive predictions

# ANN Predictions
ann_model.eval()
with torch.no_grad():
    future_data_tensor_ann = torch.FloatTensor(future_data_scaled.values)
    future_pred_ann = ann_model(future_data_tensor_ann)

predictions['ANN'] = np.clip(future_pred_ann.numpy().flatten(), a_min=0, a_max=None)  # Ensure positive predictions

# Save Future Predictions
predictions_df = pd.DataFrame(predictions, index=future_dates)
predictions_df['Date'] = future_dates
predictions_df = predictions_df.reset_index(drop=True)
predictions_df.to_csv(r"C:\Users\Mi\Downloads\model_comparison_predictions_2025.csv", index=False)

# Plotting predictions
plt.figure(figsize=(14, 7))
for name in predictions:
    plt.plot(predictions_df['Date'], predictions_df[name], label=name)
plt.title('Predicted Solar Energy (kWh) for 2025 and beyond')
plt.xlabel('Date')
plt.ylabel('Solar Energy (kWh)')
plt.legend()
plt.grid()
plt.savefig('predictions_plot_2025.png')
plt.show()

# Plot RMSE and R²
metrics_df = pd.DataFrame({
    'RMSE': [metrics['RMSE'] for metrics in results.values() if 'RMSE' in metrics],
    'R²': [metrics['R²'] for metrics in results.values() if 'R²' in metrics]
}, index=[name for name in results.keys() if 'RMSE' in results[name]])

plt.figure(figsize=(12, 8))
plt.plot(metrics_df.index, metrics_df['RMSE'], marker='o', color='r', label='RMSE', linewidth=2)
plt.plot(metrics_df.index, metrics_df['R²'], marker='s', color='g', label='R²', linewidth=2)
plt.ylabel('Metrics')
plt.title('Model Comparison: RMSE and R²')
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison_plot_rmse_r2.png')
plt.show()
