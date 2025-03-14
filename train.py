import os
from comet_ml import Experiment
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize Comet.ml experiment
experiment = Experiment(
    api_key="qVaqH5DmdCc2UWO5If2SIm1bz",  # Replace with your Comet.ml API key
    project_name="ghi-forecasting",
    workspace=None  # This will use your default workspace
)

# Generate sample data (replace this with your actual data loading)
X = np.random.rand(100, 4)  # Sample features
y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, 100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log dataset parameters
experiment.log_parameters({
    "train_size": len(X_train),
    "test_size": len(X_test),
    "features": X.shape[1]
})

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics to Comet.ml
experiment.log_metrics({
    "mean_squared_error": mse,
    "r2_score": r2
})

# Log feature importances
feature_importance = dict(zip([f"feature_{i}" for i in range(X.shape[1])], 
                            model.feature_importances_))
experiment.log_parameters({"feature_importance": feature_importance})

print(f"Training completed!")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# End the experiment
experiment.end() 