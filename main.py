import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the data from the Excel sheet into a Pandas DataFrame
df = pd.read_excel('F1 table.xlsx')

# Step 2: Prepare the data for training the model
X = df.iloc[:, 1:-1]  # Input features (finishing time for previous races)
y = df.iloc[:, -1]    # Target variable (finishing time for the final race)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose a regression model and train it on the training data
model = RandomForestRegressor()  # You can replace this with any other regression model
model.fit(X_train, y_train)

# Step 5: Predict the finishing time for the final race using the trained model
final_race_predictions = model.predict(X_test)

# Step 6: Evaluate the performance of the model using mean squared error (MSE)
#mse = mean_squared_error(y_test, final_race_predictions)

# Optional: Retrain the model on the entire dataset and make predictions for the final race
#model.fit(X, y)
#X_final_race = ...  # Input features for the final race
#final_race_predictions = model.predict(X_final_race)

  