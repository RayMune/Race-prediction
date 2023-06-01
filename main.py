import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Read the Excel sheet into Python.
df = pd.read_excel("f1_data.xlsx")

# Create a new column for the predicted finishing position.
df["predicted_finishing_position"] = np.nan

# Loop over each driver.
for driver in df["driver"].unique():

    # Get the data for the current driver.
    driver_data = df[df["driver"] == driver]

    # Fit a linear regression model to the data.
    model = LinearRegression()
    model.fit(driver_data[["p1", "p2", "p3", "quali"]], driver_data["final"])

    # Predict the finishing position for the current driver.
    predicted_finishing_position = model.predict([[driver, driver, driver, driver]])[0]

    # Set the predicted finishing position in the DataFrame.
    df.loc[df["driver"] == driver, "predicted_finishing_position"] = predicted_finishing_position

# Print the predicted finishing positions.
print(df["predicted_finishing_position"])
  