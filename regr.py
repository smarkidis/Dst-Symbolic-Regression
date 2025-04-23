import pandas as pd
import numpy as np
import random
import os
from pysr import PySRRegressor

# Define ANSI escape codes for red bold text.
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"

# -------------------------------
# Data Preprocessing
# -------------------------------

# Load dataset
file_path = "Cleaned_NASA_OMNI_Dataset.csv"
df = pd.read_csv(file_path)

df["DATE"] = pd.to_datetime(df["DATE"])
df = df[(df["DATE"] >= "1995-01-01") & (df["DATE"] <= "2021-5-31")]

# List the columns you want to ensure have no NaN values
columns_to_fill = [
    "Dst-index, nT",           # DST values
    "SW Plasma Speed, km/s",   # Solar wind speed
    "BZ, nT (GSM)",            # Magnetic field component Bz
    "SW Proton Density, N/cm^3",  # Proton density
    "Vector B Magnitude,nT"    # Magnetic field magnitude
    # Add additional columns such as Temperature if needed
]

# Fill missing values by first interpolating, then forward/backward filling any edge issues
for col in columns_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear')
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

# -------------------------------
# Compute Derivative and Derived Quantities
# -------------------------------

# Calculate dDst/dt using a central difference; this produces NaNs at the boundaries.
df["dDst_dt"] = (df["Dst-index, nT"].shift(-1) - df["Dst-index, nT"].shift(1)) / 2

# Replace the boundary NaN values with one-sided differences.
df.loc[df.index[0], "dDst_dt"] = df["Dst-index, nT"].iloc[1] - df["Dst-index, nT"].iloc[0]
df.loc[df.index[-1], "dDst_dt"] = df["Dst-index, nT"].iloc[-1] - df["Dst-index, nT"].iloc[-2]

# Compute additional derived quantities
df["Ey"] = -df["SW Plasma Speed, km/s"] * df["BZ, nT (GSM)"] * 1e-3  # Convective electric field in mV/m
df["P_dyn"] = 1.6726e-6 * df["SW Proton Density, N/cm^3"] * (df["SW Plasma Speed, km/s"] ** 2)  # Dynamic pressure

# Compute magnetic pressure using mu_0 (Permeability of free space)
mu_0 = 4 * np.pi * 1e-7  
df["P_B"] = (df["Vector B Magnitude,nT"]**2) / (2 * mu_0)

# -------------------------------
# Generate Lagged Features
# -------------------------------

variables = ["Dst-index, nT", "Ey", "P_dyn", "P_B"]
n_lags = 1  # You can adjust the number of lags as needed

for var in variables:
    for lag in range(1, n_lags + 1):
        df[f"{var}_lag{lag}"] = df[var].shift(lag)

# Fill any NaNs introduced by lagging.
df = df.fillna(method='ffill').fillna(method='bfill')

# Define feature matrix and target
features = [f"{var}_lag{lag}" for var in variables for lag in range(1, n_lags + 1)]
target = "dDst_dt"

X = df[features].to_numpy()
y = df[target].to_numpy()

# -------------------------------
# Simulation Loop and Global Results Update
# -------------------------------

# Global file to store combined simulation results.
results_file = "equations_ranked.csv"

# If the file exists, load it; otherwise, start with an empty DataFrame.
if os.path.exists(results_file):
    global_results = pd.read_csv(results_file)
else:
    global_results = pd.DataFrame()

# Run 100 simulations in series.
for sim in range(1, 101):
    print(f"{RED_BOLD}Running simulation {sim}...{RESET}", flush=True)
    
    # Randomly choose parsimony and populations values.
    parsimony_value = random.uniform(0.0, 0.9)
    populations_value = random.randint(20, 120)
    print(f"{RED_BOLD}  Parsimony: {parsimony_value:.3f}, Populations: {populations_value} {RESET}", flush=True)
    
    # Create the PySR model with the randomized parameters.
    model = PySRRegressor(
        niterations=1000,
        populations=populations_value,
        binary_operators=["+", "-", "*", "/", "greater", "max", "min"],
        unary_operators=["sqrt", "square", "sign"],
        elementwise_loss="L1DistLoss()",
        parsimony=parsimony_value,
        batching=True,
        batch_size=50,  # You can adjust this value as needed.
        denoise=False,
        progress=True,  # Set to True if you want to see progress.
        maxsize=50,
        timeout_in_seconds=3600
    )
    
    # Fit the model on your training data.
    model.fit(X, y, variable_names=["DST", "Ey", "P_dyn", "P_B"])
    
    # Extract the discovered equations as a DataFrame.
    eq_df = model.equations_.copy()
    
    # Filter to include only equations with a maximum complexity of 18.
    eq_df = eq_df[eq_df['complexity'] <= 18].copy()
    
    # Record the simulation parameters and simulation number.
    eq_df['simulation'] = sim
    eq_df['parsimony'] = parsimony_value
    eq_df['populations'] = populations_value
    
    # Append the new simulation results to the global results.
    global_results = pd.concat([global_results, eq_df], ignore_index=True)
    
    # Optional: Remove duplicate equations (if the same equation appears in different simulations).
    if 'equation' in global_results.columns:
        global_results = global_results.drop_duplicates(subset=["equation"], keep='first')
    
    # Rank equations by loss (assuming lower loss is better).
    global_results = global_results.sort_values(by='loss', ascending=True)
    
    # Write the updated results to file.
    global_results.to_csv(results_file, index=False)
    
    print(f"{RED_BOLD} Simulation {sim} complete. Global results updated in '{results_file}'. {RESET} \n", flush=True)
