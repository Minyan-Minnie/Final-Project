import pandas as pd
import numpy as np
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Data from SQLite ---
conn = sqlite3.connect('adult_fairness.db')

# Load the reweighted training dataset from SQLite
data = pd.read_sql('SELECT * FROM train_reweighted', conn)
conn.close()

# Preview loaded data
print("Loaded data preview from SQLite:")
print(data.head())

# --- Demographic Parity Calculations ---

# Calculate demographic parity difference by gender
# Assuming 'sex' is encoded as 0 (Female) and 1 (Male)
male_pos_rate = data[data['sex'] == 1]['income'].mean()
female_pos_rate = data[data['sex'] == 0]['income'].mean()
dp_diff_gender = abs(male_pos_rate - female_pos_rate)
print(f"Demographic Parity Difference (Gender): {dp_diff_gender:.3f}")

# Calculate demographic parity difference by race if available
# Assuming one-hot encoding with column 'race_White' for privileged group
if 'race_White' in data.columns:
    white_pos_rate = data[data['race_White'] == 1]['income'].mean()
    non_white_pos_rate = data[data['race_White'] == 0]['income'].mean()
    dp_diff_race = abs(white_pos_rate - non_white_pos_rate)
    print(f"Demographic Parity Difference (Race): {dp_diff_race:.3f}")
else:
    print("Column 'race_White' not found in data. Skipping race-based parity difference.")

# --- Visualization: Income Distribution by Gender ---

# Map 'sex' column to human-readable labels for plotting
data['sex_label'] = data['sex'].apply(lambda x: 'Male' if x == 1 else 'Female')

# Plot income distribution stacked by gender
sns.histplot(data=data, x='income', hue='sex_label', multiple='stack', bins=2)
plt.title('Income Distribution by Gender')
plt.xlabel('Income (0 = <=50K, 1 = >50K)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("income_distribution.png")

print("Plot saved as 'income_distribution.png'")

# --- Visualization: Income Distribution by Race ---

# Extract race from one-hot encoded columns
race_columns = [col for col in data.columns if col.startswith('race_')]
def extract_race(row):
    for col in race_columns:
        if row[col] == 1:
            return col.replace('race_', '').replace('_', ' ')
    return 'Unknown'

data['race_label'] = data.apply(extract_race, axis=1)

# Plot income distribution stacked by race
sns.histplot(data=data, x='income', hue='race_label', multiple='stack', bins=2)
plt.title('Income Distribution by Race')
plt.xlabel('Income (0 = <=50K, 1 = >50K)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("income_distribution_race.png")

print("Plot saved as 'income_distribution_race.png'")
