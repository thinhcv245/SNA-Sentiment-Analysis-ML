import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the Sentiment140 dataset
data_path = "sentiment140.csv"  # Update with the correct path
df = pd.read_csv(data_path, encoding="latin1", header=None)
df.columns = ["target", "id", "date", "flag", "user", "text"]

# Map target values to emotions (0 = negative, 4 = positive)
df["sentiment"] = df["target"].map({0: "negative", 4: "positive"})

# Calculate initial positive ratio
positive_tweets = df[df["sentiment"] == "positive"]
initial_positive_ratio = len(positive_tweets) / len(df)

# Define parameters
alpha_values = [0.1, 0.3, 0.5, 0.7, 0.9]
P_infinity = 1.0  # Maximum proportion of positive emotions
time_steps = 100
time = np.linspace(0, 10, time_steps)

# Function to compute positive emotion proportion
def positive_emotion_proportion(alpha, t, P_infinity):
    return P_infinity * (1 - np.exp(-alpha * t))

# Plot results
plt.figure(figsize=(8, 6))

for alpha in alpha_values:
    P_t = positive_emotion_proportion(alpha, time, P_infinity)
    plt.plot(time, P_t, label=f'alpha = {alpha}')

# Customize plot
plt.title(f"Emotion Propagation with Initial Positive Ratio = {initial_positive_ratio:.2f}")
plt.xlabel("Time")
plt.ylabel("Proportion of Positive Emotions")
plt.legend()
plt.grid()
plt.show()
