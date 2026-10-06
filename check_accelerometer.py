import pandas as pd

df = pd.read_pickle("trials.pkl")

acc = df[df["sensor"] == "Accelerometer"].copy()

print("All trials:", len(df))
print("Accelerometer trials:", len(acc))

print("\nLeak types:")
print(acc["leak_type"].value_counts())

print("\nNetworks:")
print(acc["network"].value_counts())

print("\nChannels:")
print(acc["channel"].value_counts())

print("\nSample rates:")
print(acc["sample_rate_hz"].value_counts())