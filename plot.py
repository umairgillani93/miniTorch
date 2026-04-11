import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("loss.csv")

plt.figure(figsize=(10,6))
plt.plot(df["step"], df["loss"])
plt.title("Training Loss — Transformer Built in Pure C")
plt.xlabel("Training Steps")
plt.ylabel("MSE Loss")
plt.grid()

plt.savefig("training_curve.png", dpi=300)
plt.show()
