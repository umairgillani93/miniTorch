import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("loss.csv", names = ['epochs', 'loss'])
except FileNotFoundError:
    print("Error: loss.csv not found. Run your C program first to generate data!")
    exit()

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

ax.plot(df['epochs'], df['loss'], color='#1f77b4', linewidth=2, label='MSE Loss')

ax.set_title("miniTorch: Custom C Autograd Engine Convergence", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Training Steps (Batches)", fontsize=12, labelpad=10)
ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12, labelpad=10)


ax.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("minitorch_loss_curve.png", bbox_inches='tight')
print("Success! 'minitorch_loss_curve.png' has been saved successfully.")
