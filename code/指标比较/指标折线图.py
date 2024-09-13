# import matplotlib.pyplot as plt
#
# # Data
# methods = ['BiLSTM', 'without BiGRU','BiGRU']
# SN = [0.8482, 1,0.8513]
# SP = [0.8452, 0,0.85]
# MCC = [0.6933, float('nan'),0.701]
# ACC = [0.8467, 0.4858,0.8506]
# AUROC = [0.9238, 0.5935,0.9209]
#
# # Plotting
# plt.figure(figsize=(10, 6))
#
# plt.plot(methods, SN, marker='o', label='SN')
# plt.plot(methods, SP, marker='o', label='SP')
# plt.plot(methods, MCC, marker='o', label='MCC')
# plt.plot(methods, ACC, marker='o', label='ACC')
# plt.plot(methods, AUROC, marker='o', label='AUROC')
#
# plt.xlabel('Method')
# plt.ylabel('Scores')
# plt.title('Comparison of Methods')
# plt.legend()
#
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Data
zhibiao = ['SN','SP','MCC','ACC','AUC']
BiLSTM=[0.8482,0.8452,0.6933,0.8467,0.9238]
without_BiGRU = [1,0,np.nan,0.4858,0.5935]
BiGRU = [0.8513,0.85,0.701,0.8506,0.9209]


# Convert data to radians
angles = np.linspace(0, 2 * np.pi, len(zhibiao), endpoint=False).tolist()
angles += angles[:1]

# Create figure and axis
fig, ax = plt.subplots(figsize=(20, 20), subplot_kw={'polar': True})

# Plot data
ax.plot(angles, BiLSTM + [BiLSTM[0]], label='BiLSTM', color='y', linestyle='--', linewidth=2.0)
ax.plot(angles, without_BiGRU + without_BiGRU[:1], label='without_BiGRU', linewidth=1.0)
ax.plot(angles, BiGRU + BiGRU[:1], label='BiGRU', linewidth=1.0)

# Set labels and title
ax.set_xticks(angles[:-1])
ax.set_xticklabels(zhibiao, fontsize=12)
ax.set_yticklabels([])
ax.set_title('Comparison of Methods')

# Add legend
ax.legend()

# Display the plot
plt.show()








