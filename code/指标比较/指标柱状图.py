import matplotlib.pyplot as plt

#四种编码组合
# x_x = ['SN','SP','MCC','ACC','AUC']
# onehot_BLOSUM62_Binary1_Binary2 = [0.8603,0.8262,0.6863,0.8428,0.9197]
# BLOSUM62_Zscale_Binary1_Binary2 = [0.8422,0.8500,0.6921,0.8462,0.9172]
# onehot_Zscale_Binary1_Binary2 = [0.8412,0.8310,0.6719,0.8359,0.9138]
# onehot_BLOSUM62_Zscale_Binary1 = [0.8422,0.8329,0.6748,0.8374,0.9164]
# onehot_BLOSUM62_Zscale_Binary2 = [0.8442,0.8329,0.6768,0.8384,0.9171]
# Five_encodings = [0.8513,0.85,0.701,0.8506,0.9209]
#
#
# # Plotting
# x = range(len(x_x))
# width = 0.15
#
# fig, ax = plt.subplots(figsize=(10, 6))
# rects1 = ax.bar(x, onehot_BLOSUM62_Binary1_Binary2, width, label='onehot_BLOSUM62_Binary1_Binary2')
# rects2 = ax.bar([i + width for i in x], BLOSUM62_Zscale_Binary1_Binary2, width, label='BLOSUM62_Zscale_Binary1_Binary2')
# rects3 = ax.bar([i + 2 * width for i in x], onehot_Zscale_Binary1_Binary2, width, label='onehot_Zscale_Binary1_Binary2')
# rects4 = ax.bar([i + 3 * width for i in x], onehot_BLOSUM62_Zscale_Binary1, width, label='onehot_BLOSUM62_Zscale_Binary1')
# rects5 = ax.bar([i + 1 * width for i in x], onehot_BLOSUM62_Zscale_Binary2, width, label='onehot_BLOSUM62_Zscale_Binary2')
# rects6 = ax.bar([i + 5 * width for i in x],Five_encodings,width,label='Five_encodings')
#
# # Set labels and title
# ax.set_xlabel('Feature Encoding')
# ax.set_ylabel('Scores')
# ax.set_title('Comparison of Quadruple Feature Encodings')
# ax.set_xticks([i + 2 * width for i in x])
# ax.set_xticklabels(x_x)
#
# # Move the legend to the right side of the plot
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
# plt.xticks(rotation=45)
#
# # Display the plot
# plt.tight_layout()
# plt.show()


#三种编码组合
# x_x = ['SN','SP','MCC','ACC','AUC']
# onehot_BLOSUM62_Zscale =[0.8302,0.849,0.6794,0.8398,0.9185]
# onehot_BLOSUM62_Binary1 =[0.8322,0.8319,0.6639,0.832,0.9132]
# onehot_BLOSUM62_Binary2 =[0.8332,0.8357,0.6688,0.8345,0.9076]
# onehot_Zscale_Binary1 =[0.8251,0.8585,0.6843,0.8423,0.9174]
# one_hot_Zscale_Binary2 =[0.8533,0.8319,0.6849,0.8423,0.9187]
# onehot_Binary1_Binary2 =[0.8633,0.8139,0.6772,0.8379,0.91]
# BLOSUM62_Zscale_Binary1 =[0.8161,0.8632,0.6806,0.8403,0.9156]
# BLOSUM62_Zscale_Binary2 =[0.8332,0.8462,0.6794,0.8398,0.914]
# BLOSUM62_Binary1_Binary2 =[0.8191,0.8604,0.6805,0.8403,0.9173]
# Zscale_Binary1_Binary2 =[0.8312,0.8509,0.6823,0.8413,0.9161]
# Five_encodings = [0.8513,0.85,0.701,0.8506,0.9209]
#
# # Plotting
# x = range(len(x_x))
# width = 0.08
#
# fig, ax = plt.subplots(figsize=(10, 6))
# rects1 = ax.bar(x, onehot_BLOSUM62_Zscale, width, label='onehot_BLOSUM62_Zscale')
# rects2 = ax.bar([i + width for i in x], onehot_BLOSUM62_Binary1, width, label='onehot_BLOSUM62_Binary1')
# rects3 = ax.bar([i + 2 * width for i in x], onehot_BLOSUM62_Binary2, width, label='onehot_BLOSUM62_Binary2')
# rects4 = ax.bar([i + 3 * width for i in x], onehot_Zscale_Binary1, width, label='onehot_Zscale_Binary1')
# rects5 = ax.bar([i + 1 * width for i in x], one_hot_Zscale_Binary2, width, label='one_hot_Zscale_Binary2')
# rects6 = ax.bar([i + 5 * width for i in x], onehot_Binary1_Binary2, width, label='onehot_Binary1_Binary2')
# rects7 = ax.bar([i + 6 * width for i in x], BLOSUM62_Zscale_Binary1, width, label='BLOSUM62_Zscale_Binary1')
# rects8 = ax.bar([i + 7 * width for i in x], BLOSUM62_Zscale_Binary2, width, label='BLOSUM62_Zscale_Binary2')
# rects9 = ax.bar([i + 8 * width for i in x], BLOSUM62_Binary1_Binary2, width, label='BLOSUM62_Binary1_Binary2')
# rects10 = ax.bar([i + 9 * width for i in x], Zscale_Binary1_Binary2, width, label='Zscale_Binary1_Binary2')
# rects11 = ax.bar([i + 10 * width for i in x],Five_encodings,width,label='Five_encodings')
#
# # Set labels and title
# ax.set_xlabel('Feature Encoding')
# ax.set_ylabel('Scores')
# ax.set_title('Comparison of Triple Feature Encodings')
# ax.set_xticks([i + 2 * width for i in x])
# ax.set_xticklabels(x_x)
#
# # Move the legend to the right side of the plot
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
# plt.xticks(rotation=45)
#
# # Display the plot
# plt.tight_layout()
# plt.show()




#两种编码组合
x_x = ['SN','SP','MCC','ACC','AUC']
onehot_BLOSUM62 = [0.808,0.8538,0.6629,0.8315,0.9086]
onehot_Zscale = [0.8432,0.8367,0.6797,0.8398,0.916]
onehot_Binary1 = [0.8492,0.8243,0.6733,0.8364,0.9169]
onehot_Binary2 = [0.808,0.8538,0.6629,0.8315,0.9086]
BLOSUM62_Zscale = [0.8352,0.8338,0.6689,0.8345,0.9139]
BLOSUM62_Binary1 = [0.8492,0.8319,0.6809,0.8403,0.915]
BLOSUM62_Binary2 = [0.8422,0.8386,0.6806,0.8403,0.9156]
Zscale_Binary1 = [0.803,0.8566,0.6611,0.8306,0.9093]
Zscale_Binary2 = [0.8382,0.8291,0.667,0.8335,0.9117]
Binary1_Binary2 = [0.8402,0.8424,0.6824,0.8413,0.9141]
Five_encodings = [0.8513,0.85,0.701,0.8506,0.9209]

# Plotting
x = range(len(x_x))
width = 0.08

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x, onehot_BLOSUM62, width, label='onehot_BLOSUM62')
rects2 = ax.bar([i + width for i in x], onehot_Zscale, width, label='onehot_Zscale')
rects3 = ax.bar([i + 2 * width for i in x], onehot_Binary1, width, label='onehot_Binary1')
rects4 = ax.bar([i + 3 * width for i in x], onehot_Binary2, width, label='onehot_Binary2')
rects5 = ax.bar([i + 4 * width for i in x], BLOSUM62_Zscale, width, label='BLOSUM62_Zscale')
rects6 = ax.bar([i + 5 * width for i in x], BLOSUM62_Binary1, width, label='BLOSUM62_Binary1')
rects7 = ax.bar([i + 6 * width for i in x], BLOSUM62_Binary2, width, label='BLOSUM62_Binary2')
rects8 = ax.bar([i + 7 * width for i in x], Zscale_Binary1, width, label='Zscale_Binary1')
rects9 = ax.bar([i + 8 * width for i in x], Zscale_Binary2, width, label='Zscale_Binary2')
rects10 = ax.bar([i + 9 * width for i in x], Binary1_Binary2, width, label='Binary1_Binary2')
rects11 = ax.bar([i + 10 * width for i in x],Five_encodings,width,label='Five_encodings')

# Set labels and title
ax.set_xlabel('Feature Encoding')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Dual Feature Encodings')
ax.set_xticks([i + 2 * width for i in x])
ax.set_xticklabels(x_x)

# Move the legend to the right side of the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()

#单个编码
# x_x = ['SN','SP','MCC','ACC','AUC']
# onehot = [0.8583,0.8148,0.673,0.8359,0.9106]
# BLOSUM62 = [0.8533,0.8319,0.6849,0.8423,0.9175]
# Zscale = [0.8211,0.7949,0.6158,0.8076,0.8835]
# Binary1 = [0.7879,0.6676,0.458,0.7261,0.7911]
# Binary2 = [0.8231,0.8357,0.6589,0.8296,0.9068]
# Five_encodings = [0.8513,0.85,0.701,0.8506,0.9209]
#
# # Plotting
# x = range(len(x_x))
# width = 0.08
#
# fig, ax = plt.subplots(figsize=(10, 6))
# rects1 = ax.bar(x, onehot, width, label='onehot')
# rects2 = ax.bar([i + width for i in x], BLOSUM62, width, label='BLOSUM62')
# rects3 = ax.bar([i + 2 * width for i in x], Zscale, width, label='Zscale')
# rects4 = ax.bar([i + 3 * width for i in x], Binary1, width, label='Binary1')
# rects5 = ax.bar([i + 1 * width for i in x], Binary2, width, label='Binary2')
# rects6 = ax.bar([i + 5 * width for i in x],Five_encodings,width,label='Five_encodings')
#
# # Set labels and title
# ax.set_xlabel('Feature Encoding')
# ax.set_ylabel('Scores')
# ax.set_title('Comparison of Single Feature Encodings')
# ax.set_xticks([i + 2 * width for i in x])
# ax.set_xticklabels(x_x)
#
# # Move the legend to the right side of the plot
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
# plt.xticks(rotation=45)
#
# # Display the plot
# plt.tight_layout()
# plt.show()












