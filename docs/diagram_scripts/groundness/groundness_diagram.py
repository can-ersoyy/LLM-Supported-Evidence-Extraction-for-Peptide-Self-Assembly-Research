import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    'font.size': 11,          
    'axes.labelsize': 12,     
    'axes.titlesize': 14,     
    'xtick.labelsize': 11,    
    'ytick.labelsize': 11,    
})

# values for location based gorunding
df_loc = pd.DataFrame({
    'LLM': [0, 1, 0.76],
    'Search': [0, 1, 0.39] 
})

# values for evidence based grodunding
df_evidence = pd.DataFrame({
    'LLM': [0, 1, 0.83],
    'Search': [0.2, 0.8, 0.46] 
})

df = df_evidence

labels = ['LLM', 'Search']
colors = ['blue', 'green']
x_positions = [1, 2]

fig, ax = plt.subplots(figsize=(5, 5))

for i, (col, color) in enumerate(zip(df.columns, colors)):
    vals = df[col]
    min_rate = vals.iloc[0]    
    max_rate = vals.iloc[1]    
    avg_rate = vals.iloc[2]  
    x = x_positions[i]
    
    ax.vlines(x=x, ymin=min_rate, ymax=max_rate, color=color, linewidth=2, alpha=0.7)
    
    ax.hlines(y=min_rate, xmin=x-0.08, xmax=x+0.08, color=color, linewidth=2)
    ax.hlines(y=max_rate, xmin=x-0.08, xmax=x+0.08, color=color, linewidth=2)
    
    ax.scatter([x], [avg_rate], color=color, s=100, zorder=3, marker='o')
    ax.text(x + 0.12, avg_rate, f'Avg: {avg_rate:.2f}', va='center', fontsize=10, color=color, fontweight='bold')

ax.set_xticks(x_positions)
ax.set_xticklabels(labels)
ax.set_xlim(0.5, 2.5)
ax.set_ylim(-0.1, 1.1)

#ax.set_title("with Location (28 entries)", pad=12, fontweight='bold')
ax.set_title("with Evidence (32 entries)", pad=12, fontweight='bold')
ax.set_ylabel("Groundness Rate", labelpad=12, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.3)


plt.tight_layout()
plt.show()