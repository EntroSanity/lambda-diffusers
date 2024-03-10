import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to read and preprocess data
def read_and_preprocess(filename):
    df = pd.read_csv(filename)
    df = df[(df['latency'] != 'error') & (df['memory_usage'] != 'error')]
    df['latency'] = pd.to_numeric(df['latency'])
    df['memory_usage'] = pd.to_numeric(df['memory_usage'])
    df['resolution'] = df['width'].astype(str) + 'x' + df['height'].astype(str)
    return df

# Load and preprocess the data
lpw_df = read_and_preprocess('results_LPW.csv')
sdxl_df = read_and_preprocess('results_SDXL.csv')

# Set the style of seaborn
sns.set(style="whitegrid")

# Plotting latency
plt.figure(figsize=(12, 6))
sns.lineplot(data=pd.concat([lpw_df, sdxl_df]), x='resolution', y='latency', hue='pipeline_type', style='precision', markers=True, dashes=False)
plt.title('Latency by Resolution for Different Pipelines and Precisions')
plt.xlabel('Resolution (Width x Height)')
plt.ylabel('Latency (s)')
plt.xticks(rotation=45)
plt.legend(title='Pipeline / Precision')
plt.tight_layout()
plt.savefig('latency_by_resolution.png')  # Save the figure as a PNG file
plt.show()

# Plotting memory usage
plt.figure(figsize=(12, 6))
sns.lineplot(data=pd.concat([lpw_df, sdxl_df]), x='resolution', y='memory_usage', hue='pipeline_type', style='precision', markers=True, dashes=False)
plt.title('Memory Usage by Resolution for Different Pipelines and Precisions')
plt.xlabel('Resolution (Width x Height)')
plt.ylabel('Memory Usage (GiB)')
plt.xticks(rotation=45)
plt.legend(title='Pipeline / Precision')
plt.tight_layout()
plt.savefig('memory_usage_by_resolution.png')  # Save the figure as a PNG file
plt.show()
