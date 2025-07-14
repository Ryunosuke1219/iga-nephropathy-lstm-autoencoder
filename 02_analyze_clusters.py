
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os

# --- Load Data ---
LATENT_REPS_FILE = 'output/latent_representations.npz'
BASELINE_FILE = 'data/baseline_873.csv'
OUTCOME_FILE = 'data/outcome.csv'

latent_data = np.load(LATENT_REPS_FILE)
latent_reps = latent_data['latent_reps']
patient_ids = latent_data['patient_ids']

baseline_df = pd.read_csv(BASELINE_FILE)
outcome_df = pd.read_csv(OUTCOME_FILE)

# --- K-means Clustering ---
def find_optimal_clusters(data, max_k=6):
    """Find the optimal number of clusters using silhouette and Calinski-Harabasz scores."""
    scores = {}
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        sil = silhouette_score(data, labels)
        cal = calinski_harabasz_score(data, labels)
        scores[k] = {'silhouette': sil, 'calinski_harabasz': cal}
    return pd.DataFrame(scores).T

cluster_scores = find_optimal_clusters(latent_reps)
print("Cluster Scores:\n", cluster_scores)

# Based on the paper, K=3 is optimal
OPTIMAL_K = 3
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(latent_reps)

# --- t-SNE Visualization ---
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_results = tsne.fit_transform(latent_reps)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=cluster_labels, palette=sns.color_palette("hls", OPTIMAL_K), legend='full')
plt.title('t-SNE visualization of patient clusters')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
if not os.path.exists('output/figures'):
    os.makedirs('output/figures')
plt.savefig('output/figures/tsne_clusters.png')
plt.show()

# --- Survival Analysis ---
# Merge dataframes
df = pd.DataFrame({'patient_id': patient_ids, 'cluster': cluster_labels})
df = pd.merge(df, baseline_df, on='patient_id')
df = pd.merge(df, outcome_df, on='patient_id')

# Kaplan-Meier Curve
kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 7))

for i in range(OPTIMAL_K):
    cluster_data = df[df['cluster'] == i]
    kmf.fit(cluster_data['time_to_event'], event_observed=cluster_data['event'], label=f'Cluster {i+1}')
    kmf.plot_survival_function()

# Log-rank test
results = logrank_test(df['time_to_event'][df['cluster'] == 0], df['time_to_event'][df['cluster'] != 0],
                       df['event'][df['cluster'] == 0], df['event'][df['cluster'] != 0])

plt.title('Kaplan-Meier Survival Curves by Cluster')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.text(0.5, 0.1, f'Log-rank p-value: {results.p_value:.4f}', transform=plt.gca().transAxes)
plt.savefig('output/figures/kaplan_meier_curves.png')
plt.show()

# --- Save Results ---
df.to_csv('output/tables/clustered_patient_data.csv', index=False)

print("Cluster analysis and visualization complete.")
