import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Load the data
data_path = 'd:/ICH_phenogrouping/model_building.csv'
data = pd.read_csv(data_path)

# Prepare the data (assuming the first column is the patient ID)
X = data.iloc[:, 1:]  # All columns except the first one
y = data.iloc[:, 0]   # The first column as patient IDs

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Principal Component Analysis (PCA)
pca = PCA(n_components=33)
X_pca = pca.fit_transform(X_scaled)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Add the clustering result to the original data
data['Phenogroup'] = clusters

# Save the model
model_filename = 'd:/ICH_phenogrouping/phenogrouping.joblib'
dump({'scaler': scaler, 'pca': pca, 'kmeans': kmeans}, model_filename)

# Load the new patient data
sample_data_path = 'd:/ICH_phenogrouping/sample_data.csv'
sample_data = pd.read_csv(sample_data_path)

# Prepare the new patient data (assuming the first column is the patient ID)
X_sample = sample_data.iloc[:, 1:]  # All columns except the first one
X_sample_scaled = scaler.transform(X_sample)  # Standardize using the previously fitted scaler
X_sample_pca = pca.transform(X_sample_scaled)  # Reduce dimensions using the previously fitted PCA

# Use the model to classify the new patient
predicted_cluster = kmeans.predict(X_sample_pca)

# Output the classification result including the patient ID
print(f'Patient ID: {sample_data.iloc[:, 0].values[0]}')
print(f'Predicted Phenogroup: {predicted_cluster[0]}')