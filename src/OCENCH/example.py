from sklearn.datasets import make_blobs
import pandas as pd
from ocench import *

num_normal_samples = 1000 # Training dataset size (only normal data)
num_abnormal_samples = 10 # Number of anomalies to classify in test

# Create a toy dataset using two isotropic Gaussian blobs (one for each class)
X_train, _ = make_blobs(n_samples=num_normal_samples, centers= [(1,1)], n_features=10, cluster_std=1, random_state=0)
X_test_abnormal, _ = make_blobs(n_samples=num_abnormal_samples, centers=[(20,20)], n_features=10, cluster_std=1, random_state=0)
X_test_normal, _ = make_blobs(n_samples=num_abnormal_samples, centers=[(1,1)], n_features=10, cluster_std=1, random_state=0)
Y_train = [0] * num_normal_samples
Y_test_abnormal = [1] * num_abnormal_samples
Y_test_normal = [0] * num_abnormal_samples
X_test = np.concatenate((X_test_normal, X_test_abnormal), axis=0)
Y_test = np.concatenate((Y_test_normal, Y_test_abnormal), axis=0)

model = OCENCH_train(X=X_train, n_projections=20, l=2, extend=0.3) # Train the model with only normal data
prediction = OCENCH_classify(X=X_test, model=model) # Predict new (normal and abnormal) data

print("[0 = Normal | 1 = Anomaly]")
print("Real classes: ", Y_test)
print("Predictions: ", prediction)
