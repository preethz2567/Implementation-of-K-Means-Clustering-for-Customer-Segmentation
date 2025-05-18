# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
**1.Import the necessary packages using the import statement.**

Start by importing essential Python libraries such as pandas for data handling, matplotlib.pyplot for visualization, and KMeans from sklearn.cluster for clustering the data.

**2.Read the given CSV file using read_csv() method and print the number of contents to be displayed using df.head().**

Load the dataset using pandas.read_csv() and display the first few rows using df.head() to understand the structure and contents of the data.

**3.Import KMeans and use a for loop to cluster the data.**

Apply the Elbow Method by running a for loop from 1 to 10 clusters. In each iteration, initialize and fit the KMeans model, and calculate the WCSS (Within-Cluster Sum of Squares) to find the optimal number of clusters.

**4.Predict the cluster and plot data graphs.**

Using the optimal number of clusters, fit the KMeans model again and use .predict() to assign each data point to a cluster. Visualize the clusters using scatter plots and mark the centroids.

**5.Print the outputs and end the program.**

Display the final clustered data, centroids, or evaluation if needed. Ensure all graphs and results are shown clearly before completing the program.

## Program / Output:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: PREETHI D
RegisterNumber: 212224040250
```

**Import Required Libraries**
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```
```
print(df.head())
```
![image](https://github.com/user-attachments/assets/d14ba765-1d4b-4033-9e14-eca9b6e97555)

![image](https://github.com/user-attachments/assets/0d5c4278-dc9c-4b14-acb4-8d7c0e9b0a41)

U**se Elbow Method to Find Optimal Number of Clusters**
```
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/2cb88bbc-3697-4029-b922-f57b93d1914c)

**Apply K-Means with Optimal Clusters**
```
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)
```
**Visualize the Clusters**
```
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']

for i in range(5):
    plt.scatter(x.values[y_kmeans == i, 0], x.values[y_kmeans == i, 1],
                s=100, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids')

plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/6773356d-0005-4821-813d-aca7c4e54e58)

```
df['Cluster'] = y_kmeans
print(df.head())
```
![image](https://github.com/user-attachments/assets/f7ca2358-de6b-42f2-aae3-ffc9a9cbb065)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
