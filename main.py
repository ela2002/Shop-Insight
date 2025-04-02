import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. Chargement des données ---
def load_data(file_path):
    """Charge les données depuis un fichier Excel et effectue des visualisations initiales."""
    logging.info("Chargement des données...")
    try:
        data = pd.read_excel(file_path)
        logging.info(f"Données chargées avec {data.shape[0]} lignes et {data.shape[1]} colonnes")

        # Visualisation des premières lignes
        logging.info("Aperçu des premières lignes du dataset :")
        logging.info(data.head())

        # Visualisation des types de données
        logging.info("Types de données :")
        logging.info(data.dtypes)

        # --- Visualisation initiale des valeurs manquantes ---
        plt.figure(figsize=(10, 5))
        sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
        plt.title("Valeurs manquantes dans les données")
        plt.show()

        return data
    except Exception as e:
        logging.error(f"Erreur lors du chargement des données : {e}")
        raise


# --- 2. Nettoyage des données ---
def clean_data(data):
    """Nettoie les données en supprimant les valeurs manquantes et aberrantes."""
    logging.info("Nettoyage des données...")

    # Suppression des valeurs manquantes dans 'CustomerID'
    data.dropna(subset=['CustomerID'], inplace=True)

    # Suppression des transactions annulées (InvoiceNo commençant par 'C')
    data = data[~data['InvoiceNo'].astype(str).str.startswith('C')]

    # Suppression des valeurs aberrantes (Quantité et Prix unitaire positifs)
    data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]

    # Création de la colonne TotalPrice
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

    logging.info(f"Dataset après nettoyage : {data.shape[0]} lignes")

    # --- Visualisation des valeurs manquantes après nettoyage ---
    plt.figure(figsize=(10, 5))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title("Valeurs manquantes après nettoyage")
    plt.show()

    """Affiche les distributions de Quantity, UnitPrice et TotalPrice après transformation logarithmique."""

    cols = ['Quantity', 'UnitPrice', 'TotalPrice']

    plt.figure(figsize=(15, 5))
    for i, col in enumerate(cols):
        plt.subplot(1, 3, i + 1)
        transformed_data = np.log1p(data[col])  # log(1+x) pour éviter les problèmes avec 0
        sns.histplot(transformed_data, bins=50, kde=True)
        plt.title(f"Distribution de {col} (Log)")
        plt.xlabel(f'Log(1 + {col})')

    plt.tight_layout()
    plt.show()

    return data


# --- 3. Calcul des variables RFM ---
def calculate_rfm(data):
    """Calcule les variables RFM (Recency, Frequency, Monetary)."""
    logging.info("Calcul des variables RFM...")

    # Date de référence pour le calcul de la Recency
    current_date = data['InvoiceDate'].max() + timedelta(days=1)

    # Calcul des métriques RFM
    rfm = data.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (current_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    )

    # --- Visualisation des distributions RFM ---
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(['Recency', 'Frequency', 'Monetary']):
        plt.subplot(1, 3, i + 1)
        sns.histplot(rfm[col], bins=50, kde=True)
        plt.title(f"Distribution de {col}")
    plt.tight_layout()
    plt.show()

    # --- Visualisation de la matrice de corrélation RFM ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(rfm.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matrice de Corrélation des Variables RFM")
    plt.show()

    logging.info("Calcul des variables RFM terminé")
    return rfm

# 4. K-Means avec Elbow Method
def elbow_method(rfm_scaled):
    logging.info("Application de la méthode Elbow pour déterminer le nombre optimal de clusters...")
    wcss = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('WCSS')
    plt.title('Méthode Elbow')
    plt.show()

# --- 4. Clustering RFM avec K-means ---
def perform_kmeans_clustering(rfm, n_clusters=4):
    """Effectue un clustering K-means sur les données RFM."""
    logging.info("Clustering K-means en cours...")

    # Normalisation des données
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Clustering K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster_KMeans'] = kmeans.fit_predict(rfm_scaled)

    # Calcul de l'inertie
    inertia = kmeans.inertia_
    logging.info(f"Inertie du modèle K-means : {inertia}")

    # Calcul du score de silhouette
    silhouette_avg_kmeans = silhouette_score(rfm_scaled, rfm['Cluster_KMeans'])
    logging.info(f"Score de silhouette K-means : {silhouette_avg_kmeans}")

    # --- Visualisation des clusters en 2D ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Recency', y='Monetary', hue='Cluster_KMeans', data=rfm, palette='Set1', alpha=0.8, edgecolor='black')
    plt.title("Clusters RFM Kmeans")
    plt.xlabel("Recency (jours depuis dernier achat)")
    plt.ylabel("Monetary (Montant total dépensé)")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.show()

    # --- Visualisation des clusters en 3D ---
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'], c=rfm['Cluster_KMeans'], cmap='Set1')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title('Clusters RFM en 3D')
    plt.show()

    logging.info("Clustering K-means terminé")
    return rfm

# --- 5. Clustering Hiérarchique (CAH) ---
def perform_hierarchical_clustering(rfm, method='ward'):
    """Effectue un clustering hiérarchique (CAH) et visualise le dendrogramme."""
    logging.info("Clustering hiérarchique (CAH) en cours...")

    # Normalisation des données
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    # Création de la matrice de distance
    Z = linkage(rfm_scaled, method=method)

    # Visualisation du dendrogramme
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title(f"Dendrogramme - Méthode {method}")
    plt.xlabel("Index des clients")
    plt.ylabel("Distance")
    plt.show()

    # Détermination des clusters à partir du dendrogramme
    n_clusters = 4  # Exemple, ajuster selon l'analyse du dendrogramme
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    rfm['Cluster_Hierarchical'] = hierarchical.fit_predict(rfm_scaled)

    logging.info(f"Clustering hiérarchique effectué avec {n_clusters} clusters")

    # --- Visualisation des clusters en 2D ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Recency', y='Monetary', hue='Cluster_Hierarchical', data=rfm, palette='Set1', alpha=0.8,
                    edgecolor='black')
    plt.title("Clusters RFM CAH")
    plt.xlabel("Recency (jours depuis dernier achat)")
    plt.ylabel("Monetary (Montant total dépensé)")
    plt.legend(title="Cluster")
    plt.grid(True)
    plt.show()

    # Calcul du score de silhouette pour CAH
    silhouette_avg_hierarchical = silhouette_score(rfm_scaled, rfm['Cluster_Hierarchical'])
    logging.info(f"Score de silhouette CAH : {silhouette_avg_hierarchical}")

    return rfm


# 6. Association Rules avec Apriori et FP-Growth
def extract_association_rules(data, min_support=0.02):
    logging.info("Préparation des données pour Apriori et FP-Growth...")
    item_counts = data['Description'].value_counts()
    frequent_items = item_counts[item_counts > 50].index
    filtered_data = data[data['Description'].isin(frequent_items)]
    basket = (filtered_data[filtered_data['Quantity'] > 0]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().fillna(0))
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Apriori
    logging.info("Extraction des règles d'association avec Apriori...")
    frequent_itemsets_apriori = apriori(basket, min_support=min_support, use_colnames=True)
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1.0)

    # FP-Growth
    logging.info("Extraction des règles d'association avec FP-Growth...")
    frequent_itemsets_fpgrowth = fpgrowth(basket, min_support=min_support, use_colnames=True)
    rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1.0)


    # Afficher les principales métriques
    logging.info(f"Nombre total de règles générées avec Apriori : {len(rules_apriori)}")
    logging.info("Métriques des règles d'association :")
    logging.info(f"Support moyen des règles : {rules_apriori['support'].mean()}")
    logging.info(f"Confiance moyenne des règles : {rules_apriori['confidence'].mean()}")
    logging.info(f"Lift moyen des règles : {rules_apriori['lift'].mean()}")

    # Afficher les principales métriques
    logging.info(f"Nombre total de règles générées avec fpgrowth : {len(rules_fpgrowth)}")
    logging.info("Métriques des règles d'association :")
    logging.info(f"Support moyen des règles : {rules_fpgrowth['support'].mean()}")
    logging.info(f"Confiance moyenne des règles : {rules_fpgrowth['confidence'].mean()}")
    logging.info(f"Lift moyen des règles : {rules_fpgrowth['lift'].mean()}")
    return rules_apriori, rules_fpgrowth


# 7. Visualisation des règles d'association
def plot_association_rules(rules_apriori, rules_fpgrowth):
    if not rules_apriori.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='support', y='lift', size='confidence', hue='lift', data=rules_apriori, sizes=(20, 200))
        plt.title('Règles d\'Association avec Apriori')
        plt.show()

    if not rules_fpgrowth.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='support', y='lift', size='confidence', hue='lift', data=rules_fpgrowth, sizes=(20, 200))
        plt.title('Règles d\'Association avec FP-Growth')
        plt.show()



# --- 7. Pipeline principal ---
def main(file_path):
    """Pipeline principal pour l'analyse RFM et les règles d'association."""
    # Chargement des données
    data = load_data(file_path)

    # Nettoyage des données
    data = clean_data(data)

    # Calcul des variables RFM
    rfm = calculate_rfm(data)

    elbow_method(rfm[['Recency', 'Frequency', 'Monetary']])

    # Clustering K-means
    rfm_Kmeans = perform_kmeans_clustering(rfm, n_clusters=4)

    # CAH clustering
    rfm_CAH = perform_hierarchical_clustering(rfm)

    #Extraction des règles d'association
    rules_apriori, rules_fpgrowth = extract_association_rules(data, min_support=0.02)

    # Visualisation des règles d'association
    plot_association_rules(rules_apriori, rules_fpgrowth)

# --- Exécution du pipeline ---
if __name__ == '__main__':
    main("Online_Retail.xlsx")
