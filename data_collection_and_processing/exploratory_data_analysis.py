import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

class DataAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        self.transactions = pd.read_csv(self.data_path)
        self.transactions_original = self.transactions.copy()

    def display_initial_info(self):
        print(self.transactions.head(10))
        print(self.transactions['avg val sent'].unique())
        print(self.transactions.shape)
        print(self.transactions.columns)
        self.transactions.info()
        print(self.transactions.describe())
        print(self.transactions.nunique())

    def plot_density_sent_tnx(self):
        sns.kdeplot(self.transactions['Sent tnx'])
        plt.show()

    def filter_transactions(self):
        self.filtered_transactions = self.transactions[(self.transactions['Sent tnx'] < 2) & (self.transactions['FLAG'] == 1)]
        print(self.filtered_transactions.shape)

    def clean_data(self):
        self.transactions_cleaned = self.transactions.dropna()
        empty_cols = [' ERC20 avg time between contract tnx', ' ERC20 max val sent contract',
              ' ERC20 min val sent contract', ' ERC20 avg val sent contract', ' ERC20 avg time between sent tnx',
              ' ERC20 avg time between rec tnx', ' ERC20 avg time between rec 2 tnx']
        self.transactions_cleaned = self.transactions_cleaned.drop(columns=empty_cols)

    def analyze_features(self):
        features_df = self.transactions_cleaned.drop(columns='FLAG')
        numeric_features = features_df.select_dtypes(include=np.number)
        correlation_matrix = numeric_features.corr().round(2)
        upper_triangle_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.figure(figsize=(20,12))
        sns.heatmap(correlation_matrix, mask=upper_triangle_mask, annot=True)
        plt.show()

    def dimensionality_reduction(self):
        erc20_features = [' ERC20 total Ether received', ' ERC20 avg val sent', ' ERC20 max val rec',
                  ' ERC20 total ether sent', ' ERC20 avg val rec', ' ERC20 max val sent']
        erc20_subset = self.transactions[erc20_features]
        scaler = MinMaxScaler()
        erc20_scaled = scaler.fit_transform(erc20_subset)
        pca_model = PCA(n_components=2)
        pca_result = pca_model.fit_transform(erc20_scaled)
        tsne_model = TSNE(learning_rate=50)
        tsne_result = tsne_model.fit_transform(pca_result)
        self.transactions['tsne_x'] = tsne_result[:, 0]
        self.transactions['tsne_y'] = tsne_result[:, 1]
        sns.scatterplot(x='tsne_x', y='tsne_y', data=self.transactions, alpha=0.6, hue='FLAG', style='FLAG')
        plt.show()

    def execute(self):
        self.load_data()
        self.display_initial_info()
        self.plot_density_sent_tnx()
        self.filter_transactions()
        self.clean_data()
        self.analyze_features()
        self.dimensionality_reduction()
