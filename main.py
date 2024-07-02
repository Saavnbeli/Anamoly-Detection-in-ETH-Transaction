import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from data_collection_and_processing.mining_data import EthereumTransactionAnalyzer
from data_collection_and_processing.combining_data import DataCombiner
from data_collection_and_processing.exploratory_data_analysis import DataAnalyzer
from data_collection_and_processing.feature_selection import FeatureSelector
from modeling.base_model_allfeatures import AllFeaturesBaseModel
from modeling.base_model_extractedfeatures import ExtractedFeaturesBaseModel
from modeling.tuned_model_allfeatures import TunedModel
from modeling.evaluation import AUPRCPlotter
from dotenv import load_dotenv

#EDA
analyzer = DataAnalyzer.load_data(r'..\Data\address_data_kaggle.csv')
analyzer.execute()

#Mining Data
load_dotenv()
api_key = "ETHERSCAN_API_KEY" #Not shared for security reasons
address_file_path = r'..\Data\addresses_mined_not_in_kaggle.csv'
output_file_path = r'..\Data\address_data_ethereum.csv'
analyzer = EthereumTransactionAnalyzer(api_key, address_file_path, output_file_path)
analyzer.process_addresses()

#Combining Data
combiner = DataCombiner(r'..\Data\address_data_ethereum.csv',
                        r'..\Data\address_data_kaggle.csv',
                        r'..\Data\address_data_combined.csv')
combiner.execute()

#Preparing data for modeling
df = pd.read_csv(r'..\Data\address_data_combined.csv')
df = df.drop(['Address'], axis=1)

feature_selector = FeatureSelector(df)
feature_selector.fit_model()

# Get important features
feature_selector.get_feature_importances()
feature_selector.select_top_features()
selected_df = feature_selector.get_selected_dataframe()

# Models with hyperparameter grids
models_dict = {
    'RandomForest': (RandomForestClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
    'SVM': (SVC(random_state=42), {'C': [1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}),
    'KNeighbors': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}),
    'MLP': (MLPClassifier(random_state=42), {'hidden_layer_sizes': [(100,), (50, 50), (30, 30, 30)], 'alpha': [0.0001, 0.001, 0.01]}),
    'LogisticRegression': (LogisticRegression(random_state=42), {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7]}),
    'GaussianNB': (GaussianNB(), {}),
    'DecisionTree': (DecisionTreeClassifier(random_state=42), {'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}),
    'AdaBoost': (AdaBoostClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 0.5, 1.0]}),
    'ExtraTrees': (ExtraTreesClassifier(random_state=42), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}),
}

#Base Model with all features
model_evaluator = AllFeaturesBaseModel(models=models_dict, features=df, target='FLAG')
model_evaluator.evaluate_models()

#Base Model with extracted features
model_evaluator = ExtractedFeaturesBaseModel(models=models_dict, features=selected_df, target='FLAG')
model_evaluator.evaluate_models()

#Hyperparameter Tuned with all features(had better results)
model_evaluator = TunedModel(models=models_dict, features=df, target='FLAG', hyperparameter_tuning=True)
model_evaluator.evaluate_models()

#Plotting the  curve
auprc_plotter = AUPRCPlotter(model_evaluator)
auprc_plotter.plot_auprc()