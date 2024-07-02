import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class FeatureSelector:
    def __init__(self, df):
        self.df = df
        self.X = df.drop(['Address', 'FLAG'], axis=1)
        self.y = df['FLAG']
        self.rf_model = RandomForestClassifier(random_state=42)
        self.feature_importance_df = None
        self.selected_features = None

    def fit_model(self):
        self.rf_model.fit(self.X, self.y)

    def get_feature_importances(self):
        feature_importances = self.rf_model.feature_importances_
        self.feature_importance_df = pd.DataFrame({'Feature': self.X.columns, 'Importance': feature_importances})
        self.feature_importance_df = self.feature_importance_df.sort_values(by='Importance', ascending=False)

    def select_top_features(self, top_n=8):
        self.selected_features = self.feature_importance_df.head(top_n)['Feature']

    def get_selected_dataframe(self):
        selected_df = self.df[['FLAG'] + list(self.selected_features)]
        return selected_df
