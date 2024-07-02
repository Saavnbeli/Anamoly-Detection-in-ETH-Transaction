import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class TunedModel:
    def __init__(self, models, features, target, test_size=0.2, random_state=42, hyperparameter_tuning=False):
        self.models = models
        self.features = features
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.hyperparameter_tuning = hyperparameter_tuning

    def _preprocess_data(self):
        X = self.features.drop(self.target, axis=1)
        y = self.features[self.target]

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Feature scaling
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)

    def _evaluate_model(self, model, name, param_grid=None):
        if self.hyperparameter_tuning and param_grid is not None:
            # Use grid search for hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid, scoring='f1', cv=3, n_jobs=-1)
            grid_search.fit(self.X_train_scaled, self.y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model

        best_model.fit(self.X_train_scaled, self.y_train)
        y_pred = best_model.predict(self.X_test_scaled)

        # Evaluate
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        # Add results to the DataFrame
        self.results_df = self.results_df.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
        }, ignore_index=True)

    def evaluate_models(self):
        self._preprocess_data()

        # Create a DataFrame to store model comparison results
        self.results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

        # Train and evaluate each model
        for name, (model, param_grid) in self.models.items():
            self._evaluate_model(model, name, param_grid)

        # Display the results
        print(self.results_df)
