from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

class AUPRCPlotter:
    def __init__(self, model_evaluator):
        self.model_evaluator = model_evaluator

    def plot_auprc(self):
        # Find the best model based on F1-Score
        best_model_name = self.model_evaluator.results_df.sort_values('F1-Score', ascending=False).iloc[0]['Model']
        best_model = [model for name, (model, _) in self.model_evaluator.models.items() if name == best_model_name][0]

        # Train the best model on the full training set and predict probabilities on the test set
        best_model.fit(self.model_evaluator.X_train_scaled, self.model_evaluator.y_train)
        y_scores = best_model.predict_proba(self.model_evaluator.X_test_scaled)[:, 1]

        # Calculate precision and recall for various thresholds
        precision, recall, _ = precision_recall_curve(self.model_evaluator.y_test, y_scores)

        # Calculate AUPRC
        auprc = auc(recall, precision)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'AUPRC = {auprc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {best_model_name}')
        plt.legend(loc='best')
        plt.show()