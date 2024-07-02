import pandas as pd

class DataCombiner:
    def __init__(self, mined_data_path, kaggle_data_path, output_path):
        self.mined_data_path = mined_data_path
        self.kaggle_data_path = kaggle_data_path
        self.output_path = output_path

    def load_data(self):
        self.data_mined = pd.read_csv(self.mined_data_path)
        self.data_kaggle = pd.read_csv(self.kaggle_data_path)

    def preprocess_kaggle_data(self):
        self.data_kaggle.drop(['Index', 'Unnamed: 0'], axis=1, inplace=True)
        self.data_kaggle.drop_duplicates(subset=['Address'], inplace=True)

    def drop_erc20_columns(self):
        erc20_columns = [' Total ERC20 tnxs', ' ERC20 total Ether received', ' ERC20 total ether sent',
                 ' ERC20 total Ether sent contract', ' ERC20 uniq sent addr', ' ERC20 uniq rec addr',
                 ' ERC20 uniq sent addr.1', ' ERC20 uniq rec contract addr', ' ERC20 avg time between sent tnx',
                 ' ERC20 avg time between rec tnx', ' ERC20 avg time between rec 2 tnx',
                 ' ERC20 avg time between contract tnx', ' ERC20 min val rec', ' ERC20 max val rec',
                 ' ERC20 avg val rec', ' ERC20 min val sent', ' ERC20 max val sent', ' ERC20 avg val sent',
                 ' ERC20 min val sent contract', ' ERC20 max val sent contract', ' ERC20 avg val sent contract',
                 ' ERC20 uniq sent token name', ' ERC20 uniq rec token name', ' ERC20 most sent token type',
                 ' ERC20_most_rec_token_type']
        self.data_kaggle.drop(erc20_columns, axis=1, inplace=True)

    def round_mined_data(self):
        round_columns = ['Avg min between sent tnx', 'Avg min between received tnx', 'Time Diff between first and last (Mins)']
        self.data_mined[round_columns] = self.data_mined[round_columns].round(2)

    def merge_datasets(self):
        self.merged_data = pd.merge(self.data_mined, self.data_kaggle, how='outer')

    def drop_unnecessary_columns(self):
        columns_to_remove = ['total ether sent contracts', 'max val sent to contract', 'Received Tnx',
                     'Sent tnx', 'total Ether sent', 'min value sent to contract', 'avg value sent to contract',
                     'Number of Created Contracts', 'max val sent', 'Unique Sent To Addresses']
        self.merged_data.drop(columns_to_remove, axis=1, inplace=True)

    def save_merged_data(self):
        self.merged_data.to_csv(self.output_path, index=False)

    def check_uniqueness(self):
        return self.merged_data['Address'].nunique() == self.merged_data.shape[0]

    def execute(self):
        self.load_data()
        self.preprocess_kaggle_data()
        self.drop_erc20_columns()
        self.round_mined_data()
        self.merge_datasets()
        self.drop_unnecessary_columns()
        self.save_merged_data()
        return self.check_uniqueness()