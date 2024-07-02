import os
import requests
import pandas as pd
from web3 import Web3

class EthereumTransactionAnalyzer:
    def __init__(self, api_key, address_file_path, output_file_path):
        self.api_key = api_key
        self.address_file_path = address_file_path
        self.output_file_path = output_file_path
        self.combined_df = pd.DataFrame()
        self.total_txns_count = 0

    def create_api_url(self, wallet_address, page_number, limit):
        base_url = "https://api.etherscan.io/api"
        params = f"module=account&action=txlist&address={wallet_address}&startblock=0&endblock=99999999"
        params += f"&page={page_number}&offset={limit}&sort=asc&apikey={self.api_key}"
        return f"{base_url}?{params}"

    def analyze_ethereum_transactions(self, wallet_address):
        api_url = self.create_api_url(wallet_address, 1, 0)
        api_response = requests.get(api_url)
        if api_response.status_code != 200:
            return pd.DataFrame()
        transaction_data = pd.DataFrame(api_response.json()['result'])

        # Convert Wei to ETH
        transaction_data['value_in_eth'] = transaction_data['value'].apply(lambda x: Web3.fromWei(int(x), 'ether'))
        transaction_data['transaction_type'] = transaction_data['from'].apply(
            lambda x: 'outgoing' if x == wallet_address else 'incoming')

        # Analyze sent transactions
        sent_transactions = transaction_data[transaction_data['transaction_type'] == 'outgoing']
        sent_transactions_sorted = sent_transactions.sort_values(by=['timeStamp'])
        sent_transactions_sorted['timeStamp'] = sent_transactions_sorted['timeStamp'].astype(int)

        # Transactions to contracts
        contract_transactions = sent_transactions[sent_transactions['contractAddress'] != '']

        # Statistics for sent transactions
        stats = {
            'total_sent_transactions': len(sent_transactions),
            'min_sent_value': sent_transactions['value_in_eth'].min(),
            'max_sent_value': sent_transactions['value_in_eth'].max(),
            'avg_sent_value': sent_transactions['value_in_eth'].mean(),
            'min_contract_value': contract_transactions['value_in_eth'].min(),
            'max_contract_value': contract_transactions['value_in_eth'].max(),
            'avg_contract_value': contract_transactions['value_in_eth'].mean(),
            'total_eth_sent': sent_transactions['value_in_eth'].sum(),
            'total_eth_sent_to_contracts': contract_transactions['value_in_eth'].sum(),
            'unique_sent_addresses': len(sent_transactions['to'].unique()),
        }

        # Analyze received transactions
        received_transactions = transaction_data[transaction_data['transaction_type'] == 'incoming']
        received_transactions_sorted = received_transactions.sort_values(by=['timeStamp'])
        received_transactions_sorted['timeStamp'] = received_transactions_sorted['timeStamp'].astype(int)

        # Statistics for received transactions
        stats.update({
            'total_received_transactions': len(received_transactions),
            'min_received_value': received_transactions['value_in_eth'].min(),
            'max_received_value': received_transactions['value_in_eth'].max(),
            'avg_received_value': received_transactions['value_in_eth'].mean(),
            'total_eth_received': received_transactions['value_in_eth'].sum(),
            'unique_received_addresses': len(received_transactions['from'].unique()),
        })

        # Time analysis
        transaction_data['timeStamp'] = transaction_data['timeStamp'].astype(int)
        transaction_data_sorted = transaction_data.sort_values(by=['timeStamp'])
        time_diff = transaction_data_sorted['timeStamp'].diff()
        time_stats = transaction_data_sorted.groupby('transaction_type')['timeStamp'].sum() / 60

        # Additional stats
        stats.update({
            'time_diff_first_last': (transaction_data_sorted['timeStamp'].max() - transaction_data_sorted[
                'timeStamp'].min()) / 60,
            'total_transactions': len(transaction_data),
            'num_created_contracts': len(transaction_data[transaction_data['contractAddress'] != '']),
            'avg_time_between_sent': time_stats['outgoing'] / stats['total_sent_transactions'],
            'avg_time_between_received': time_stats['incoming'] / stats['total_received_transactions'],
            'total_eth_balance': stats['total_eth_received'] - stats['total_eth_sent'],
        })

        return pd.DataFrame([stats])

    def handle_empty_address(self, wallet_address):
        empty_stats = {
            'Address': wallet_address, 'FLAG': 1,
            'total_sent_transactions': 0, 'total_received_transactions': 0,
            'num_created_contracts': 0,
            'unique_received_addresses': 0, 'unique_sent_addresses': 0,
            'min_received_value': 0, 'max_received_value': 0, 'avg_received_value': 0,
            'min_sent_value': 0, 'max_sent_value': 0, 'avg_sent_value': 0,
            'min_contract_value': 0, 'max_contract_value': 0, 'avg_contract_value': 0,
            'total_transactions': 0,
            'total_eth_sent': 0, 'total_eth_received': 0,
            'total_eth_sent_to_contracts': 0, 'total_eth_balance': 0,
            'time_diff_first_last': 0, 'avg_time_between_sent': 0, 'avg_time_between_received': 0,
        }
        return pd.DataFrame([empty_stats])

    def process_addresses(self):
        address_list = pd.read_csv(self.address_file_path)
        addresses = address_list['Address'].tolist()

        for index, address in enumerate(addresses):
            try:
                temp_df = self.analyze_ethereum_transactions(address)
                mode = 'w' if index == 0 else 'a'
                header = True if index == 0 else False
                temp_df.to_csv(self.output_file_path, mode=mode, index=False, header=header)
                self.combined_df = pd.concat([self.combined_df, temp_df])
                current_txns = temp_df.iloc[0]['total_transactions']
                self.total_txns_count += current_txns
                print(f"Address {index}: {address} processed. {current_txns} transactions retrieved. Total: {self.total_txns_count}.")
            except Exception as e:
                empty_df = self.handle_empty_address(address)
                self.combined_df = pd.concat([self.combined_df, empty_df])
                empty_df.to_csv(self.output_file_path, mode='a', index=False, header=False)
                print(f"Address {index}: {address} processed. No transactions retrieved. Total: {self.total_txns_count}.")

        self.combined_df.reset_index(drop=True, inplace=True)

