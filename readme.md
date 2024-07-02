This project aims to identify fraudulent accounts in the Ethereum blockchain using a combination of data from Kaggle and data scraped using the Etherscan API.

Kaggle Dataset Link - https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

Etherscan API Key request - https://info.etherscan.com/api-keys/

How to Run Code -

Upon substituting the ETHERSCAN_API_KEY string in the api_key variable in main.py, the entire code can be run using calls inside the main.py file.
Replace all the file names in the main.py to the referenced path. 

All the data used for the project is in the Data folder.

Contribution - The collaborators of this project can be found in the repository.
Saavn Beli - Business Understanding, Data Cleaning and Preprocessing, Modeling
Sai Mohith Gandrapu - Exploratory Data Analysis, Data Collection and Mining, Modeling
Sonali Arcot - Business Understanding, Data Transformation, Modeling

project_root/

├──Data/

│   ├── address_data_combined.csv

│   ├── address_data_ethereum.csv

│   ├── address_data_kaggle.csv

│   └── addresses_mined_not_in_kaggle.csv

│──data_collection_and_processing/

│   ├── combining_data.py

│   ├── exploratory_data_analysis.py

│   ├── feature_selection.py

│   └── mining_data.py

├──modeling/

│   ├── base_model_allfeatures.py

│   ├── base_model_extractedfeatures.py

│   ├── evaluation.py

│   └── tuned_model_allfeatures.py


├── main.py

└── readme.md
