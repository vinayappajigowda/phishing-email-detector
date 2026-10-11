# Data

This project uses the **Phishing Email Dataset** on Kaggle:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

1. Download it from Kaggle (free account needed) and unzip it.
2. Copy these CSV files into `data/raw/`:
   `CEAS_08.csv`, `Enron.csv`, `Ling.csv`, `Nazario.csv`, `Nigerian_Fraud.csv`, `SpamAssasin.csv`
3. You can skip `phishing_email.csv`. It's a merged copy of the others, and the loader ignores it when the individual files are present.

Label `1` = phishing or malicious email, `0` = legitimate.
The CSVs are not committed to GitHub (see `.gitignore`); link to the Kaggle page instead.
