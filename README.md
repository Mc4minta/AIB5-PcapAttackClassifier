# AIB5-PcapAttackClassifier

Medium Blog : https://medium.com/@mc4minta/aib2025-malicous-pcap-file-classifier-8014e98f8d02

Hugging Face Model : https://huggingface.co/Mc4minta/RandomForest400IntPortCIC1718/tree/main

# Deployment - Streamlit Framework
- only suport wsl/linux for now
## Jupyter Notebook ([demo.ipynb](https://github.com/Mc4minta/AIB5-PcapAttackClassifier/blob/main/demo.ipynb))
- Setup google api key via Secrets function in google colab
- Cell Descriptions
  - Dependencies setup : install dependencies
  - src/*.py setup : create or python file in src
  - main.py : create main.py
  - Demo Link below this cell (Localtunnel) : Run this cell to deploy run streamlit and use localtunnel no generate link
- use the public ip address printed by the program as a password to access local tunnel website  
## Local Run [Github Release](https://github.com/Mc4minta/AIB5-PcapAttackClassifier/releases/tag/AIB5-PcapAttackClassifier-1.0)
- Download and unzip release file
- Install requirements
```
pip install -r requirements.txt
```
- Setup google api key
```
export GOOGLE_API_KEY='API_KEY'
```
- Start streamlit program (main.py)
```
python -m streamlit run main.py
```
- (Optional) Clean up environment after run
```
python cleanup.py
```


