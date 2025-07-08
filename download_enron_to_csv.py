import os
import pandas as pd
from datasets import load_dataset

def main():
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    print('Loading ENRON-spam dataset from HuggingFace...')
    ds = load_dataset('bvk/ENRON-spam')
    print('Converting to DataFrame...')
    df = pd.DataFrame({
        'email_content': ds['train']['email'],
        'label': ds['train']['label']
    })
    out_path = 'data/spam_data.csv'
    df.to_csv(out_path, index=False)
    print(f'Saved ENRON dataset to {out_path}')

if __name__ == '__main__':
    main() 