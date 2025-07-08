import pandas as pd
import os
import subprocess

def main():
    # Sample 5000 emails from the full dataset
    df = pd.read_csv('data/spam_data.csv')
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    out_path = 'data/spam_data_sample.csv'
    df_sample.to_csv(out_path, index=False)
    print(f'Saved sample of {sample_size} emails to {out_path}')

    # Update ModelTrainer to use the sample file
    # (Assumes ModelTrainer uses data_path argument)
    print('Retraining model on sample...')
    subprocess.run(['python', '-m', 'models.model_trainer', '--data_path', out_path])

if __name__ == '__main__':
    main() 