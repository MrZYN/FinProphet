import pandas as pd

name="test"

# read parquet file
df = pd.read_parquet(f"{name}.parquet")

# keep question, answer, gold evidence column
df = df[['question', 'answer', 'gold_evidence']]

# remove rows with empty answer
df = df[df['answer'].notna()]

# gold evidence is a list of string, concat it
df['gold_evidence'] = df['gold_evidence'].apply(lambda x: ' '.join(x))

# rename columns
df = df.rename(columns={'gold_evidence': 'instruction'})
df = df.rename(columns={'question': 'input'})
df = df.rename(columns={'answer': 'output'})

# save to json with correct format
df.to_json(f'{name}.json', orient='records')

