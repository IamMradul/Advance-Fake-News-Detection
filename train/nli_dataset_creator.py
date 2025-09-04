import pandas as pd

true_df = pd.read_csv('D:/Project/Hackathon/dataset/True.csv')
fake_df = pd.read_csv('D:/Project/Hackathon/dataset/Fake.csv')
true_df['label'] = 1  # entailment (real)
fake_df['label'] = 0  # contradiction (fake)

# Use title as claim (premise), text as evidence (hypothesis)
true_df['premise'] = true_df['title']
true_df['hypothesis'] = true_df['text']
fake_df['premise'] = fake_df['title']
fake_df['hypothesis'] = fake_df['text']

nli_df = pd.concat([true_df[['premise', 'hypothesis', 'label']], fake_df[['premise', 'hypothesis', 'label']]], ignore_index=True)
nli_df.to_csv('D:/Project/Hackathon/dataset/nli_pairs.csv', index=False)