import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm import MLE

# Load the dataset
df = pd.read_json('project_data/article/result.json')

# Tokenize the text and create a list of tokenized sentences
tokenized_text = [word_tokenize(sentence.lower()) for sentence in df['text']]

# Preprocess the data
n = 3  # n-gram order
train_data = [list(pad_both_ends(sent, n=n)) for sent in tokenized_text]

# Train the language model
language_model = MLE(n)
language_model.fit(train_data)

# Function to calculate perplexity
def calculate_perplexity(sentence):
    tokenized_sentence = word_tokenize(sentence.lower())
    padded_sentence = list(pad_both_ends(tokenized_sentence, n=n))
    perplexity = language_model.perplexity(padded_sentence)
    return perplexity

# calculate perplexity of each sentence
perplexity = df.map(lambda x: calculate_perplexity(x['text']), axis=1)

# filter articles with top 10% high perplexity
df = df[perplexity > perplexity.quantile(0.9)]

# save as json
df.to_json('project_data/article/filtered_data.json', orient='records', lines=True)
