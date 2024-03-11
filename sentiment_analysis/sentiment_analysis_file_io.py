from transformers import pipeline
import threading

import numpy as np
import pandas as pd

import time
import matplotlib.pyplot as plt

df_ = pd.read_csv('AirlineTweets.csv')
df = df_[['airline_sentiment', 'text']].copy()
df['airline_sentiment'].hist()

df = df[df['airline_sentiment'] != 'neutral']  # remove neutral statements

target_map = {'positive': 1, 'negative': 0}
df['target'] = df['airline_sentiment'].map(target_map)

start_time = time.perf_counter()
classifier = pipeline("sentiment-analysis",  device_map=3)
texts = df['text'].tolist()
predictions = classifier(texts)
end_time = time.perf_counter()


def print_result(result, statement):
    print("Statement: ", statement, "\nClass: ", result['label'], "\nScore: ", result['score'])


for i in range(10):
    print_result(predictions[i], texts[i])

print("Ran ", len(predictions), " classifications in ", (end_time - start_time), "s")
