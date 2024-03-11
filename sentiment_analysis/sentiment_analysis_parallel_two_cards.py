from transformers import pipeline
import threading

import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt

df_ = pd.read_csv('AirlineTweets.csv')
df = df_[['airline_sentiment', 'text']].copy()
df['airline_sentiment'].hist()

df = df[df['airline_sentiment'] != 'neutral']  # remove neutral statements

target_map = {'positive': 1, 'negative': 0}
df['target'] = df['airline_sentiment'].map(target_map)
texts = df['text'].tolist()

start_time = time.perf_counter()


def classify(tweets, results, device):
    classifier = pipeline("sentiment-analysis", device_map=device)
    classified_results = classifier(tweets)
    for j in range(len(classified_results)):
        results[j] = classified_results[j]


half_length = math.floor(len(texts)/2)
results1 = [{}] * half_length
results2 = [{}] * (half_length + 1)

t1 = threading.Thread(target=classify, name='First Thread', args=(texts[0:half_length], results1, 0))
t2 = threading.Thread(target=classify, name='Second Thread', args=(texts[half_length:len(texts)], results2, 1))

t1.start()
t2.start()

t1.join()
t2.join()

end_time = time.perf_counter()


def print_result(result, statement):
    print("Statement: ", statement, "\nClass: ", result['label'], "\nScore: ", result['score'])


for i in range(10):
    print_result(results2[i], texts[half_length:len(texts)][i])

print("Ran {0} classifications in {1}s".format((len(results1) + len(results2)).__str__(),
                                               (end_time - start_time).__str__()))
