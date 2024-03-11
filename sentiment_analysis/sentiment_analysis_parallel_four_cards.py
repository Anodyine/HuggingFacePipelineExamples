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
    for i in range(len(classified_results)):
        results[i] = classified_results[i]


quarter_length = math.floor(len(texts)/4)
results1 = [{}] * quarter_length
results2 = [{}] * quarter_length
results3 = [{}] * quarter_length
results4 = [{}] * (quarter_length + 3)

t1 = threading.Thread(target=classify, name='First Thread', args=(texts[0:quarter_length], results1, 0))
t2 = threading.Thread(target=classify, name='Second Thread', args=(texts[quarter_length:quarter_length*2], results2, 1))
t3 = threading.Thread(target=classify, name='Third Thread', args=(texts[quarter_length*2:quarter_length*3], results3, 2))
t4 = threading.Thread(target=classify, name='Fourth Thread', args=(texts[quarter_length*3:len(texts)], results4, 3))

t1.start()
t2.start()
t3.start()
t4.start()

t1.join()
t2.join()
t3.join()
t4.join()

end_time = time.perf_counter()


def print_result(result, statement):
    print("Statement: ", statement, "\nClass: ", result['label'], "\nScore: ", result['score'])


for i in range(10):
    print_result(results4[i], texts[quarter_length*3:len(texts)][i])

print("Ran {0} classifications in {1}s".format((len(results1) + len(results2) + len(results3) + len(results4)).__str__(),
                                                  (end_time - start_time).__str__()))
