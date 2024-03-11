from transformers import pipeline

import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import time

initial_time = time.perf_counter()

classifier = pipeline("sentiment-analysis")

statements = [
    "This is such a great movie!",
    "This show was interesting",
    "This show was not interesting",
    "This show was not bad at all",
    "I can't say that this was a good movie"
]

results = classifier(statements)
for i, result in enumerate(results):
    print("Statement: ", statements[i], "\nClass: ", result['label'], "\nScore: ", result['score'])

print("Run Time: ", (time.perf_counter() - initial_time)*1000, "ms")

