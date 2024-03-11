from transformers import pipeline

import numpy as np
import pandas as pd
import seaborn as sn
import torch

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from huggingface_hub import login

from accelerate import init_empty_weights
import time

initial_time = time.perf_counter()
print(torch.cuda.is_available())
print(torch.cuda.current_device())

login("#huggingface token")

summarizer = pipeline("summarization", "facebook/bart-large-cnn", device=0)
print(summarizer("""ome educators draw back from teaching science because they feel unprepared or don't know where to start. They may also feel they do not have the time for science lessons, since science sometimes needs extra explanation, especially during experiments. Some school districts may not have the budget to offer students the lab equipment that they need. Not all educators have a strong background in science, but that does not mean they cannot teach the subject. As with anything else, the more you get involved with a subject the more you will feel confident and ready to teach. Each time you teach a subject, try to learn new things about it yourself as you prepare, and try to think of new ways to present the information or to help the students discover the principles for themselves. The educator is the ultimate role model to children so it is important to show genuine interest in the subject and keep a positive attitude. By doing so it can spark curiosity and increase the joy of learning. You do not need to know all the answers to a question--the willingness to look and explore to find the answers will enhance the learning process. There is not a single way to teach science--every educator has different strengths and weaknesses--but applying learning theories and "best practices" can help you become more effective. Science is universal and can be included in many other subjects including art, music, language arts, math, and more. The main goal of elementary science is to capture the curiosity of young minds, to help them dream of finding new solutions and contributing to society in new ways. Science influences so many aspects of our lives, and the more we learn the more it broadens our perspectives. As educators we have the opportunity to create a base of curiosity, sound thinking, and a scientific framework so our students can become happier and more effective adults.

Here are some ideas to get you thinking about your future classroom and how you can impact the thinkers and inventors of the future:

    Expand interest in all things by being curious and making discoveries together
    Guide and explain basic concepts of science so they can later apply it for future studies
    Teach students to look for and discover new answers through experimentation and measurement
    Help students develop problem-solving skills
    Increase level of science literacy by using scientific language correctly and demonstrating use of critical thinking
    Establish a positive relationship between students and promote cooperative problem solving
    Do experiments that will interest the students and challenge their understandings
    Teach process skills, such as measurement, observation, and presentation of data

"""))

print("Run Time: ", (time.perf_counter() - initial_time)*1000, "ms")