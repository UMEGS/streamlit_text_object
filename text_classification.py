from transformers import pipeline
import time


def make_response(res, time_taken):
    pre = {}

    if res[0]['label'] == 'NEGATIVE':
        pre['POSITIVE'] = 1 - round(res[0]['score'], 2)
        pre['NEGATIVE'] = round(res[0]['score'], 2)

    if res[0]['label'] == 'POSITIVE':
        pre['POSITIVE'] = round(res[0]['score'], 2)
        pre['NEGATIVE'] = 1 - round(res[0]['score'], 2)

    pre['time_taken'] = round(time_taken, 2)

    return pre


def classifications(text):
    classifier = pipeline("sentiment-analysis")

    start_time = time.time()

    res = classifier(text)
    time_taken = (time.time() - start_time) * 1000

    res = make_response(res, time_taken)

    return res
