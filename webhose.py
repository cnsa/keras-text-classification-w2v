# coding=utf-8
from datetime import datetime

import itertools
import csv

from webhoseio import query, get_next, config
from web_data import data_folder_path
import os

config(token=os.environ['WEBHOSE'])

FOLDER = data_folder_path('cnsa')


def timestamp():
    dt = datetime.utcnow()
    old_dt = dt.replace(month=dt.month - 1)

    def old_date(dt):
        return dt - datetime(1970, 1, 1)

    ms = str(old_date(old_dt).microseconds)[:3]

    return str(int(old_date(old_dt).total_seconds())) + ms


TIMESTAMP = timestamp()


REQUESTS =  dict(enumerate([
    u"Рост числа туристов",
    u"Рост времени пребывания туристов",
    u"Увеличение среднего чека",
    u"Повышение прямой доходности от турпотока",
    u"Увеличение числа коллективных средств размещения",
    u"Создание специализированных шоппинг-центров",
    u"Увеличение числа событийных мероприятий",
    [u"Создание специализированных въездных туроператоров", u"облегченный визовый въезд"],
    u"Повышение качества экскурсионного обслуживания",
    u"Повышение качества навигации и транспортного обслуживания",
    u"Создание туристских кластеров и точек потребления",
    u"PR продвижение Москвы в СМИ и Соцсетях",
    u"Реклама туристских возможностей среди выездных туроператоров за рубежом",
]))

texts = {}


def search(request, quiet=False):
    print(request)

    if quiet is False:
        result = query("filterWebContent", {"ts": TIMESTAMP, "sort": "crawled",
                                          "q": "language:russian text:" + request + " site_type:news"})
        print(result["totalResults"])

        return result


def save():
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    with open(os.path.join(FOLDER, 'webhose.csv'), 'wb') as csvfile:
        fieldnames = ['Title', 'Text', 'Factor1', 'Factor2']
        writer = csv.DictWriter(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_NONNUMERIC,
                                fieldnames=fieldnames)

        writer.writeheader()

        for idx, items in texts.items():
            for text in items:
                writer.writerow(text)


def iter_str(request):
    if isinstance(request, list):
        return itertools.chain(request)
    else:
        return itertools.chain([request])

def uniq_str(string):
    words = string.lower().split()
    return " ".join(sorted(set(words), key=words.index))

def post_obj(post, idx):
    return { "Text": post['text'].encode("utf-8"), "Title": post['title'].encode("utf-8"), "Factor1": idx, "Factor2": None}

#
# for request1, request2 in itertools.permutations(REQUESTS.items(), 2):
#     request_1 = iter_str(request1)
#     request_2 = iter_str(request2)
#
#     for r in request_1:
#         for r2 in request_2:
#             search(uniq_str(r + " " + r2), quiet=False)

for idx, request in REQUESTS.items():
    requests = iter_str(request)
    print(idx)

    for r in requests:
        output = search(uniq_str(r), quiet=False)

        if output["totalResults"] > 0:
            if not texts.has_key(idx):
                texts[idx] = []

            few_texts = []

            for post in output['posts']:
                few_texts.append(post_obj(post, idx))

            texts[idx] = texts[idx] + few_texts



save()
