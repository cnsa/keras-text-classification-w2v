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

REQUESTS = dict(enumerate([
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


class SearchIterator:
    def __init__(self, request):
        self.request = request
        self.query = None

        print(request)

    def __iter__(self):
        return self

    def next(self):  # Python 3: def __next__(self)
        if self.query is None:
            self.query = query("filterWebContent", {"ts": TIMESTAMP, "sort": "crawled",
                                                    "q": "language:russian text:" + self.request + " site_type:news"})
            print(self.query["totalResults"])
            return self.query
        elif self.query["moreResultsAvailable"] < 1:
            raise StopIteration
        else:
            self.query = get_next()
            return self.query


def save():
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    with open(os.path.join(FOLDER, 'webhose.csv'), 'wb') as csvfile:
        fieldnames = ['Title', 'Text', "UUID", "Date", 'Factor1', 'Factor2']
        writer = csv.DictWriter(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_NONNUMERIC,
                                fieldnames=fieldnames)

        writer.writeheader()

        for uuid, text in texts.items():
            writer.writerow(text)


def iter_str(request):
    if isinstance(request, list):
        return itertools.chain(request)
    else:
        return itertools.chain([request])


def uniq_str(string):
    words = string.lower().split()
    return " ".join(sorted(set(words), key=words.index))


def post_obj(post, idx, idx2=""):
    return {"Text": post['text'].encode("utf-8"),
            "Title": post['title'].encode("utf-8"),
            "Date": post['published'],
            "UUID": post['uuid'],
            "Factor1": idx,
            "Factor2": idx2}


def factor_to_factor(text, old_text, old_factor2, num="1"):
    if (text['Factor' + num] not in old_text['Factor1']) and (text['Factor' + num] not in old_factor2):
        old_factor2.append(text['Factor' + num])
    return old_factor2


def add_text(uuid, text):
    if texts.has_key(uuid):
        old_text = texts[uuid]
        old_factor2 = old_text['Factor2'].split(';')
        if old_factor2[0] is "":
            old_factor2 = []

        old_factor2 = factor_to_factor(text, old_text, old_factor2, "1")
        old_factor2 = factor_to_factor(text, old_text, old_factor2, "2")
        old_text['Factor2'] = ";".join(old_factor2)
        texts[uuid] = old_text
    else:
        texts[uuid] = text


for idx, request in REQUESTS.items():
    requests = iter_str(request)
    print(idx)

    for r in requests:
        for output in SearchIterator(uniq_str(r)):

            if output["totalResults"] > 0:
                for post in output['posts']:
                    add_text(post['uuid'], post_obj(post, str(idx)))

for idx1, idx2 in itertools.combinations(REQUESTS.keys(), 2):
    request_1 = iter_str(REQUESTS[idx1])
    request_2 = iter_str(REQUESTS[idx2])

    for r in request_1:
        for r2 in request_2:
            for output in SearchIterator(uniq_str(r + " " + r2)):

                if output["totalResults"] > 0:
                    for post in output['posts']:
                        add_text(post['uuid'], post_obj(post, str(idx1), str(idx2)))

save()
