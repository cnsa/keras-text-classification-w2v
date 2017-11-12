# coding=utf-8
from datetime import datetime

import itertools
import csv
import yaml

from webhoseio import query, get_next, config
from web_data import data_folder_path
import os

config(token=os.environ['WEBHOSE'])

FOLDER = data_folder_path('cnsa')

REQUESTS = {}
REQUESTS_PREFIX = ""

with open(FOLDER + 'webhose.yaml', 'r') as stream:
    try:
        data = yaml.load(stream)
        REQUESTS = dict(enumerate(data['queries']))
        REQUESTS_PREFIX = data['query_prefix']
    except yaml.YAMLError as exc:
        print(exc)


def timestamp():
    dt = datetime.utcnow()
    old_dt = dt.replace(month=dt.month - 1)

    def old_date(dt):
        return dt - datetime(1970, 1, 1)

    ms = str(old_date(old_dt).microseconds)[:3]

    return str(int(old_date(old_dt).total_seconds())) + ms


TIMESTAMP = timestamp()

texts = {}


class SearchIterator:
    def __init__(self, request, first_only=False, limit=1000):
        self.request = request + " AND " + REQUESTS_PREFIX
        self.query = None
        self.first_only = first_only
        self.limit = limit
        self.loaded = 0

        print(self.request)

    def __iter__(self):
        return self

    def __is_limit (self):
        if self.first_only is True:
            return True
        if self.limit is 0:
            return False

        return self.loaded > self.limit

    def __update_counter(self, total=None):
        if total is not None and total < 100:
            self.loaded += total
        else:
            self.loaded += 100

    def next(self):  # Python 3: def __next__(self)
        if self.query is None:
            self.query = query("filterWebContent", {"ts": TIMESTAMP, "sort": "relevancy",
                                                    "q": "language:russian text:" + self.request + " site_type:news"})
            print(self.query["totalResults"])
            self.__update_counter(total=self.query["totalResults"])
            return self.query
        elif self.__is_limit() or self.query["moreResultsAvailable"] < 1:
            raise StopIteration
        else:
            self.query = get_next()
            self.__update_counter()
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
        for output in SearchIterator(r, first_only=False):

            if output["totalResults"] > 0:
                for post in output['posts']:
                    add_text(post['uuid'], post_obj(post, str(idx)))

for idx1, idx2 in itertools.combinations(REQUESTS.keys(), 2):
    request_1 = iter_str(REQUESTS[idx1])
    request_2 = iter_str(REQUESTS[idx2])

    for r in request_1:
        for r2 in request_2:
            print(str(idx1) + " " + str(idx2))
            for output in SearchIterator("(" +r + ") AND (" + r2 + ")", first_only=False):

                if output["totalResults"] > 0:
                    for post in output['posts']:
                        add_text(post['uuid'], post_obj(post, str(idx1), str(idx2)))

save()
