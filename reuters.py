#!/usr/bin/env python

from __future__ import print_function

import os.path
import tarfile

import re
import xml.sax.saxutils as saxutils
from urllib import urlretrieve

import sys
from bs4 import BeautifulSoup
from pandas import DataFrame

from data_helper import update_frequencies, to_category_vector


def _not_in_sphinx():
    # Hack to detect whether we are running by the sphinx builder
    return '__file__' in globals()


###############################################################################
# Reuters Dataset related routines
# --------------------------------
#

class ReutersParser():
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, category_data, columns=None, selected_categories=None):
        if selected_categories is None:
            selected_categories = ['pl_usa', 'to_cocoa', 'pl_uruguay']

        self.selected_categories = selected_categories

        self.sgml_number_of_files = 22
        self.sgml_file_name_template = 'reut2-NNN.sgm'

        if columns is None:
            columns = ['Name', 'Type', 'Newslines']

        self.columns = columns

        self.news_categories = DataFrame(data=category_data, columns=columns)

    def strip_tags(self, text):
        return re.sub('<[^<]+?>', '', text).strip()

    def unescape(self, text):
        return saxutils.unescape(text)

    def parse(self, newsline=None):
        if newsline is None:
            return None

        document_categories = []

        # News-line Id
        document_id = newsline['newid']

        # News-line text
        document_body = self.strip_tags(str(newsline('text')[0])).replace('reuter\n&#3;', '')
        document_body = self.unescape(document_body)

        # News-line categories
        topics = newsline.topics.contents
        places = newsline.places.contents
        people = newsline.people.contents
        orgs = newsline.orgs.contents
        exchanges = newsline.exchanges.contents

        for topic in topics:
            document_categories.append('to_' + self.strip_tags(str(topic)))

        for place in places:
            document_categories.append('pl_' + self.strip_tags(str(place)))

        for person in people:
            document_categories.append('pe_' + self.strip_tags(str(person)))

        for org in orgs:
            document_categories.append('or_' + self.strip_tags(str(org)))

        for exchange in exchanges:
            document_categories.append('ex_' + self.strip_tags(str(exchange)))

        # Create new document
        self.news_categories = update_frequencies(self.news_categories, document_categories)

        return document_body, to_category_vector(document_categories, self.selected_categories), document_id


def stream_reuters_documents(category_data, columns=None, selected_categories=None, data_folder=None):
    if data_folder is None:
        data_folder = data_folder_path(data_folder)

    download_reuters_documents(data_folder)

    # Parse SGML files
    document_X = {}
    document_Y = {}

    parser = ReutersParser(category_data, columns=columns, selected_categories=selected_categories)

    # Iterate all files
    for i in range(parser.sgml_number_of_files):
        if i < 10:
            seq = '00' + str(i)
        else:
            seq = '0' + str(i)

        file_name = parser.sgml_file_name_template.replace('NNN', seq)
        print('Reading file: %s' % file_name)

        with open(data_folder + file_name, 'r') as file:
            content = BeautifulSoup(file.read().lower(), "lxml")

            for newsline in content('reuters'):
                X, Y, document_id = parser.parse(newsline)
                document_X[document_id] = X
                document_Y[document_id] = Y

    parser.news_categories.sort(columns='Newslines', ascending=False, inplace=True)
    parser.news_categories.head(20)

    return document_X, document_Y, parser.news_categories


def data_folder_path(data_folder=None):
    if data_folder is None:
        pathname = os.path.dirname(sys.argv[0])
        data_folder = os.path.join(pathname, 'data', 'reuters21578/')

    return data_folder


def get_category_data(data_folder=None):
    if data_folder is None:
        data_folder = data_folder_path()

    download_reuters_documents(data_folder)

    # Category files
    category_files = {
        'to_': ('Topics', 'all-topics-strings.lc.txt'),
        'pl_': ('Places', 'all-places-strings.lc.txt'),
        'pe_': ('People', 'all-people-strings.lc.txt'),
        'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
        'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
    }

    # Read all categories
    category_data = []

    for category_prefix in category_files.keys():
        with open(data_folder + category_files[category_prefix][1], 'r') as file:
            for category in file.readlines():
                category_data.append([category_prefix + category.strip().lower(),
                                      category_files[category_prefix][0],
                                      0])

    return category_data


def download_reuters_documents(data_path=None):
    """The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    """
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'
    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/' + ARCHIVE_FILENAME)

    if data_path is None:
        data_path = data_folder_path(data_path)

    if not os.path.exists(data_path):
        os.makedirs(data_path, mode=0o755)

    check_path = os.path.join(data_path, 'reut2-000.sgm')

    if not os.path.exists(check_path):
        """Download the dataset."""
        print("downloading dataset (once and for all) into %s" %
              data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb),
                      end='')

        archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
        urlretrieve(DOWNLOAD_URL, filename=archive_path,
                    reporthook=progress)
        if _not_in_sphinx():
            print('\r', end='')
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)
        print("done.")

    return True
