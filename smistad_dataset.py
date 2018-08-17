import os
from os.path import join
from random import randint
import numpy as np

def create_validation_set(set, subjects_for_validation, exclude=[]):
    validation_set = []
    while len(validation_set) < subjects_for_validation:
        subject = set[randint(0, len(set) - 1)]
        while subject in exclude:
            subject = set[randint(0, len(set) - 1)]

        exclude.append(subject)
        validation_set.append(subject)

    return validation_set


def get_dataset_files(path, exclude=None, only=None):
    filelist = []
    for file in os.listdir(path):
        if file[-3:] != 'hd5':
            continue
        exclude_file = False
        if only:
            exclude_file = True
            for pattern in only:
                if file == str(pattern) + '.hd5':
                    exclude_file = False
                    break
        elif exclude:
            exclude_file = False
            for ex in exclude:
                if file == str(ex) + '.hd5':
                    exclude_file = True
                    break
        if exclude_file:
            continue
        filelist.append(join(path, file))

    return filelist