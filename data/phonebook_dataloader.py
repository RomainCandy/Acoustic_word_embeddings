import numpy as np
import random
import shelve
import torch
import warnings
from torch.utils.data import Dataset

from .utils import random_padding
from .utils import transform, choose_words, get_all_word


class DataTriplet(Dataset):
    def __init__(self, mod, problem, load_file, all_words, nb_examples, transf=transform, lstm=False):
        """
        Args:
            mod: either train val or test
            problem: words or speakers
            load_file: a shelve file which contains all the acoustic features of word or speaker on each mod
                       with the keys define as "problem"
            all_words: file word:list_word
            nb_examples: nb of iteration for each epochs
            transf: other pre-processing step
            lstm: if lstm chose the acoustic features with no 0 padding.
        """
        assert mod in {'train', 'val', 'test'}
        assert problem in {'words', 'speakers'}
        if mod in {'val', 'test'}:
            warnings.warn(
                "{} is mostly for debugging not for training".format(mod),
                SyntaxWarning
            )
        self.problem = problem
        self.nb_examples = nb_examples
        self.mod = mod
        self.load_file = load_file
        self.transform = transf

        self.all_words = get_all_word(all_words)

        with shelve.open(load_file) as db:
            if lstm:
                self.loader = db['lstm_{}'.format(mod)]

            elif self.problem == 'speakers':
                self.loader = db['speakers_{}'.format(mod)]

            else:
                self.loader = db[mod]

        self.words = sorted(list(self.all_words.keys()))
        self.words_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_words = {i: w for w, i in self.words_to_idx.items()}
        self.probability_matrix = np.ones((len(self.words), len(self.words)))
        np.fill_diagonal(self.probability_matrix, 0)

    def choose_anchor(self):
        word = random.choice(self.words)
        anchor, posifif = random.sample(self.all_words[word], 2)
        return self.words_to_idx[word], anchor, posifif

    def update(self, y_scores, idxa, idxn):
        for ida, idn, score in zip(idxa, idxn, y_scores):
            self.probability_matrix[int(ida.item()), int(idn.item())] += score.item()
            self.probability_matrix[int(idn.item()), int(ida.item())] += score.item()

    def reset(self):
        self.probability_matrix = np.ones((len(self.words), len(self.words)))
        np.fill_diagonal(self.probability_matrix, 0)

    def choose_negatif(self, word_idx):
        row = self.probability_matrix[word_idx].clip(0) + 1e-4
        wordn_idx = torch.multinomial(torch.from_numpy(row), 1).item()
        wordn = self.idx_to_words[wordn_idx]
        return wordn_idx, choose_words(self.all_words, wordn)

    def __getitem__(self, index):
        worda, wa, wp = self.choose_anchor()
        wordn, wn = self.choose_negatif(worda)
        a = self.loader[wa]
        p = self.loader[wp]
        n = self.loader[wn]
        if self.transform is not None:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)
        return (worda, a, p), (wordn, n)

    def __len__(self):
        return self.nb_examples


class DataSiamese(Dataset):
    def __init__(self, mod, data_file, load_file, problem, transf=transform):
        """
        Args:
            mod: either train val or test
            problem: words or speakers
            load_file: a shelve file which contains all the acoustic features of word or speaker on each mod
                       with the keys define as "problem"
            transf: other pre-processing step
        """
        assert mod in {'train', 'val', 'test'}
        assert problem in {'words', 'speakers', 'both'}
        self.mod = mod
        self.problem = problem
        self.data_file = data_file
        self.load_file = load_file
        self.transform = transf

        with open(data_file) as lines:
            self.file = [line.strip().split(',') for line in lines.readlines()][1:]

        with shelve.open(load_file) as db:
            if self.problem == 'speakers':
                self.loader = db['speakers_{}'.format(mod)]

            else:
                self.loader = db[mod]

    def __getitem__(self, index):

        wa, wp, problem = self.file[index]
        a = self.loader[wa]
        p = self.loader[wp]
        if self.transform is not None:
            a = self.transform(a)
            p = self.transform(p)
        return a, p, torch.LongTensor([int(problem)])

    def __len__(self):
        return len(self.file)


class DataAll(Dataset):
    def __init__(self, mod, problem, load_file, transf=transform):
        assert mod in {'train', 'val', 'test'}
        assert problem in {'words', 'speakers'}
        self.problem = problem

        self.mod = mod
        self.load_file = load_file
        self.transform = transf

        with shelve.open(load_file) as db:
            if self.problem == 'speakers':
                self.loader = db['speakers_{}'.format(mod)]
            else:
                self.loader = db[mod]

        self.files = tuple(self.loader.keys())

    def __getitem__(self, index):

        file = self.files[index]
        inp = self.loader[file]
        if self.transform is not None:
            inp = self.transform(inp)
        return inp, file

    def __len__(self):
        return len(self.files)


class DataTripletRandomPad(Dataset):
    def __init__(self, mod, problem, load_file, all_words, nb_examples, width_final=300, transf=transform):
        # all_words {word:[files]}
        assert mod in {'train', 'val', 'test'}
        assert problem in {'words', 'speakers'}
        if mod in {'val', 'test'}:
            warnings.warn(
                "{} is mostly for debugging not for training".format(mod),
                SyntaxWarning
            )
        self.problem = problem
        self.width_final = width_final
        self.nb_examples = nb_examples
        self.mod = mod
        self.load_file = load_file
        self.transform = transf

        self.all_words = get_all_word(all_words)

        with shelve.open(load_file) as db:
            if self.problem == 'speakers':
                self.loader = db['lstm_speakers_{}'.format(mod)]
            else:
                self.loader = db['lstm_{}'.format(mod)]

        self.words = sorted(list(self.all_words.keys()))
        self.words_to_idx = {w: i for i, w in enumerate(self.words)}
        self.idx_to_words = {i: w for w, i in self.words_to_idx.items()}
        self.probability_matrix = np.ones((len(self.words), len(self.words)))
        np.fill_diagonal(self.probability_matrix, 0)

    def choose_anchor(self):
        word = random.choice(self.words)
        anchor, posifif = random.sample(self.all_words[word], 2)
        return self.words_to_idx[word], anchor, posifif

    def update(self, y_scores, idxa, idxn):
        for ida, idn, score in zip(idxa, idxn, y_scores):
            self.probability_matrix[int(ida.item()), int(idn.item())] += score.item()
            self.probability_matrix[int(idn.item()), int(ida.item())] += score.item()

    def reset(self):
        self.probability_matrix = np.ones((len(self.words), len(self.words)))
        np.fill_diagonal(self.probability_matrix, 0)

    def choose_negatif(self, word_idx):
        row = self.probability_matrix[word_idx].clip(0) + 1e-4
        wordn_idx = torch.multinomial(torch.from_numpy(row), 1).item()
        wordn = self.idx_to_words[wordn_idx]
        return wordn_idx, choose_words(self.all_words, wordn)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        worda, wa, wp = self.choose_anchor()
        wordn, wn = self.choose_negatif(worda)
        a = random_padding(self.loader[wa], self.width_final)
        p = random_padding(self.loader[wp], self.width_final)
        n = random_padding(self.loader[wn], self.width_final)
        if self.transform is not None:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)
        return (worda, a, p), (wordn, n)

    def __len__(self):
        return self.nb_examples


class Data(Dataset):
    def __init__(self, mod, data_file, load_file, problem, is_8k=False, transf=transform):
        assert mod in {'train', 'val', 'test'}
        assert problem in {'words', 'speakers'}
        self.problem = problem
        self.mod = mod
        self.data_file = data_file
        self.load_file = load_file
        self.transform = transf

        with open(data_file) as lines:
            self.file = [line.strip().split(',') for line in lines.readlines()][1:]

        with shelve.open(load_file) as db:
            if is_8k:
                self.loader = db['{}_CNN'.format(mod)]
                self.file = [['/'.join(x.split('/')[-2:]) for x in y] for y in self.file]
            elif problem == 'speakers':
                self.loader = db['{}_{}'.format(problem, mod)]
            else:
                self.loader = db[mod]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        wa, wp, wn = self.file[index]
        a = self.loader[wa]
        p = self.loader[wp]
        n = self.loader[wn]
        if self.transform is not None:
            a = self.transform(a)
            p = self.transform(p)
            n = self.transform(n)

        return a, p, n

    def __len__(self):
        return len(self.file)
