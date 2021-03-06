import argparse

from evaluations import main_f1_ap, main_tsne, main_hist


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("kind", help="triplet or siamese network", choices=['triplet', 'siamese'])
    parser.add_argument("version", help="choose configuration version")
    parser.add_argument("file_words", help="file where the model is saved")
    parser.add_argument('res', help='if main get ap, auc and f1;'
                                    'if hist get hist of pos and neg distances;'
                                    'if tsne get tsne reduction', choices=['main', 'hist', 'tsne'])
    parser.add_argument("--mod", help="train, val or test", choices=['train', 'val', 'test'], default='test')
    parser.add_argument("--pb", help="words or speakers", choices=['words', 'speakers'], default='words')

    parser.add_argument("--margin", help="threshold for computing the metrics", type=float, default=.4)

    parser.add_argument("--keep", help="keep how many examples for tsne", type=int, default=20)

    parser.add_argument("--path_to_save", help="Where to save plots if nothing just show plots", default='')

    args = parser.parse_args()
    mod = args.mod
    problem = args.pb
    version = args.version
    file_words = args.file_words
    path_to_save = args.path_to_save
    keep = args.keep
    res = args.res
    triplet = args.kind == 'triplet'
    margin = args.margin
    if triplet:
        config_file = 'configuration_files/triplet_phonebook.ini'
    else:
        config_file = 'configuration_files/siamese_phonebook.ini'
    lab_to_word = 'data/phonebook_lab_to_words.csv'
    if res == 'main':
        main_f1_ap(mod=mod, problem=problem, config_file=config_file, version=version, file_words=file_words,
                   triplet=triplet, lab_to_word=lab_to_word, path_to_save=path_to_save, margin=margin, n_keep_neg=keep)

    elif res == 'hist':
        main_hist(mod=mod, problem=problem, config_file=config_file, version=version, file_words=file_words,
                  triplet=triplet, lab_to_word=lab_to_word, path_to_save=path_to_save, n_keep_neg=keep)

    else:
        main_tsne(mod=mod, problem=problem, config_file=config_file, version=version, file_words=file_words,
                  triplet=triplet, lab_to_word=lab_to_word, path_to_save=path_to_save, n_keep=keep)


if __name__ == "__main__":
    main()
