# coding: utf-8
import sys
import os
sys.path.append('..')
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')
import pickle
import numpy as np

# ★このURLはGitHubにアップしてある分かち書き済みの青空文庫作品詰め合わせのダウンロードURL
#  詳細は https://github.com/segavvy/wakatigaki-aozorabunko を参照してください。
url_base = 'https://raw.githubusercontent.com/segavvy/wakatigaki-aozorabunko/master/'
key_file = {
    'train': '20200516merge.txt',
    'test': '',  # ★まだ使わないので準備していない
    'valid': ''  # ★まだ使わないので準備していない
}
save_file = {
    'train': 'aozorabunko.train.npy',
    'test': 'aozorabunko.test.npy',
    'valid': 'aozorabunko.valid.npy'
}
vocab_file = 'aozorabunko.vocab.pkl'

dataset_dir = os.path.dirname(os.path.abspath(__file__))


def _download(file_name):
    file_path = dataset_dir + '/' + file_name
    if os.path.exists(file_path):
        return

    print('Downloading ' + file_name + ' ... ')

    try:
        urllib.request.urlretrieve(url_base + file_name, file_path)
    except urllib.error.URLError: # type: ignore
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + file_name, file_path)

    print('Done')


# ★テキストの分割が2箇所で使われるため関数化。実装は超その場しのぎ……
def _split_data(text):
    return text.replace('\n', '<eos> ').replace('。', '<eos> ').strip().split()


def load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    data_type = 'train'
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name

    _download(file_name)

    words = _split_data(open(file_path).read())

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(data_type='train'):
    '''
        :param data_type: データの種類：'train' or 'test' or 'valid (val)'
        :return:
    '''
    if data_type == 'val': data_type = 'valid'
    save_path = dataset_dir + '/' + save_file[data_type]

    word_to_id, id_to_word = load_vocab()

    if os.path.exists(save_path):
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    _download(file_name)

    words = _split_data(open(file_path).read())
    corpus = np.array([word_to_id[w] for w in words])

    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word


if __name__ == '__main__':
    for data_type in ('train', 'val', 'test'):
        load_data(data_type)

