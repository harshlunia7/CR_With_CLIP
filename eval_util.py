import torch
import os
import shutil
import pickle as pkl
import torch.nn as nn
import copy

from data.lmdbReader import lmdbDataset, resizeNormalize
from eval_config import config
from shutil import copyfile

mse_loss = nn.MSELoss()
alphabet_character_file = open(config['alpha_path'], 'r', encoding="utf-8")
alphabet_character = list(alphabet_character_file.read().strip())
alphabet_character_raw = ['START']

for item in alphabet_character:
    alphabet_character_raw.append(item)

alphabet_character_raw.append('END')
alphabet_character = alphabet_character_raw

alp2num_character = {}

for index, char in enumerate(alphabet_character):
    alp2num_character[char] = index

r2num = {}
alphabet_radical = []
alphabet_radical.append('PAD')
lines = open(config['radical_path'], 'r', encoding="utf-8").readlines()
for line in lines:
    alphabet_radical.append(line.strip('\n'))
alphabet_radical.append('$')
for index, char in enumerate(alphabet_radical):
    r2num[char] = index

dict_file = open(config['decompose_path'], 'r', encoding="utf-8").readlines()
char_radical_dict = {}
for line in dict_file:
    line = line.strip('\n')
    try:
        char, r_s = line.split(':')
    except:
        char, r_s = ':', ':'
    char_radical_dict[char] = list(''.join(r_s.split(' ')))

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)
    
    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys
    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))
    
    return model
    

def load_vit_encoder_weights(model, encoder_checkpoint_path):
    checkpoint_model =  torch.load(encoder_checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint_model['model']
    return load_state_dict(model, checkpoint_model, prefix='')

def get_dataloader(root,shuffle=False):
    if root.endswith('pkl'):
        f = open(root,'rb')
        dataset = pkl.load(f)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], shuffle=shuffle, num_workers=4,
        )
    else:
        dataset = lmdbDataset(root,resizeNormalize((config['imageW'],config['imageH'])))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], shuffle=shuffle, num_workers=4,
        )
    return dataloader, dataset


def get_data_package():
    test_dataset = []
    test_dataloader = None
    if len(config['test_dataset']) > 0:
        for dataset_root in config['test_dataset'].split(','):
            _ , dataset = get_dataloader(dataset_root,shuffle=True)
            test_dataset.append(dataset)
        test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=4,
        )

    return test_dataloader

def convert_char(label):
    r_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])
        r_tmp.append('$')
        r_label.append(r_tmp)

    text_tensor = torch.zeros(batch, 30).long().cuda()
    for i in range(batch):
        tmp = r_label[i]
        for j in range(len(tmp)):
            text_tensor[i][j] = r2num[tmp[j]]
    return text_tensor

def get_radical_alphabet():
    return alphabet_radical

def converter(label):

    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            text_input[i][j + 1] = alp2num[label[i][j]]

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])

    return length, text_input, text_all, string_label

def get_alphabet():
    return alphabet_character

def tensor2str(tensor):
    alphabet = get_alphabet()
    string = ""
    for i in tensor:
        if i == (len(alphabet)-1):
            continue
        string += alphabet[i]
    return string

def must_in_screen():
    text = os.popen('echo $STY').readlines()
    string = ''
    for line in text:
        string += line
    if len(string.strip()) == 0:
        print("run in the screen!")
        exit(0)

def saver():
    try:
        shutil.rmtree('./history/{}'.format(config['exp_name']))
    except:
        pass
    os.makedirs(f"./history/{config['exp_name']}", exist_ok=True)

    import time

    print('**** Experiment Name: {} ****'.format(config['exp_name']))

    localtime = time.asctime(time.localtime(time.time()))
    f = open(os.path.join('./history', config['exp_name'], str(localtime)),'w+')
    f.close()

