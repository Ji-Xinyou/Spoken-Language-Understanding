# The data augmentations
import json
import numpy as np
import copy
import random
import os

factor = 10

# open lexicons
poi_path = os.path.join('data', 'lexicon', 'poi_name.txt')
ordinal_path = os.path.join('data', 'lexicon', 'ordinal_number.txt')
op_path = os.path.join('data', 'lexicon', 'operation_verb.txt')
train_path = os.path.join('data', 'train.json')

# mark before aug
train = json.load(open(train_path, 'r', encoding='utf-8'))
prev_size = len(train)

# make lexicon a list
ords = open(ordinal_path, 'r', encoding='utf-8').readlines()
ords = [ord.strip() for ord in ords]
ops = open(op_path, 'r', encoding='utf-8').readlines()
ops = [op.strip() for op in ops]
pois = open(poi_path, 'r', encoding='utf-8').readlines()
pois = [poi.strip() for poi in pois]

def setup_new(utt, ss, asr, manual, s, replace):
    new = copy.deepcopy(utt)
    new['manual_transcript'] = manual.replace(s[2], replace)
    new['asr_1best'] = asr.replace(s[2], replace)
    new['semantic'] = [[s[0], s[1], replace] if i == s else i for i in ss]
    return new

# This augmented the poi name
#
# arg_factor is the number of size you want to extend
# e.g. aug_factor = 100 <=> 100->10000
def aug1(train, _ords, _ops, pois, aug_factor):
    aug = copy.deepcopy(train)
    while len(aug) / len(train) < aug_factor:
        for utts in train:
            for utt in utts:
                _id = utt['utt_id']
                ss = utt['semantic']
                asr = utt['asr_1best']
                manual = utt['manual_transcript']

                for s in ss:
                    option1 = 'poi名称'
                    option2 = 'poi修饰'
                    option3 = 'poi目标'
                    option4 = '起点名称'
                    option5 = '起点修饰'
                    option6 = '起点目标'
                    option7 = '终点名称'
                    option8 = '终点修饰'
                    option9 = '终点目标'
                    option10 = '途经点名称'
                    if s[1] in [option1, option2, option3, option4, option5, option6, option7, option8, option9, option10]:
                        replace = random.choice(pois)
                        new = setup_new(utt, ss, asr, manual, s, replace)
                        aug.append([new])
    return aug

ontology = json.load(open(os.path.join('data', 'ontology.json'), 'r', encoding='utf-8'))

# this augments the slot value
def aug2(train, ords, _ops, _pois, aug_factor):
    aug = copy.deepcopy(train)
    while len(aug) / len(train) < aug_factor:
        for utts in train:
            for utt in utts:
                _id = utt['utt_id']
                ss = utt['semantic']
                asr = utt['asr_1best']
                manual = utt['manual_transcript']

                for s in ss:
                    if s[1] == '请求类型':
                        replace = random.choice(ontology['slots']['请求类型'])
                        new = setup_new(utt, ss, asr, manual, s, replace)
                        aug.append([new])
                    elif s[1] == '路线偏好':
                        replace = random.choice(ontology['slots']['路线偏好'])
                        new = setup_new(utt, ss, asr, manual, s, replace)
                        aug.append([new])
                    elif s[1] == '对象':
                        replace = random.choice(ontology['slots']['对象'])
                        new = setup_new(utt, ss, asr, manual, s, replace)
                        aug.append([new])
                    elif s[1] == '页码':
                        replace = '下一页' if s[2] == '上一页' else '上一页'
                        new = setup_new(utt, ss, asr, manual, s, replace)
                        aug.append([new])
    return aug

def aug3(train, ords, ops, _pois, aug_factor):
    aug = copy.deepcopy(train)
    while len(aug) / len(train) < aug_factor:
        for utts in train:
            for utt in utts:
                _id = utt['utt_id']
                ss = utt['semantic']
                asr = utt['asr_1best']
                manual = utt['manual_transcript']

                for s in ss:
                    if s[1] == '操作':
                        replace = random.choice(ops)
                        new = setup_new(utt, ss, asr, manual, s, replace)
                        aug.append([new])
                    elif s[1] == '序列号':
                        replace = random.choice(ords)
                        new = setup_new(utt, ss, asr, manual, s, replace)
                        aug.append([new])
                        
    return aug

aug_train = copy.deepcopy(train)
print("augmenting on poi names")
aug_train.extend(aug1(train, ords, ops, pois, factor))
print("augmenting on slot values")
aug_train.extend(aug2(train, ords, ops, pois, factor))
print("augmenting on ordinal and operation")
aug_train.extend(aug3(train, ords, ops, pois, factor))
print(f"augmentation done, length {len(train)} -> {len(aug_train)}")

print("dumping augmented data")
filename = f'aug_train_{factor}.json'
with open(os.path.join('data', filename), 'w', encoding='utf-8') as f:
    json.dump(aug_train, f, ensure_ascii=False)
