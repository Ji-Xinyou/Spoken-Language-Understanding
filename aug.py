# The data augmentations
import json
import copy
import random
import os
import argparse

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
    aug = []
    print()
    while (len(aug) / len(train)) < aug_factor:
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
    aug = []
    while (len(aug) / len(train)) < aug_factor:
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

# This augments the ordinal number and operation
def aug3(train, ords, ops, _pois, aug_factor):
    aug = []
    while (len(aug) / len(train)) < aug_factor:
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

# DEPRECATED: This augments by generating simple datas
def badbadbadaug4(train, ords, ops, pois, aug_factor):
    aug = copy.deepcopy(train)
    while len(aug) / len(train) < aug_factor:
        poi = random.choice(pois)
        patt1 = f"我想去{poi}"
        patt2 = f"带我去{poi}"
        patt3 = f"去{poi}怎么走"
        patt4 = f"{poi}怎么走"
        patt = random.choice([patt1, patt2, patt3, patt4])
        utt = {"utt_id": 1,
            "manual_transcript": patt,
            "asr_1best": patt,
            "semantic": [['inform', '终点名称', poi]]
        }
        aug.append([utt])
        utt = {"utt_id": 1,
                "manual_transcript": poi,
                "asr_1best": poi,
                "semantic": [['inform', 'poi名称', poi]]
            }
        aug.append([utt])
    return aug

noise_cn = ["一", "个", "生", "到", "导", "去", "二", "五", "航"]
def gen_noise():
    l = random.randint(1, 7)
    noise = ""
    for _ in range(l):
        noise += random.choice(noise_cn)
    return noise

# augment by generating a small amount of noisy sample
def aug5(trainsz):
    aug = []
    for _ in range(trainsz // 33):
        noise = gen_noise()
        utt = {
            "utt_id": 1,
            "manual_transcript": noise,
            "asr_1best": noise,
            "semantic": []
        }
        aug.append([utt])
    return aug

# this augment the data by adding deny semantic samples
def aug6(train, _ords, ops, pois, aug_factor):
    aug = copy.deepcopy(train)
    while (len(aug) / len(train)) < aug_factor:
        for op in ops:
            text1 = f"不{op}"
            text2 = f"不要{op}"
            text3 = f"{op}错了"
            for text in [text1, text2, text3]:
                utt = {
                    "utt_id": 1,
                    "manual_transcript": text,
                    "asr_1best": text,
                    "semantic": [['deny', '操作', op]]
                }
                aug.append([utt])
        for i in range(0, len(pois) - 1):
            poi1 = pois[i]
            poi2 = pois[i + 1]
            utt1 = {
                "utt_id": 2,
                "manual_transcript": f"是{poi1}不是{poi2}",
                "asr_1best": f"是{poi1}不是{poi2}",
                "semantic": [
                    [
                        "inform",
                        "poi名称",
                        poi1,
                    ],
                    [
                        "deny",
                        "poi名称",
                        poi2,
                    ]
                ]
            }
            utt2 = {
                "utt_id": 2,
                "manual_transcript": f"是{poi1}不是{poi2}你这个笨蛋",
                "asr_1best": f"是{poi1}不是{poi2}你个笨蛋",
                "semantic": [
                    [
                        "inform",
                        "poi名称",
                        poi1,
                    ],
                    [
                        "deny",
                        "poi名称",
                        poi2,
                    ]
                ]
            }
            utt3 = {
                "utt_id": 2,
                "manual_transcript": f"不是{poi1}",
                "asr_1best": f"不是{poi1}",
                "semantic": [
                    [
                        "deny",
                        "poi名称",
                        poi1,
                    ]
                ]           
            }
            utt4 = {
                "utt_id": 2,
                "manual_transcript": f"你搞错了应该是{poi1}",
                "asr_1best": f"你搞错了应该是{poi1}",
                "semantic": [
                    [
                        "inform",
                        "poi名称",
                        poi1,
                    ]
                ]           
            }
            aug.append([utt1])
            aug.append([utt2])
            aug.append([utt3])
            aug.append([utt4])
            break
    return aug
    
def customize_main(args, f1, f2, f3, f4):
    filename = f'aug_{f1}_{f2}_{f3}_{f4}_{args.noise}.json'

    aug_train = copy.deepcopy(train)

    aug_train.extend(aug1(train, ords, ops, pois, f1))
    aug_train.extend(aug2(train, ords, ops, pois, f2))
    aug_train.extend(aug3(train, ords, ops, pois, f3))
    aug_train.extend(aug6(train, ords, ops, pois, f4))

    if args.noise:
        aug_train.extend(aug5(len(aug_train)))

    print(f"augmentation done, length {len(train)} -> {len(aug_train)}")

    if not args.check_size:
        print("dumping augmented data")
        with open(os.path.join('data', filename), 'w', encoding='utf-8') as f:
            json.dump(aug_train, f, ensure_ascii=False)   

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()   
    arg_parser.add_argument('--f1', default=1, type=int, help='augmentation factor')
    arg_parser.add_argument('--f2', default=1, type=int, help='augmentation factor')
    arg_parser.add_argument('--f3', default=1, type=int, help='augmentation factor')
    arg_parser.add_argument('--f4', default=1, type=int, help='augmentation factor')
    arg_parser.add_argument('--noise', action='store_true', help='')
    arg_parser.add_argument('--gen', action='store_true', help='whether to generate samples, the performance is not good')
    arg_parser.add_argument('--seed', default=42, help='')
    arg_parser.add_argument('--check_size', action='store_true', help='whether to check the size of the dataset only')
    args = arg_parser.parse_args()

    random.seed(args.seed)

    # filename = f'aug_train_{args.factor}.json'
    # main(args, filename)
    customize_main(args, args.f1, args.f2, args.f3, args.f4)
