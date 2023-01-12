import json
import os
import torch
import argparse
import sys

install_path = os.path.abspath(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from model.bertlstm import BertLSTM
from utils.initialization import set_torch_device
from utils.batch import from_example_list
from utils.example import Example
from utils.args import add_argument_base
from utils.vocab import PAD


arg_parser = argparse.ArgumentParser()

#! Arguments here also use the values from ./utils/args.py
add_argument_base(arg_parser)
arg_parser.add_argument('--model_path', default='./models_saved/weight.bin', help='model path')
arg_parser.add_argument('--test_src', default='./data/test_unlabelled.json', help='test unlabelled path')

args = arg_parser.parse_args()

Example.configuration(args.dataroot, train_path=None, word2vec_path=None)

device = set_torch_device(args.device)

test_dataset = Example.load_dataset(args.test_src)

print("Length of test dataset: %d" % (len(test_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.num_tags = Example.label_vocab.num_tags
args.pad_idx = Example.word_vocab[PAD]
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)

model_path = args.model_path
print("loading weight from %s" % (model_path))
model_dict = torch.load(open(model_path, 'rb'), map_location=device)

state_dict = model_dict['weight']

model = BertLSTM(args, device).to(device)
print("\nloading state_dict to model")
model.load_state_dict(state_dict)

model.eval()

result = []

print("Start predicting")
with torch.no_grad():
    current_batch = from_example_list(args, test_dataset, device, train=True)
    pred, label, loss = model.decode(Example.label_vocab, current_batch)
    for j in range(len(current_batch)):
        if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
            print(current_batch.utt[j], pred[j], label[j])
        data = [{"utt_id": 1, "asr_1best": current_batch.utt[j], "pred": [s.split('-') for s in pred[j]]}]
        result.append(data)

print("Prediction done, dumping prediction to test.json")
with open(os.path.join('test.json'), 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)
print("dumped")
