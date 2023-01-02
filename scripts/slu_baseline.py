#coding=utf8
import gc
import os
import sys
import time

from torch.optim import Adam
from tqdm import tqdm

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from model.slu_baseline_tagging import SLUTagging
from utils.args import init_args
from utils.batch import from_example_list
from utils.example import Example
from utils.initialization import *
from utils.vocab import PAD

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')

Example.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)

train_dataset = Example.load_dataset(train_path)
dev_dataset = Example.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example.word_vocab.vocab_size
args.num_tags = Example.label_vocab.num_tags
args.pad_idx = Example.word_vocab[PAD]
args.tag_pad_idx = Example.label_vocab.convert_tag_to_idx(PAD)


model = SLUTagging(args).to(device)
Example.word2vec.load_embeddings(model.word_embed, Example.word_vocab, device=device)


def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for _, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = Example.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


if not args.testing:
    num_training_steps = \
        ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))

    optimizer = set_optimizer(model, args)

    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size

    print('Start training ......')
    torch.cuda.empty_cache()
    gc.collect()
    for i in tqdm(range(args.max_epoch)):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0

        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list(args, cur_dataset, device, train=True)
            output, loss = model(current_batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1

        print('Training: \t \
              Epoch: %d\t \
              Time: %.4f\t \
              Training Loss: %.4f' \
              % (i, time.time() - start_time, epoch_loss / count))

        # open if have low memory
        # torch.cuda.empty_cache()
        # gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']

        print('Evaluation: \t \
              Epoch: %d\t \
              Time: %.4f\t \
              Dev acc: %.2f\t \
              Dev fscore(p/r/f): (%.2f/%.2f/%.2f)' \
              % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], \
            best_result['dev_f1'], best_result['iter'] \
            = dev_loss, dev_acc, dev_fscore, i

            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('model.bin', 'wb'))

            print('NEW BEST MODEL: \
                  \tEpoch: %d\t \
                  Dev loss: %.4f\t \
                  Dev acc: %.2f\t \
                  Dev fscore(p/r/f): (%.2f/%.2f/%.2f)' \
                  % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \t \
          Epoch: %d\t \
          Dev loss: %.4f\t \
          Dev acc: %.4f\t \
          Dev fscore(p/r/f): (%.4f/%.4f/%.4f)' \
          % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']

    print("Evaluation costs %.2fs ; \
          Dev loss: %.4f\t \
          Dev acc: %.2f\t \
          Dev fscore(p/r/f): (%.2f/%.2f/%.2f)" \
          % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
