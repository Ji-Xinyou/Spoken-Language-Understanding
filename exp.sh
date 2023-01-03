#!/usr/bin/sh

python ./scripts/slu_baseline.py --model=focus --encoder_cell=LSTM --max_epoch=100 --lr=1e-4 --dropout=0.1 --num_layer=2
python ./scripts/slu_baseline.py --model=focus --encoder_cell=GRU --max_epoch=100 --lr=1e-4 --dropout=0.1 --num_layer=2
python ./scripts/slu_baseline.py --model=focus --encoder_cell=RNN --max_epoch=100 --lr=1e-4 --dropout=0.1 --num_layer=2
python ./scripts/slu_baseline.py --model=baseline --encoder_cell=LSTM --max_epoch=100 --lr=1e-4 --dropout=0.1 --num_layer=2
python ./scripts/slu_baseline.py --model=baseline --encoder_cell=GRU --max_epoch=100 --lr=1e-4 --dropout=0.1 --num_layer=2
python ./scripts/slu_baseline.py --model=baseline --encoder_cell=RNN --max_epoch=100 --lr=1e-4 --dropout=0.1 --num_layer=2
