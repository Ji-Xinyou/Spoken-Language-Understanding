# SLU 口语语义理解Project
本组成员有纪信佑和陈景浩两位同学

本README包含每个文件的说明，训练、测试脚本运行方式以及依赖环境

**请注意不要开proxy**，会导致无法从huggingface上拉取模型，本次提交已经提供了Bert-Base-Chinese，但是Tokenizer仍然默认从huggingface拉取

---
[toc]

---
## 文件说明
文件说明中包含我们添加的文件的说明

```
.
|   aug.py  // 数据增强的python脚本，运行方式在后面说明
|   history.log  // 实验所用的命令以及运行结果
|   LICENSE
|   README.md
|   README4PROJECT.md  // 本文件
|   word2vec-768.txt
|
+---data
|   |   development.json
|   |   ontology.json
|   |   test_unlabelled.json
|   |   train.json
|   |
|   \---lexicon
|           operation_verb.txt
|           ordinal_number.txt
|           poi_name.txt
|
+---model
|   |   baselinemlp.py  // NOTUSED: 本次Project的尝试，Baseline后的feedforward改为mlp，因为没有什么价值没有写入报告中。
|   |   bert.py         // Bert模型，使用了bert-base-chinese预训练模型
|   |   bertlstm.py     // BertLSTM模型
|   |   focus.py        // NOTUSED: 本次Project的尝试，尝试用生成式的方式建模，可能由于实现的原因效果较差，因此没有写入报告中。
|   |   slu_baseline_tagging.py
|   |   twolstm.py      // NOTUSED: 本次Proejct的尝试，尝试使用第一个lstm的encoding作为第二个lstm的输入，没有什么意义，因此没有写入报告中。
|   |
|   +---bert-base-chinese  // 下载的预训练模型，放在此处因为本机无法正常从huggingface下载BertModel模型
|   |       config.json
|   |       pytorch_model.bin
|   |       vocab.txt
|   |
|   +---models_saved  // 最终提交的效果最好的模型，使用了BertLSTM, Dev Acc为80.17%
|   |       bertlstm_3113_8017.bin
|   |
|
+---scripts
|       slu_baseline.py
|       test.py  // 输出test.json的脚本
|
\---utils
    |   args.py
    |   batch.py
    |   evaluator.py
    |   example.py
    |   initialization.py
    |   vocab.py
    |   word2vec.py
```
---
## 训练方式
首先，根据使用`aug.py`想增强的方式**生成数据增强后的训练集**
aug.py的使用方式如下
```sh
python aug.py --f1 F1 --f2 F2 --f3 F3 --f4 F4 --noise --seed SEED [--check_size]
```
其中参数的意义如下
* F1：使用poi_name等效替换数据增强的倍数
* F2: 使用其他key-value等效替换数据增强的倍数
* F3: 使用operation和ordinal_number等效替换数据增强的倍数
* F4: 添加deny类型数据的增强倍数
* noise: 当指定这个flag，添加2%的噪音数据
* SEED: 指定随机数种子
* check_size: 当指定这个flag，将不会dump训练文件，只会输出增强后训练样本的数据数量
生成的训练数据命名规则为aug_F1_F2_F3_F4_{NOISE}.json

举例:

```sh
python aug.py --f1 3 --f2 1 --f3 1 --f4 3 --noise
```

可以生成Train Scale为(3, 1, 1, 3)的训练集，并且加入2%的噪音，生成的文件为`./data/aug_3_1_1_3_False.json`

然后使用你生成的训练集对模型进行训练, 通过`--tr_filename`指定训练文件

* 训练文件必须在`./data`下，`--model`指定模型。

* model的范围有`baseline`, `baselinemlp`, `twolstm`, `focus`, `bert`, `bertlstm`
    * 其中，`baselinemlp`, `twolstm`, `focus`是**失败或者无意义的模型**，请使用`bert`, `bertlstm`或者`baseline`中的一种。

举例：

```sh
python ./scripts/slu_baseline.py --model=bertlstm --lr=3e-5 --dropout=0.2 --tr_filename=aug_3_1_1_3_False.json --batch_size=64
```

使用了你生成的在`./data/aug_3_1_1_3_False.json`训练集，指定了模型为`BertLSTM`，设定了学习率和dropout等
* 具体的args内容可以在`./utils/args.py`中查看。训练完成后，最优的模型会在`./model.bin`。
---
## 测试脚本运行方式
测试脚本所在路径为`./scripts/test.py`

运行方式如下
```sh
python ./scripts/test.py --model_path MODEL_PATH
```

通过`MODEL_PATH`指定模型文件，本次提交的模型为`./models_saved/bertlstm_3113_8017.bin`，运行如下命令会产生`test.json`在当前目录。

```sh
python ./scripts/test.py --model_path ./models_saved/bertlstm_3113_8017.bin
```
---
## 运行环境要求
* torch, cuda能正常运行即可，huggingface的transformers库(引入Bert)。
* 本次提交提供了bert模型在./model/bert-base-chinese中，在运行训练脚本时需要存在本地目录
* 在运行脚本时，请不要开启proxy，否则bert tokenizer将无法从huggingface上被拉取
* 在运行测试脚本的时候，需要指定模型的路径，请保证模型的路径有效
