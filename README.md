# Enhancing the BERT training with Semi-supervised Generative Adversarial Networks
`GAN-Bert` combines the power of pre-trained `Bert` and `GANs`. 
## Preparing data set for model
`GAN-Bert` uses three data sets:
* `labeled.tsv` - Examples with labels for supervised training
* `unlabeled.tsv` - Examples without labels for adversial training
* `test.tsv` - Examples with labels for evaluation

Every example in `labeled.tsv`, `unlabeled.tsv` and `test.tsv` must come from same distribution (source).
## Structure of data set
For K-class classification task, modify `line: 105` in `data_processors.py` to include the class labels (in upper case) along with `UNK` label for _unlabeled_ examples.
* ### Structure of `labeled.tsv` and `test.tsv`
  ```
  label sentence
  label_1 sentence_1
  label_2 sentence_2
  ...
  ```
* ### Structure of `unlabeled.tsv`
  ```
  label sentence
  UNK sentence_1
  UNK sentence_2
  ...
  ```
## Training and evaluating the model
See `classification_with_GANBert.ipynb` for training on Google Colab.

To run the codes, install `BERT_base_cased` model as
```python
!wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
!unzip cased_L-12_H-768_A-12.zip
```
The codes were tested on Google Colab using GPU runtime. To perform similar experiment, execute
```python
!pip uninstall tensorflow
!pip install tensorflow-gpu==1.14.0
!pip install gast==0.2.2
!pip install git+https://github.com/guillaumegenthial/tf_metrics.git
```
To make sure that the runtime will be using GPU, try this
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
!python -c 'import tensorflow as tf; tf.test.gpu_device_name()'
```
Then, run the `ganbert` model as
```python
%%shell
python -u ganbert.py \
        --num_classes=3 \
        --label_rate=0.02 \
        --do_train=true \
        --do_eval=true \
        --do_predict=false \
        --data_dir=data \
        --vocab_file=cased_L-12_H-768_A-12/vocab.txt \
        --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
        --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt \
        --max_seq_length=64 \
        --train_batch_size=64 \
        --learning_rate=2e-5 \
        --num_train_epochs=3 \
        --warmup_proportion=0.1 \
        --do_lower_case=false \
        --output_dir=ganbert_output_model
```
