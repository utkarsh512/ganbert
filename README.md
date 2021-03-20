# Enhancing the BERT training with Semi-supervised Generative Adversarial Networks
`GANBERT` combines the power of pre-trained `BERT` and `GANs`. To run the codes, install `BERT_base_cased` model as
```bash
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
