#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifarnet_on_cifar10.sh
set -e

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/cifarnet-model

# Where the dataset is saved to.
DATASET_DIR=/tmp/cifar10

# Download the dataset
python download_and_convert_data.py \
  --dataset_name=cifar10 \
  --dataset_dir=${DATASET_DIR}

# Run training.
python train_image_classifier.py \
  --train_dir=/tmp/cifarnet-model \
  --dataset_name=cifar10 \
  --dataset_split_name=train \
  --dataset_dir=/tmp/cifar10 \
  --model_name=densenet \
  --batch_size=32 \
  --save_interval_secs=120 \
  --save_summaries_secs=120 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --learning_rate=0.001 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004
  
python train_image_classifier.py \
  --dataset_name=cifar10 \
  --dataset_dir=/tmp/cifar10 \
  --model_name=inception_v4 \
  --train_dir=/tmp/cifar10-inception_v4-model \
  --learning_rate=0.001 \
  --optimizer=rmsprop \
  --batch_size=32 \
  --clone_on_cpu=True
  
python train_image_classifier.py \
  --dataset_name=cifar10 \
  --dataset_dir=/tmp/cifar10 \
  --model_name=densenet \
  --train_dir=/tmp/cifarnet-model \
  --learning_rate=0.1 \
  --optimizer=rmsprop  \
  --batch_size=32 \
  --clone_on_cpu=True
  
python eval_image_classifier.py 
--dataset_name=cifar10 
--dataset_dir=/tmp/cifar10 
--dataset_split_name=train 
--model_name=densenet 
--checkpoint_path=/tmp/to/train_ckpt
--eval_dir=/tmp/cifarnet-model 
--batch_size=32 
--max_num_batches=128

python train_eval_image_classifier.py
 --dataset_name=cifar10
 --dataset_dir=/tmp/cifar10 
 --model_name=densenet
 --optimizer=sgd
 --checkpoint_exclude_scopes=densenet/DenseNet/Logits,densenet/DenseNet/AuxLogits/Aux_logits
 --train_dir=/tmp/to/log/train_ckpt 
 --learning_rate=0.01
 --learning_rate_decay_type=exponential
 --learning_rate_decay_factor=0.57
 --num_epochs_per_decay=2 
 --dataset_split_name=test 
 --eval_dir=/tmp/to/eval_den
 --max_num_batches=128
 --allow_soft_placement=True
 --clone_on_cpu=True
  
  python train_image_classifier.py --dataset_name=cifar10 --dataset_dir=/tmp/cifar10 --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --train_dir=/tmp/to/train_ckpt --learning_rate=0.001 --optimizer=rmsprop  --batch_size=32

# Run evaluation.
python eval_image_classifier.py \
  --checkpoint_path=/tmp/to/log/train_ckpt \
  --eval_dir=/tmp/to/eval_den \
  --dataset_name=cifar10 \
  --dataset_split_name=test \
  --dataset_dir=/tmp/cifar10 \
  --model_name=densenet
