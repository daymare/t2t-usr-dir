# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data generators for CASAS smart home data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf


EOS = text_encoder.EOS

def _build_vocab(filename, vocab_path):
    f = open(filename, "r")
    text = f.read()
    lines = text.split("\n")
    words = set()

    # find unique sensors in file
    for line in lines:
        tokens = line.split()
        
        if len(tokens) < 5:
            # found an invalid line
            continue

        if tokens[4] == "Control4-Motion" and tokens[3] == "ON":
            sensor = tokens[2]
            if sensor not in words:
                words.add(sensor)

    # put words into vocab file
    words = list(words)
    vocab_file = open(vocab_path, "w")
    vocab_file.write("\n".join(words))

def _get_token_encoder(data_path, vocab_path):
    if not tf.gfile.Exists(vocab_path):
        _build_vocab(data_path, vocab_path)
    return text_encoder.TokenTextEncoder(vocab_path)

@registry.register_problem
class SensorModel(text_problems.Text2SelfProblem):
    """ CASAS tokyo sensors. """

    @property
    def approx_vocab_size(self):
        return 50

    @property
    def is_generate_per_split(self):
        return False

    @property
    def vocab_filename(self):
        return "vocab.txt"

    @property
    def dataset_splits(self):
        """ Splits of data to produce and number of output shards for each.
            90% train
            10% eval
        """
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]
    
    def generate_samples(self, 
                         data_dir=None, 
                         tmp_dir=None, 
                         dataset_split=None):
        """
            grab data from text file in data dir
        """
        # get data file
        data_path = data_dir + "/tokyo.txt"
        vocab_path = data_dir + "/vocab.txt"

        # get token encoder
        _get_token_encoder(data_path, vocab_path)

        with tf.gfile.GFile(data_path, "r") as f:
            sensors = []
            for line in f:
                if len(sensors) >= 10:
                    yield {
                        "targets": " ".join(sensors)
                    }
                    sensors = []

                tokens = line.split()
                if tokens[4] == "Control4-Motion" and tokens[3] == "ON":
                    sensors.append(tokens[2])


