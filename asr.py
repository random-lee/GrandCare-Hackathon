#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Run inference for pre-processed data with a trained model.
"""
import time
import argparse
import logging

from utils import setup_asr, transcribe_file

def flatten_txt(nested_list):
    txt = []

    nested_list = nested_list[:]
    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            txt.append(sublist)
            
    return ' '.join(txt)

class ASR():
    
    def __init__(self):
        logging.basicConfig(format='pid %(process)5s [%(asctime)s] ' + \
                           '%(filename)15.15s:%(lineno)4d ' + \
                           '%(levelname)8s: %(message)s', level=logging.INFO)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info('loading model ...')
        
        # hard coded args
        args_dict = {
            'beam': 40,
            'bpe': None,
            'cpu': False,
            'criterion': 'cross_entropy',
            'ctc': False,
            'data': './data',
            'dataset_impl': None,
            'decoding_format': None,
            'diverse_beam_groups': -1,
            'diverse_beam_strength': 0.5,
            'empty_cache_freq': 0,
            'force_anneal': None,
            'fp16': False,
            'fp16_init_scale': 128,
            'fp16_scale_tolerance': 0.0,
            'fp16_scale_window': None,
            'gen_subset': 'test',
            'input_file': '',
            'iter_decode_eos_penalty': 0.0,
            'iter_decode_force_max_iter': False,
            'iter_decode_max_iter': 10,
            'iter_decode_with_beam': 1,
            'iter_decode_with_external_reranker': False,
            'kspmodel': None,
            'lenpen': 1,
            'lm_weight': 0.2,
            'log_format': None,
            'log_interval': 1000,
            'lr_scheduler': 'fixed',
            'lr_shrink': 0.1,
            'match_source_len': False,
            'max_len_a': 0,
            'max_len_b': 200,
            'max_sentences': None,
            'max_tokens': 10000000,
            'memory_efficient_fp16': False,
            'min_len': 1,
            'min_loss_scale': 0.0001,
            'model_overrides': '{}',
            'momentum': 0.99,
            'nbest': 1,
            'no_beamable_mm': False,
            'no_early_stop': False,
            'no_progress_bar': False,
            'no_repeat_ngram_size': 0,
            'num_shards': 1,
            'num_workers': 1,
            'optimizer': 'nag',
            'path': './data/model.pt',
            'prefix_size': 0,
            'print_alignment': False,
            'print_step': False,
            'quiet': False,
            'remove_bpe': None,
            'replace_unk': None,
            'required_batch_size_multiple': 8,
            'results_path': None,
            'retain_iter_history': False,
            'rnnt': False,
            'rnnt_decoding_type': 'greedy',
            'rnnt_len_penalty': -0.5,
            'sacrebleu': False,
            'sampling': False,
            'sampling_topk': -1,
            'sampling_topp': -1.0,
            'score_reference': False,
            'seed': 1,
            'shard_id': 0,
            'silence_token': '▁',
            'skip_invalid_size_inputs_valid_test': False,
            'task': 'speech_recognition',
            'temperature': 1.0,
            'tensorboard_logdir': '',
            'threshold_loss_scale': None,
            'tokenizer': None,
            'unkpen': 0,
            'unnormalized': False,
            'user_dir': './fairseq/examples/speech_recognition',
            'warmup_updates': 0,
            'weight_decay': 0.0,
            'wfstlm': None}

        self.args = argparse.Namespace(**args_dict)
        self.task, self.generator, self.models, self.sp, self.tgt_dict = setup_asr(self.args, self.logger)
        
        self.logger.info('model ready')
        
    def get_transcription(self, fpath):
        
        self.args.input_file = fpath
        _, trans = transcribe_file(self.args, self.task, self.generator, self.models, self.sp, self.tgt_dict)

        trans = flatten_txt(trans)
        self.logger.info("transcription: {}".format(trans))

        return trans

def triggers(trans):
    
    # respones 
    tokens = trans.split()
    
    if 'no' in tokens:
        print('remind again')
    
    elif 'not yet' in tokens:
        print('remind again')
        
    elif 'later' in tokens:
        print('remind again')
        
    elif 'yes' in tokens:
        print('good')
        
    elif 'ok' in tokens:
        print('good')
        
    else:
        print("sorry didn't hear that, please try again")

if __name__ == "__main__":
    
    asr = ASR()

    # what
    trans = asr.get_transcription('./what.wav')
    triggers(trans)
