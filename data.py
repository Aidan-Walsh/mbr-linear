# Copyright (c) Meta Platforms, Inc. and affiliates.

import collections
import json
import os
import regex


class MBPPGoogleDataset(object):
    def __init__(self, path='data/mbpp/mbpp.jsonl', mode='function_name'):
        raw_data = sorted([json.loads(x)
                          for x in open(path)], key=lambda x: x['task_id'])
        for i, data_item in enumerate(raw_data):
            assert data_item['task_id'] == i + 1
        self.raw_data = collections.defaultdict()
        self.mode = mode
        # 374 for training, 100 heldout, 500 test
        self.raw_data['train'] = raw_data[:10] + raw_data[510:]
        self.raw_data['test'] = raw_data[10:510]
        # data for codex collector, in input-output-info format
        self.data = collections.defaultdict()
        for split in self.raw_data:
            self.data[split] = self.extract_data(self.raw_data[split], mode)

    @staticmethod
    def extract_data(raw_data, mode):
        if mode == 'function_name':
            def get_function_name(test_example): return regex.match(
                'assert [\(]*([^\(]+)\(', test_example).group(1)
            info = [get_function_name(x['test_list'][0]) for x in raw_data]
        elif mode == 'assertion':
            info = [x['test_list'][0] for x in raw_data]
        elif mode == 'assertion-full':
            info = [x['test_list'] for x in raw_data]
        else:
            raise Exception(f'Mode {mode} not supported.')
        nls = [x['text'] for x in raw_data]
        codes = [x['code'] for x in raw_data]
        return list(zip(nls, codes, info))
