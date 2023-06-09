# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os
import pickle
import regex
import signal
import subprocess
import tempfile
import threading
from datasets import load_metric
from glob import glob
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from data import MBPPGoogleDataset
from utils_sql import *


class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(
                self.cmd, shell=True, preexec_fn=os.setsid)
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            os.killpg(self.process.pid, signal.SIGTERM)
            thread.join()
        return self.process.returncode


class PythonFunctionExecutor(object):
    def __init__(self, function_content, function_call, timeout=10):
        self.function_content = function_content
        self.function_call = function_call
        self.timeout = timeout

    def __call__(self):
        tempdir = tempfile.TemporaryDirectory()
        with open(f'{tempdir.name}/code.py', 'w') as fout:
            print(self.function_content, file=fout)
            print(f'result = {self.function_call}', file=fout)
            print(f'import pickle', file=fout)
            print(
                f'pickle.dump(result, open("{tempdir.name}/execution_result.pkl", "wb"))', file=fout)
        command = Command(f'python {tempdir.name}/code.py >/dev/null 2>&1')
        execution_status = command.run(timeout=self.timeout)
        if execution_status == 0:
            try:
                execution_results = pickle.load(
                    open(f'{tempdir.name}/execution_result.pkl', 'rb'))
            except:
                execution_results = None
        else:
            execution_results = None
        tempdir.cleanup()
        return execution_status, execution_results


def execute_mbpp_google_folder(base_path):
    # single assertion
    dataset = MBPPGoogleDataset(mode='assertion')
    for path in glob(f'{base_path}/*jsonl'):  # execute first assertion call
        if os.path.exists(path.replace('jsonl', 'exec.pkl')):
            continue
        split = os.path.basename(path).split('-')[0]
        execution_results = list()
        for i, line in enumerate(tqdm(open(path).readlines())):
            assertion = dataset.data[split][i][-1]
            command = regex.match(f'assert (.+)==.+', assertion).group(1)
            item = json.loads(line)
            python_function = item['trg_prediction']
            executor = PythonFunctionExecutor(python_function, command)
            execution_result = executor()
            execution_results.append(execution_result)
        with open(path.replace('jsonl', 'exec.pkl'), 'wb') as fout:
            pickle.dump(execution_results, fout)
    # multiple assertions (cheating)
    dataset = MBPPGoogleDataset(mode='assertion-full')
    for path in glob(f'{base_path}/*jsonl'):  # execute all assertion calls
        if os.path.exists(path.replace('jsonl', 'execfull.pkl')):
            continue
        split = os.path.basename(path).split('-')[0]
        execution_results = list()
        for i, line in enumerate(tqdm(open(path).readlines())):
            execution_result = list()
            item = json.loads(line)
            python_function = item['trg_prediction']
            for assertion in dataset.data[split][i][-1]:
                command = regex.match(f'assert (.+)==.+', assertion).group(1)
                executor = PythonFunctionExecutor(python_function, command)
                execution_result.append(executor())
            execution_results.append(execution_result)
        with open(path.replace('jsonl', 'execfull.pkl'), 'wb') as fout:
            pickle.dump(execution_results, fout)
    # multiple assertions (pass or fail)
    for path in glob(f'{base_path}/*jsonl'):
        if os.path.exists(path.replace('jsonl', 'execfullpass.pkl')):
            continue
        split = os.path.basename(path).split('-')[0]
        execution_results = list()
        for i, line in enumerate(tqdm(open(path).readlines())):
            execution_result = list()
            item = json.loads(line)
            python_function = item['trg_prediction']
            for assertion in dataset.data[split][i][-1]:
                command = regex.match(f'assert (.+==.+)', assertion).group(1)
                executor = PythonFunctionExecutor(
                    python_function, f'({command})')
                execution_result.append(executor())
            execution_results.append(execution_result)
        with open(path.replace('jsonl', 'execfullpass.pkl'), 'wb') as fout:
            pickle.dump(execution_results, fout)
