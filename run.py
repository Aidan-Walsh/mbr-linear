from sample_selectors import select_mbpp as sm
from sample_selectors2 import select_mbpp as sm2
import numpy as np

hyper_params = np.array([-0.5, 0.5, -0.5, 100, 2])

# cannot run mbr_meteor in vscode/locally
# to use mbr_meteor, use colab
# available decoding schemes:
schemes = ['mbr_exec', 'logprob', 'avg_logprob', 'mbr_bleu', 'context', 'mbr_tokenmeteor', 'mbr_tokenbleu', 'executability-logprob',
           'executability-avglogprob', 'executability-mbr_bleu', 'executability-mbr_tokenbleu']  # note: context should only be used for codex data
sm(('test', 0.3, 'executability-mbr_tokenbleu',
    'data/mbr-exec-release/mbpp/', 2, 100))

# linear interpolation (only use this for linear interpolation)
sm2(('test', 0.3, 'interpolation_mbr_exec',
     'data/mbr-exec-release/mbpp/', 2, 10), hyper_params=hyper_params)
