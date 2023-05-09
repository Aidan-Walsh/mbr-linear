import collectors as c
import data as d
import submitit
from tqdm import tqdm

mbpp = d.MBPPGoogleDataset(mode='assertion')


collector = c.CollectorWithInfo(configs=None, dataset=mbpp)
args = collector.from_args(args=None, dataset=mbpp)
newcollector = c.CollectorWithInfo(configs=args.configs, dataset=mbpp)
# folder that stores execution results for the job
executor = submitit.AutoExecutor(folder='log_test')
executor.update_parameters(timeout_min=5000, gpus_per_node=2, cpus_per_task=4)
job = executor.submit(newcollector)
