import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

root = Path(os.environ.get("BORROWING_ROOT", ".")).resolve()
folder = root / 'results/final/sibling-refusal-borrowing-v1'
job = os.environ['SLURM_JOB_ID']
assert job in json.loads((root/'compute/jobs.json').read_text()), 'Unregistered job'
devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
assert len(devices) == 4, devices
pending = queue.Queue()
for task in [('gemma','en'),('gemma','uk'),('mamay','en'),('mamay','uk')]:pending.put(task)
stop = threading.Event()

def worker(device):
    while not stop.is_set():
        try:model,language = pending.get_nowait()
        except queue.Empty:return
        log = folder / f'refine-{model}-{language}-job-{job}.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
        command = [sys.executable,str(folder/'refine_strength.py'),'--model',model,'--language',language]
        print('START',model,language,'GPU',device,flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            stop.set()
            raise RuntimeError(f'{model}/{language} failed; inspect {log}')
        print('DONE',model,language,flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(worker,device) for device in devices]
    for future in concurrent.futures.as_completed(futures):future.result()
