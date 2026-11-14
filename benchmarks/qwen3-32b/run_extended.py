#!/usr/bin/env python3
"""Run engines sequentially; capture launches, logs, GPU samples and failures."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from extended import cases

root = Path(os.environ['BENCH_ROOT'])
scripts = Path(__file__).resolve().parent
original_results = root / 'results'
results = root / 'extended-results'
results.mkdir(exist_ok=True)
lock = (root / 'run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
model = (original_results / 'model-path.txt').read_text().strip()


def interrupted(signum, frame):
    raise SystemExit(f'Interrupted by signal {signum}')


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)


def status(message):
    text = f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {message}'
    print(text, flush=True)
    (results / 'status.txt').write_text(text+'\n')


def stop(process):
    if process is None:
        return
    # Kill only this launch's process group, including distributed workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=30)


for engine in ['vllm', 'sglang']:
    for tp, pp in [(4, 1), (2, 2)]:
        label = f'{engine}-tp{tp}-pp{pp}'
        if (results / f'{label}-summary.json').exists():
            if len(json.loads((results / f'{label}-summary.json').read_text())) == len(cases()):
                status(f'SKIP complete {label}')
                continue
        python = str(root / 'envs' / engine / 'bin/python')
        if engine == 'vllm':
            command = [python, '-m', 'vllm.entrypoints.openai.api_server',
                '--model', model, '--served-model-name', 'qwen3-32b',
                '--host', '127.0.0.1', '--port', '18080',
                '--dtype', 'bfloat16', '--tensor-parallel-size', str(tp),
                '--pipeline-parallel-size', str(pp), '--distributed-executor-backend', 'mp',
                '--max-model-len', '9216', '--max-num-seqs', '32',
                '--max-num-batched-tokens', '2048', '--gpu-memory-utilization', '0.88',
                '--no-enable-prefix-caching', '--enable-chunked-prefill',
                '--generation-config', 'vllm', '--safetensors-load-strategy', 'eager']
        else:
            command = [python, '-m', 'sglang.launch_server',
                '--model-path', model, '--served-model-name', 'qwen3-32b',
                '--host', '127.0.0.1', '--port', '18080',
                '--dtype', 'bfloat16', '--tp-size', str(tp), '--pp-size', str(pp),
                '--context-length', '9216', '--max-running-requests', '32',
                '--chunked-prefill-size', '2048', '--mem-fraction-static', '0.88',
                '--cuda-graph-max-bs-decode', '32', '--cuda-graph-max-bs-prefill', '2048', '--disable-radix-cache',
                '--model-loader-extra-config', json.dumps({'enable_multithread_load': False})]
        launch_env = os.environ.copy()
        launch_env['MAX_JOBS'] = '4'
        launch_env['PATH'] = str(root / 'envs' / engine / 'bin') + os.pathsep + launch_env['PATH']
        if engine == 'sglang':
            cuda_home = root / 'envs/sglang/lib/python3.12/site-packages/nvidia/cu13'
            if not (cuda_home / 'bin/nvcc').exists():
                raise RuntimeError(f'Packaged CUDA compiler missing from {cuda_home}')
            if not (cuda_home / 'lib64').exists():
                (cuda_home / 'lib64').symlink_to('lib', target_is_directory=True)
            cudart_link = cuda_home / 'lib/libcudart.so'
            if not cudart_link.exists():
                cudart_link.symlink_to('libcudart.so.13')
            launch_env['CUDA_HOME'] = str(cuda_home)
            launch_env['PATH'] = str(cuda_home / 'bin') + os.pathsep + launch_env['PATH']
            launch_env['LIBRARY_PATH'] = str(cuda_home / 'lib') + os.pathsep + launch_env.get('LIBRARY_PATH', '')
        (results / f'{label}-environment.json').write_text(json.dumps({
            key: launch_env.get(key) for key in ['CUDA_HOME', 'CUDA_VISIBLE_DEVICES',
                'CUDA_DEVICE_ORDER', 'OMP_NUM_THREADS', 'HF_HOME', 'XDG_CACHE_HOME',
                'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH', 'PATH', 'LIBRARY_PATH', 'MAX_JOBS', 'FLASHINFER_WORKSPACE_BASE']}, indent=2)+'\n')
        (results / f'{label}-command.json').write_text(json.dumps(command, indent=2)+'\n')
        process = monitor = None
        started = time.monotonic()
        try:
            status(f'START {label}')
            with (results / f'{label}-server.log').open('w') as serverlog, \
                 (results / f'{label}-gpu.csv').open('w') as gpulog:
                monitor = subprocess.Popen(['nvidia-smi',
                    '--query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem',
                    '--format=csv', '--loop-ms=1000'], stdout=gpulog, stderr=subprocess.STDOUT,
                    start_new_session=True)
                process = subprocess.Popen(command, stdout=serverlog, stderr=subprocess.STDOUT,
                    start_new_session=True, env=launch_env)
                deadline = time.monotonic() + 2700
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f'Server exited {process.returncode}')
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:18080/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        time.sleep(2)
                else:
                    raise TimeoutError('Server startup exceeded 45 minutes')
                startup_s = time.monotonic()-started
                (results / f'{label}-startup.json').write_text(json.dumps({'startup_s': startup_s})+'\n')
                smoke_body = json.dumps(dict(model='qwen3-32b', temperature=0, max_tokens=64,
                    prompt='<|im_start|>user\nWhat is 2 + 2? Answer with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')).encode()
                smoke_request = urllib.request.Request('http://127.0.0.1:18080/v1/completions',
                    data=smoke_body, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(smoke_request, timeout=300) as response:
                    smoke = json.load(response)
                (results / f'{label}-smoke.json').write_text(json.dumps(smoke, indent=2)+'\n')
                status(f'READY {label} after {startup_s:.1f}s; benchmarking')
                subprocess.run([python, str(scripts / 'extended.py'), '--label', label,
                    '--port', '18080', '--model-path', model,
                    '--original-corpus', str(original_results / 'corpus.json'), '--out', str(results)], check=True)
                status(f'COMPLETE {label}')
        except Exception as error:
            status(f'FAILED {label}: {error}')
            (results / f'{label}-failure.json').write_text(json.dumps({'error': str(error)})+'\n')
        finally:
            stop(process)
            stop(monitor)
            time.sleep(5)
status('FINISHED configurations; inspect summaries and failure files')
