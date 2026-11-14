#!/usr/bin/env python3
"""SGLang PP microbatch ablation after the main matrix; profiling is separate."""
import argparse
import hashlib
import gzip
import http.client
import json
from pathlib import Path
import time
from extended import batch, summarize


def control(port, endpoint, payload):
    connection=http.client.HTTPConnection('127.0.0.1',port,timeout=600)
    try:
        connection.request('POST',endpoint,json.dumps(payload),{'Content-Type':'application/json'})
        response=connection.getresponse(); text=response.read().decode()
        if response.status!=200: raise RuntimeError(f'{endpoint}: {response.status} {text}')
        try: return json.loads(text)
        except json.JSONDecodeError: return text
    finally: connection.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=18080); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); root=a.out
    corpus_bytes=(root/'corpus.json').read_bytes(); corpus=json.loads(corpus_bytes)['128']
    digest=hashlib.sha256(corpus_bytes).hexdigest()
    raw_path=root/'pipeline-sweep.jsonl'
    previous=[json.loads(l) for l in raw_path.read_text().splitlines()] if raw_path.exists() else []
    matrix=[(4,m) for m in [1,2,4,8,16]]+[(16,m) for m in [8,16]]
    summaries=[]; controls=[]
    def set_cap(cap):
        answer=control(a.port,'/set_internal_state',{'server_args':{'pp_max_micro_batch_size':cap}})
        assert isinstance(answer,list) and answer and all(v is True for v in answer),answer
        controls.append(dict(time=time.time(),microbatch_cap=cap,response=answer))
        (root/'pipeline-sweep-controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    try:
        with raw_path.open('a') as f:
            for case_id,(concurrency,cap) in enumerate(matrix):
                case=dict(suite='pipeline_microbatch',input_tokens=128,output_tokens=128,
                    concurrency=concurrency,requests=concurrency,repetitions=3,mode='burst',
                    microbatch_cap=cap,pipeline_slots=2)
                trials=[r for r in previous if r['case_id']==case_id]
                set_cap(cap)
                if len(trials)<3:
                    batch(a.port,corpus[:concurrency],128)
                for repeat in range(len(trials),3):
                    start=time.time()
                    result=batch(a.port,corpus[4+repeat*concurrency:4+(repeat+1)*concurrency],128)
                    result.update(label='sglang-tp2-pp2',case=case,case_id=case_id,trial=repeat,
                        corpus_sha256=digest,wall_started_at=start,wall_finished_at=time.time())
                    f.write(json.dumps(result)+'\n'); f.flush(); trials.append(result)
                    print(f'PIPELINE c={concurrency} microbatch_cap={cap} repeat={repeat} tok/s={result["aggregate_output_tokens_per_s"]:.2f}',flush=True)
                summaries.append(summarize(case,trials,'sglang-tp2-pp2'))
                (root/'pipeline-sweep-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        # These traces include profiler overhead and are excluded from all speed statistics.
        profile_results=[]
        for cap in [2,16]:
            output=root/'profiles'/f'c4-microbatch-{cap}'
            output.mkdir(parents=True,exist_ok=True)
            set_cap(cap)
            batch(a.port,corpus[:4],64)
            started=False
            try:
                answer=control(a.port,'/start_profile',dict(output_dir=str(output),
                    activities=['CPU','GPU'],with_stack=False,record_shapes=False,
                    detailed_annotations=True,merge_profiles=False,
                    profile_prefix=f'c4-mb{cap}'))
                started=True
                diagnostic=batch(a.port,corpus[4:8],64)
                control(a.port,'/stop_profile',{})
                started=False
                # The HTTP acknowledgement can precede PP1 trace export.
                deadline=time.monotonic()+120
                complete=False
                while time.monotonic()<deadline:
                    traces=list(output.glob('*.trace.json.gz'))
                    if len(traces)>=4:
                        try:
                            for trace in traces:
                                with gzip.open(trace,'rb') as file:
                                    while file.read(1024*1024): pass
                            complete=True
                            break
                        except (EOFError,OSError): pass
                    time.sleep(1)
                profile_results.append(dict(microbatch_cap=cap,start_response=answer,all_rank_exports_complete=complete,
                    diagnostic_requests=4,output_tokens=64,output_directory=str(output)))
            except Exception as error:
                profile_results.append(dict(microbatch_cap=cap,error=str(error)))
            finally:
                if started:
                    try: control(a.port,'/stop_profile',{})
                    except Exception: pass
            (root/'pipeline-profile-status.json').write_text(json.dumps(profile_results,indent=2)+'\n')
    finally:
        set_cap(16)
    print('COMPLETE pipeline microbatch sweep',flush=True)


if __name__=='__main__': main()
