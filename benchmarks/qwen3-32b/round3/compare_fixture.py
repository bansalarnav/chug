#!/usr/bin/env python3
"""Compare an engine's raw arrays with one retained reference fixture.

uv run --no-project --with numpy python compare_fixture.py fixtures/code candidate/code
Candidate manifest uses the same array names, shape, dtype and file fields.
No universal floating-point pass threshold is assumed.
"""
import argparse,json
from pathlib import Path
import numpy as np


def load(directory):
 manifest=json.loads((directory/'manifest.json').read_text());result={}
 for name,entry in manifest.items():
  dtype={'float32':'<f4','int64':'<i8'}[entry['dtype']]
  result[name]=np.fromfile(directory/entry['file'],dtype=dtype).reshape(entry['shape'])
 return result


def compare(reference,candidate):
 result={'missing':sorted(reference.keys()-candidate.keys()),'extra':sorted(candidate.keys()-reference.keys()),'arrays':{}}
 for name in sorted(reference.keys()&candidate.keys()):
  ref=reference[name];got=candidate[name];row={'reference_shape':list(ref.shape),'candidate_shape':list(got.shape)}
  result['arrays'][name]=row
  if ref.shape!=got.shape:row['error']='shape mismatch';continue
  if np.issubdtype(ref.dtype,np.integer):
   row.update(equal=bool(np.array_equal(ref,got)),mismatched_elements=int(np.count_nonzero(ref!=got)));continue
  if not np.isfinite(got).all():row['error']='nonfinite candidate values';continue
  a=ref.astype(np.float64);b=got.astype(np.float64);diff=b-a
  row.update(max_abs_error=float(np.max(np.abs(diff))),rmse=float(np.sqrt(np.mean(diff**2))),relative_rmse=float(np.sqrt(np.mean(diff**2))/max(np.sqrt(np.mean(a*a)),1e-30)))
  if 'logit' in name:
   aa=a-a.mean(axis=-1,keepdims=True);bb=b-b.mean(axis=-1,keepdims=True)
   logp=a-a.max(axis=-1,keepdims=True);logp-=np.log(np.exp(logp).sum(axis=-1,keepdims=True))
   logq=b-b.max(axis=-1,keepdims=True);logq-=np.log(np.exp(logq).sum(axis=-1,keepdims=True))
   row.update(centered_rmse=float(np.sqrt(np.mean((aa-bb)**2))),greedy_agreement=float(np.mean(a.argmax(axis=-1)==b.argmax(axis=-1))),mean_kl_reference_to_candidate_nats=float(np.mean(np.sum(np.exp(logp)*(logp-logq),axis=-1))))
 return result


def main():
 p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--out',type=Path);args=p.parse_args()
 result=compare(load(args.reference),load(args.candidate));text=json.dumps(result,indent=2)+'\n'
 if args.out:args.out.write_text(text)
 else:print(text,end='')
 if result['missing'] or any('error' in row or row.get('equal') is False for row in result['arrays'].values()):raise SystemExit(1)

if __name__=='__main__':main()
