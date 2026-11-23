#!/usr/bin/env python3
"""Delete this campaign's exact HDD tree after independently verifying its archive.

Run with the system Python from the user's home, after all benchmark processes
have exited. The home audit and private session marker must match the inode.
"""
import argparse,datetime,hashlib,json,os,shutil,stat,subprocess,tarfile
from pathlib import Path


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--delete',action='store_true');args=parser.parse_args()
 home=Path.home().resolve();records=home/'qwen3-32b-benchmark-records';audit=json.loads((records/'storage-audit.json').read_text())
 root=Path('/srv/local/bj2/faro/results2/arnavb9-qwen3-baseline')
 assert root==Path(audit['benchmark_root']) and not root.is_symlink() and root.resolve()==root
 info=root.stat();assert info.st_ino==audit['root_inode'] and info.st_uid==os.getuid()==audit['uid'] and stat.S_IMODE(info.st_mode)==0o700
 marker=json.loads((root/'.benchmark-session.json').read_text());assert marker['session_id']==audit['session_id']
 archive=args.archive.resolve();assert archive.is_relative_to(home) and archive.is_file()
 digest=hashlib.sha256()
 with archive.open('rb') as file:
  for chunk in iter(lambda:file.read(1024*1024),b''):digest.update(chunk)
 assert digest.hexdigest()==args.sha256,'Archive checksum mismatch'
 with tarfile.open(archive,'r:gz') as file:
  names=set(file.getnames())
  ref=json.load(file.extractfile('round3/fixtures/reference.json'))
  for fixture in ref['fixtures']:
   npz=file.extractfile('round3/fixtures/'+fixture['file']).read()
   assert hashlib.sha256(npz).hexdigest()==fixture['sha256'], 'Archived NPZ mismatch'
   prefix='round3/fixtures/'+fixture['name']+'/'
   manifest=json.load(file.extractfile(prefix+'manifest.json'))
   for entry in manifest.values():
    content=file.extractfile(prefix+entry['file']).read()
    assert hashlib.sha256(content).hexdigest()==entry['sha256'], 'Archived binary mismatch'
  validation=json.load(file.extractfile('round3/results/validation.json'))
  assert validation['invalid_streams']==0 and not validation['invalid_natural_prompts'], 'Invalid token counts remain'
  assert not validation['kernel_errors'], 'Component checks failed'
  assert all(x['npz_sha256_ok'] for x in validation['numerical']['npz']) and all(x['ok'] for x in validation['numerical']['binaries']), 'Fixture integrity check failed'
  assert not validation['pending_base_jobs'] and not validation['pending_addons'], 'Benchmark stages are still pending'

 required={'results/model.json','extended-results/corpus.json','round3/fixtures/reference.json','round3/results/validation.json','round3/REPORT.md'}
 assert required<=names, 'Required measurements missing from archive'
 active=[]
 for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,stat=,args='],text=True).splitlines():
  parts=line.strip().split(None,2)
  if len(parts)==3 and str(root) in parts[2] and not parts[1].startswith('Z'):active.append(line)
 assert not active,'Benchmark processes still exist: '+str(active)
 count=0
 for directory,dirs,files in os.walk(root,followlinks=False):
  for name in dirs+files:
   entry=Path(directory)/name;assert entry.lstat().st_uid==os.getuid(),'Unexpected owner: '+str(entry);count+=1
 allocated=int(subprocess.check_output(['du','-s','--block-size=1',str(root)],text=True).split()[0])
 result=dict(root=str(root),root_inode=info.st_ino,entries=count,allocated_bytes_before=allocated,archive=str(archive),archive_sha256=args.sha256,archive_members=len(names),owner_checks_passed=True,benchmark_processes=[],deleted=False)
 if args.delete:
  shutil.rmtree(root);assert not root.exists();result['deleted']=True;result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(records/'cleanup.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
