#!/usr/bin/env python3
"""Portable BF16 Qwen3 reference fixtures and cached/full-prefix numerical checks."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

root=Path(os.environ['BENCH_ROOT']);out=root/'round3/fixtures';out.mkdir(exist_ok=True)
model_path=(root/'results/model-path.txt').read_text().strip()
torch.manual_seed(20261114)
tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
prompts=[
 ('arithmetic','What is 17 * 23? Explain briefly.'),
 ('code','Write a Python function that returns the first repeated integer in a list, or None.'),
 ('unicode','Translate into English: Die Straße führt über den Fluss. 日本語で短く挨拶してください。'),
 ('context','Remember these labels: cedar=17, birch=29, oak=43. Which label has value 29?'),
 ('long_context',('The logger records timestamp, worker ID, batch size, and queue depth. '*64)+' Summarize what the logger records.')]
metadata={'model':json.loads((root/'results/model.json').read_text()),'precision':'BF16 weights, SDPA attention',
 'versions':{'torch':torch.__version__},'selected_hidden_state_indices':[0,1,16,32,48,64],
 'hidden_state_definition':'Transformers hidden_states: 0 is embedding output, 1..63 are layer-boundary states, 64 is final normalized output. Save last eight prompt positions.',
 'numerical_gate':'Require finite outputs. Report cached/full-prefix cosine, relative RMSE, max error and greedy agreement without assuming bit equality or a universal tolerance. BF16 GEMM/attention shape changes can accumulate rounding differences.',
 'fixtures':[]}
print('Loading BF16 reference model',flush=True);started=time.time()
model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,
 device_map='balanced',max_memory={i:'19GiB' for i in range(4)},low_cpu_mem_usage=True,
 attn_implementation='sdpa',local_files_only=True).eval()
metadata.update(load_s=time.time()-started,device_map={k:str(v) for k,v in model.hf_device_map.items()})
input_device=model.get_input_embeddings().weight.device
with torch.inference_mode():
 for name,text in prompts:
  formatted=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  ids=tokenizer.encode(formatted,add_special_tokens=False)
  tensor=torch.tensor([ids],device=input_device)
  result=model(tensor,use_cache=True,output_hidden_states=True,logits_to_keep=1)
  arrays={'token_ids':np.asarray(ids,dtype=np.int64),'last_logits':result.logits[0,-1].float().cpu().numpy()}
  for layer in metadata['selected_hidden_state_indices']:
   arrays[f'hidden_{layer}_last8']=result.hidden_states[layer][0,-8:].float().cpu().numpy()
  cache=result.past_key_values;cached_logits=result.logits[0,-1].float().cpu();del result
  prefix=ids.copy();checks=[];generated=[]
  for step in range(8):
   full=model(torch.tensor([prefix],device=input_device),use_cache=False,logits_to_keep=1)
   logits=full.logits[0,-1].float().cpu();del full
   delta=cached_logits-logits
   rmse=delta.square().mean().sqrt().item();relative=rmse/max(logits.square().mean().sqrt().item(),1e-12)
   cosine=torch.nn.functional.cosine_similarity(cached_logits[None],logits[None]).item()
   next_id=int(logits.argmax());cached_id=int(cached_logits.argmax())
   checks.append(dict(step=step,max_abs_error=float(delta.abs().max()),rmse=rmse,relative_rmse=relative,
      cosine=cosine,greedy_equal=next_id==cached_id,full_argmax=next_id,cached_argmax=cached_id))
   assert torch.isfinite(cached_logits).all() and torch.isfinite(logits).all(),checks[-1]
   arrays[f'cached_logits_{step}']=cached_logits.numpy()
   arrays[f'continuation_logits_{step}']=logits.numpy()
   generated.append(next_id);prefix.append(next_id)
   cached=model(torch.tensor([[next_id]],device=input_device),past_key_values=cache,
     use_cache=True,logits_to_keep=1)
   cache=cached.past_key_values;cached_logits=cached.logits[0,-1].float().cpu();del cached
  arrays['continuation_token_ids']=np.asarray(generated,dtype=np.int64)
  path=out/f'{name}.npz';np.savez_compressed(path,**arrays)
  # Raw little-endian arrays can be consumed without NumPy or Python.
  binary_dir=out/name;binary_dir.mkdir(exist_ok=True)
  manifest={}
  for key,array in arrays.items():
   array=np.ascontiguousarray(array.astype('<i8' if array.dtype.kind=='i' else '<f4'))
   binary=(binary_dir/f'{key}.bin');array.tofile(binary)
   manifest[key]=dict(file=binary.name,shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
  (binary_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  item=dict(name=name,prompt_text=text,formatted_prompt=formatted,input_tokens=len(ids),token_ids=ids,
      continuation_token_ids=generated,continuation_text=tokenizer.decode(generated),checks=checks,
      file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  metadata['fixtures'].append(item);(out/'reference.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
  print(name,len(ids),'max relative RMSE',max(c['relative_rmse'] for c in checks),'greedy agreement',sum(c['greedy_equal'] for c in checks),'/8',flush=True)
  del cache
print('REFERENCE COMPLETE',flush=True)
