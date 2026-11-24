#!/usr/bin/env python3
import argparse, ctypes, json, mmap, os, pathlib, statistics, subprocess, time
parser=argparse.ArgumentParser(description='Queued K3 single-layer expert prefetch comparison (evicts only this model file cache).')
parser.add_argument('--bin-dir',type=pathlib.Path,required=True)
parser.add_argument('--model',type=pathlib.Path,required=True)
parser.add_argument('--output-dir',type=pathlib.Path,required=True)
parser.add_argument('--modes',nargs='+',choices=['resident','mmap','mmap-eager','prefetch','prefetch-random','mmap-prefetch','mmap-normal'],default=['resident','mmap','prefetch','mmap-eager'])
args=parser.parse_args()
root=args.bin_dir.resolve()
model=str(args.model.resolve())
output=args.output_dir.resolve()
output.mkdir(parents=True,exist_ok=True)
os.chdir(root)
os.environ.update(LD_LIBRARY_PATH=str(root),GGML_SPACEMIT_BACKEND='1',GGML_SPACEMIT_WORKERS='8')
subprocess.run(['./test-spacemit-prefetch-plan'],check=True)
libc=ctypes.CDLL(None,use_errno=True)
page=os.sysconf('SC_PAGE_SIZE')
def evict():
    fd=os.open(model,os.O_RDONLY)
    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    size=os.fstat(fd).st_size
    mapping=mmap.mmap(fd,0,access=mmap.ACCESS_COPY)
    address=ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    pages=(size+page-1)//page
    vector=(ctypes.c_ubyte*pages)()
    rc=libc.mincore(ctypes.c_void_p(address),ctypes.c_size_t(size),vector)
    if rc: raise OSError(ctypes.get_errno(),'mincore')
    cached=sum(v&1 for v in vector)*page
    mapping.close();os.close(fd)
    if cached>size*0.01: raise RuntimeError(f'model cache eviction incomplete: {cached}/{size}')
    return cached
modes=list(dict.fromkeys(args.modes))
orders=[modes,list(reversed(modes)),modes[2:]+modes[:2]]
all_results=[]
reference={}
for r,order in enumerate(orders,1):
    for mode in order:
        prefix=str(output/f'matrix-r{r}-{mode}')
        cached=evict()
        print(f'START {prefix} cache_bytes={cached}',flush=True)
        t=time.monotonic()
        with open(prefix+'.stdout','wb') as out,open(prefix+'.log','wb') as err:
            subprocess.run(['./test-spacemit-expert-prefetch',model,mode,prefix,'128'],stdout=out,stderr=err,check=True,timeout=240)
        row=json.load(open(prefix+'.json'))
        if not pathlib.Path('/proc/self/io').exists():
            for field in ['load_read_bytes','first_pp_read_bytes','total_read_bytes']:
                row[field]=None
        row['all_expert_bytes_verified']=mode in ['prefetch','prefetch-random']
        row.update(round=r,cache_before_bytes=cached,process_wall_seconds=time.monotonic()-t)
        for kind in ['first','decode','warm']:
            data=pathlib.Path(prefix+'.'+kind+'.logits').read_bytes()
            if kind not in reference: reference[kind]=data
            if data!=reference[kind]: raise RuntimeError(f'logits mismatch: {prefix} {kind}')
        all_results.append(row)
        (output/'matrix-results.json').write_text(json.dumps(all_results,indent=2)+'\n')
        print('DONE '+json.dumps(row),flush=True)
summary={}
fields=['load_ms','context_ms','first_pp_ms','warm_pp_ms','decode16_ms','load_plus_first_ms','rss_after_load_kib','rss_end_kib','maxrss_kib','submit_us','overlap_ms','wait_ms','ready_ms','process_wall_seconds']
for mode in orders[0]:
    rows=[r for r in all_results if r['mode']==mode]
    summary[mode]={key:{'mean':statistics.mean(r[key] for r in rows),'min':min(r[key] for r in rows),'max':max(r[key] for r in rows)} for key in fields}
(output/'matrix-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: all processes and all first/decode/warm logits match byte-for-byte',flush=True)