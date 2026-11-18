#!/usr/bin/env python3
"""Execute production CPY guard/wrapper with a bounds-checking transpose oracle.

Host-only: this validates dispatch layout and all addresses/bytes, not RVV assembly,
SDK, backend graph execution or scheduler fallback. Native test-backend-ops remains
required. Run with python3 tests/spacemit-review/test_cpy_transpose_bounds.py.
"""
import pathlib
import resource
import subprocess
import tempfile

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

root = pathlib.Path(__file__).resolve().parents[2]
ime = (root / 'ggml/src/ggml-spacemit/ime.cpp').read_text()
rvv = (root / 'ggml/src/ggml-spacemit/rvv_kernels.cpp').read_text()

def function(text, name):
    start = text.index(name)
    start = text.rfind('\n', 0, start) + 1
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]

guard = function(ime, 'static bool spacemit_cpy_dense_transpose_supported(')
wrapper = function(rvv, 'void forward_cpy_with_permute(')
assert ime.count('if (spacemit_cpy_dense_transpose_supported(op))') == 2
source = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
using shape = std::array<int64_t, 4>;
enum { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_I32, GGML_TYPE_BF16, OTHER };
struct ggml_tensor {
 int type; shape ne; std::array<size_t,4> nb; void * data = nullptr;
 ggml_tensor * src[1] = {nullptr};
};
namespace ggml { namespace spacemit { struct context { int ith, nth; }; } }
size_t ggml_type_size(int t) { return t==GGML_TYPE_F16 || t==GGML_TYPE_BF16 ? 2 : 4; }
bool ggml_are_same_shape(const ggml_tensor *a,const ggml_tensor *b) { return a->ne==b->ne; }
bool ggml_is_contiguous(const ggml_tensor *a) {
 size_t n=ggml_type_size(a->type);
 for(int i=0;i<4;i++) { if(a->ne[i]>1 && a->nb[i]!=n) return false; n*=a->ne[i]; }
 return true;
}
size_t span(const ggml_tensor & a) {
 size_t n=ggml_type_size(a.type);
 for(int i=0;i<4;i++) n+=(a.ne[i]-1)*a.nb[i];
 return n;
}
ggml_tensor dense(int type, shape ne) {
 ggml_tensor a{}; a.type=type; a.ne=ne; size_t n=ggml_type_size(type);
 for(int i=0;i<4;i++) { a.nb[i]=n; n*=ne[i]; }
 return a;
}
ggml_tensor permute(ggml_tensor a, shape axes) {
 ggml_tensor b=a;
 for(int i=0;i<4;i++) { b.ne[axes[i]]=a.ne[i]; b.nb[axes[i]]=a.nb[i]; }
 return b;
}
size_t checks=0;
// Exact scalar semantics of rvv_transposed_s{16,32}_mn_to_nm. The production
// wrapper supplies all dimensions and strides; every memory access is checked.
void permute_transpose_impl(const ggml_tensor *s, ggml_tensor *d, int64_t batch,
 int64_t m,int64_t n,int64_t bs,int64_t ms,int64_t ns,int64_t ds,int ith,int nth) {
 const size_t sz=ggml_type_size(s->type);
 assert(ns==int64_t(sz));
 for(int64_t b=ith;b<batch;b+=nth) for(int64_t r=0;r<m;r++) for(int64_t c=0;c<n;c++) {
   size_t si=b*bs+r*ms+c*sz, di=b*bs+c*ds+r*sz;
   assert(si+sz<=span(*s)); assert(di+sz<=span(*d));
   memcpy((char*)d->data+di,(char*)s->data+si,sz); ++checks;
 }
}
'''+guard+'\n'+wrapper+r'''
void run(ggml_tensor s, ggml_tensor d, bool accept) {
 d.src[0]=&s;
 assert(spacemit_cpy_dense_transpose_supported(&d)==accept);
 if(!accept) return;
 size_t sz=ggml_type_size(s.type), bytes=span(s);
 std::vector<uint8_t> in(bytes+32), out(span(d)+32), ref(span(d)+32,0xa5);
 for(size_t i=0;i<bytes;i++) in[i]=(i*31+i/7)%251;
 std::fill(in.begin()+bytes,in.end(),0x6d);
 for(int64_t w=0;w<d.ne[3];w++) for(int64_t z=0;z<d.ne[2];z++)
 for(int64_t y=0;y<d.ne[1];y++) for(int64_t x=0;x<d.ne[0];x++) {
   size_t si=x*s.nb[0]+y*s.nb[1]+z*s.nb[2]+w*s.nb[3];
   size_t di=x*d.nb[0]+y*d.nb[1]+z*d.nb[2]+w*d.nb[3];
   memcpy(ref.data()+di,in.data()+si,sz);
 }
 auto saved=in;
 for(int nth:{1,4}) {
   std::fill(out.begin(),out.end(),0xa5); s.data=in.data(); d.data=out.data();
   for(int ith=0;ith<nth;ith++) { ggml::spacemit::context ctx{ith,nth}; forward_cpy_with_permute(ctx,&d); }
   assert(out==ref); assert(in==saved);
 }
}
int main() {
 // The exact 14 upstream failing cases, in the log's order.
 struct test {int type; shape ne; shape p; bool padded;};
 std::vector<test> failures={
 {GGML_TYPE_F16,{256,2,3,4},{1,0,2,3},false},
 {GGML_TYPE_F32,{256,2,3,4},{1,0,2,3},false},
 {GGML_TYPE_F16,{256,4,3,1},{1,0,2,3},false},
 {GGML_TYPE_F32,{256,4,3,1},{1,0,2,3},false},
 {GGML_TYPE_F32,{256,4,3,3},{1,0,2,3},false},
 {GGML_TYPE_BF16,{256,4,3,1},{1,0,2,3},false},
 {GGML_TYPE_F16,{256,4,1,1},{1,0,2,3},false},
 {GGML_TYPE_F32,{256,4,1,1},{1,0,2,3},false},
 {GGML_TYPE_I32,{256,4,1,1},{1,0,2,3},false},
 {GGML_TYPE_I32,{256,1,4,1},{1,2,0,3},false},
 {GGML_TYPE_F32,{256,1,4,1},{1,2,0,3},false},
 {GGML_TYPE_F32,{2,2097121,1,1},{1,0,2,3},false},
 {GGML_TYPE_F32,{64,16,1,1},{1,2,0,3},true},
 {GGML_TYPE_F16,{64,16,1,1},{1,2,0,3},true}};
 int i=0;
 for(auto t:failures) {
   auto s=permute(dense(t.type,t.ne),t.p); auto d=dense(t.type,s.ne);
   if(t.padded) { d=dense(t.type,{32,64,16,1}); d.ne={1,64,16,1}; }
   run(s,d,!t.padded);
   printf("upstream failure %02d: %s\n",++i,t.padded?"guard rejects; CPU fallback required":"exact bytes and bounds PASS, workers 1/4");
 }
 size_t cases=0, accepted=0;
 for(int type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_I32,GGML_TYPE_BF16})
 for(int a:{1,2,3,7,8,9,31,32,33,127,128,129,257}) for(int b:{1,2,3,8,17,65})
 for(int z:{1,2,3}) for(int w:{1,2}) {
   auto s=permute(dense(type,{a,b,z,w}),{1,0,2,3}); auto d=dense(type,s.ne);
   run(s,d,s.nb[0]!=s.nb[1]); ++cases; accepted += s.nb[0]!=s.nb[1];
   auto bad=d; bad.nb[1]+=ggml_type_size(type); if(bad.ne[1]>1) run(s,bad,false);
   bad=d; bad.type=OTHER; run(s,bad,false);
   auto bs=s; bs.nb[0]+=ggml_type_size(type); run(bs,d,false);
   if(z*w>1) { bs=s; bs.nb[2]+=ggml_type_size(type); run(bs,d,false); }
   if(w>1) { bs=s; bs.nb[3]+=ggml_type_size(type); run(bs,d,false); }
 }
 // Ordinary copies must not accidentally enter the transpose path. They retain
 // the existing F32 generic/conversion paths or are already unsupported.
 for(int type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_I32,GGML_TYPE_BF16}) {
   auto s=dense(type,{7,5,3,2}); run(s,s,false);
   auto tr=permute(s,{1,0,2,3}); run(tr,dense(type,{1,35,3,2}),false);
   auto d=dense(type,tr.ne); d.nb[0]*=2; run(tr,d,false);
 }
 printf("%zu shape configurations PASS (%zu optimized, %zu degenerate fallback); %zu checked element copies; negative guards PASS\n",cases,accepted,cases-accepted,checks);
}
'''
with tempfile.TemporaryDirectory(prefix='spacemit-cpy-bounds-') as tmp:
    cpp = pathlib.Path(tmp) / 'check.cpp'
    cpp.write_text(source)
    binary = pathlib.Path(tmp) / 'check'
    subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', str(cpp), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
    # Mutation control: the uncorrected production wrapper must fail the same
    # address oracle, proving this is not merely a test of the new guard.
    old = source.replace('int64_t m     = src0->ne[0];\n    int64_t n     = src0->ne[1];',
                         'int64_t m     = src0->ne[1];\n    int64_t n     = src0->ne[0];')
    assert old != source
    cpp.write_text(old)
    subprocess.run(['c++', '-std=c++17', '-O2', str(cpp), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True)
    assert result.returncode != 0, 'upstream dimension-swap mutation unexpectedly passed'
    print('Upstream dimension-swap mutation: rejected by address oracle as expected')
