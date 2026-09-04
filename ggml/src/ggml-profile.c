#include "ggml-profile.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef GGML_BUILD_PROFILE
#if defined(_WIN32)
int g_current_token_idx = -1;
void ggml_set_current_token_idx(int idx) { g_current_token_idx = idx; }
void ggml_profile_init_trace_file(void) {}
void ggml_profile_flush_trace(void) {}
void ggml_trace_log_begin(const char * n, const char * c, const char * a) { (void)n;(void)c;(void)a; }
void ggml_trace_log_end(const char * n, const char * c, const char * a) { (void)n;(void)c;(void)a; }
void ggml_profile_log_op_begin(struct ggml_tensor * t, int i, int n) { (void)t;(void)i;(void)n; }
void ggml_profile_log_op_end(struct ggml_tensor * t, int i, int n) { (void)t;(void)i;(void)n; }
void ggml_profile_flush_tls(void) {}
#else
#include <pthread.h>
#include <time.h>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif
static int g_trace_enabled = 0;
static pthread_once_t g_trace_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_trace_mutex = PTHREAD_MUTEX_INITIALIZER;
static FILE * g_trace_file = NULL;
#define TRACE_BUFFER_SIZE (64 * 1024)
__thread char g_trace_tls_buffer[TRACE_BUFFER_SIZE];
__thread size_t g_trace_tls_offset = 0;
__thread int g_trace_worker_named = 0;
int g_current_token_idx = -1;
void ggml_set_current_token_idx(int idx) { g_current_token_idx = idx; }
void ggml_profile_flush_trace(void);
static void ggml_trace_parse_env(void) {
    const char * e = getenv("GGML_TRACE");
    g_trace_enabled = e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y');
    if (g_trace_enabled) atexit(ggml_profile_flush_trace);
}
#define TRACE_ENABLED() (pthread_once(&g_trace_once, ggml_trace_parse_env), g_trace_enabled)
static inline long long now_us(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (long long)ts.tv_sec*1000000LL+ts.tv_nsec/1000; }
static inline void init_locked(void) { if (!g_trace_file) { g_trace_file=fopen("ggml_trace.json","w"); if(g_trace_file) fputs("[\n",g_trace_file); } }
static inline void flush_tls_locked(void) { if(g_trace_tls_offset && g_trace_file) { fwrite(g_trace_tls_buffer,1,g_trace_tls_offset,g_trace_file); g_trace_tls_offset=0; } }
static inline void trace_write(const char * s) { if(!TRACE_ENABLED()) return; size_t n=strlen(s); if(g_trace_tls_offset+n>=TRACE_BUFFER_SIZE) { pthread_mutex_lock(&g_trace_mutex); init_locked(); flush_tls_locked(); pthread_mutex_unlock(&g_trace_mutex); } memcpy(g_trace_tls_buffer+g_trace_tls_offset,s,n); g_trace_tls_offset+=n; }
static inline const char * tname(struct ggml_tensor * t) { return t&&t->name[0]?t->name:"N/A"; }
static void esc(const char * s,char * d,size_t z) { size_t i=0; for(;*s&&i+2<z;s++){if(*s=='"'||*s=='\\')d[i++]='\\';d[i++]=*s;}d[i]=0; }
static void tensor_info(struct ggml_tensor * t,const char * p,char * b,size_t z) { if(!t){snprintf(b,z,"\"%s_exists\":false",p);return;} char n[64];esc(tname(t),n,sizeof(n));snprintf(b,z,"\"%s_shape\":\"[%ld,%ld,%ld,%ld]\",\"%s_strides\":\"[%ld,%ld,%ld,%ld]\",\"%s_type\":\"%s\",\"%s_name\":\"%s\"",p,t->ne[0],t->ne[1],t->ne[2],t->ne[3],p,t->nb[0],t->nb[1],t->nb[2],t->nb[3],p,ggml_type_name(t->type),p,n); }
static void op_args(struct ggml_tensor * t, char * b,size_t z) { char o[256],a[256],c[256],n[64]; tensor_info(t,"out",o,sizeof(o));tensor_info(t?t->src[0]:NULL,"src0",a,sizeof(a));tensor_info(t?t->src[1]:NULL,"src1",c,sizeof(c));esc(t?ggml_op_name(t->op):"N/A",n,sizeof(n));snprintf(b,z,"\"op_name\":\"%s\",%s,%s,%s",n,o,a,c); }
static void lane_event(const char * name,const char * cat,char ph,const char * args,int pid,int tid) { char b[1400]; snprintf(b,sizeof(b),"{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"%c\",\"ts\":%lld,\"pid\":%d,\"tid\":%d,\"args\":{%s}},\n",name,cat,ph,now_us(),pid,tid,args?args:"");trace_write(b); }
void ggml_trace_log_begin(const char * n,const char * c,const char * a) { if(TRACE_ENABLED()) lane_event(n,c,'B',a,1,0); }
void ggml_trace_log_end(const char * n,const char * c,const char * a) { if(TRACE_ENABLED()) lane_event(n,c,'E',a,1,0); }
void ggml_profile_log_op_begin(struct ggml_tensor * t,int ith,int nth) { if(!TRACE_ENABLED())return; if(!g_trace_worker_named){char b[160];snprintf(b,sizeof(b),"{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":2,\"tid\":%d,\"args\":{\"name\":\"ggml-worker-%d\"}},\n",ith,ith);trace_write(b);g_trace_worker_named=1;} char a[1100],n[128],tn[64];op_args(t,a,sizeof(a));size_t l=strlen(a);snprintf(a+l,sizeof(a)-l,",\"ith\":%d,\"nth\":%d",ith,nth);esc(tname(t),tn,sizeof(tn));snprintf(n,sizeof(n),"%s (%s)",ggml_op_name(t->op),tn);lane_event(n,"Operator",'B',a,2,ith); }
void ggml_profile_log_op_end(struct ggml_tensor * t,int ith,int nth) { (void)nth; if(!TRACE_ENABLED())return; char n[128],tn[64];esc(tname(t),tn,sizeof(tn));snprintf(n,sizeof(n),"%s (%s)",ggml_op_name(t->op),tn);lane_event(n,"Operator",'E',NULL,2,ith); }
static void ggml_profile_log_backend_op(struct ggml_tensor * t,int ith,int nth,char ph) {
    if (!TRACE_ENABLED()) return;
    char n[128], tn[64], a[1100];
    esc(tname(t), tn, sizeof(tn));
    snprintf(n, sizeof(n), "%s (%s)", ggml_op_name(t->op), tn);
    if (ph == 'B') {
        op_args(t, a, sizeof(a));
        size_t l = strlen(a);
        snprintf(a + l, sizeof(a) - l, ",\"ith\":%d,\"nth\":%d,\"backend\":\"spacemit\"", ith, nth);
        lane_event(n, "SpacemitOperator", 'B', a, 3, ith);
    } else {
        lane_event(n, "SpacemitOperator", 'E', NULL, 3, ith);
    }
}
void ggml_profile_log_spacemit_op_begin(struct ggml_tensor * t,int ith,int nth) { ggml_profile_log_backend_op(t,ith,nth,'B'); }
void ggml_profile_log_spacemit_op_end(struct ggml_tensor * t,int ith,int nth) { ggml_profile_log_backend_op(t,ith,nth,'E'); }
void ggml_profile_init_trace_file(void) { if(!TRACE_ENABLED())return; pthread_mutex_lock(&g_trace_mutex);init_locked();pthread_mutex_unlock(&g_trace_mutex); }
void ggml_profile_flush_tls(void) { if(!TRACE_ENABLED())return; pthread_mutex_lock(&g_trace_mutex);init_locked();flush_tls_locked();pthread_mutex_unlock(&g_trace_mutex); }
void ggml_profile_flush_trace(void) { if(!TRACE_ENABLED())return; pthread_mutex_lock(&g_trace_mutex);if(g_trace_file){flush_tls_locked();long p=ftell(g_trace_file);if(p>=2)fseek(g_trace_file,p-2,SEEK_SET);fputs("\n]\n",g_trace_file);fclose(g_trace_file);g_trace_file=NULL;}pthread_mutex_unlock(&g_trace_mutex); }
#endif
#endif
