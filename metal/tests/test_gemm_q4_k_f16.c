/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* Copyright (C) 2026 David (Corey) Herrera */
/* Research/reference code, not used in production builds. Portions modelled on
 * ggml (MIT License, Copyright (c) 2023-2026 The ggml authors,
 * https://github.com/ggml-org/ggml), modified. See NOTICE. */
/* Validate + benchmark the large-M prefill GEMM fm_gemm_q4_k_f16 against a CPU
 * F32 reference (dequant of the same Q4_K bytes, then GEMM). F16 tolerance.
 *
 * Usage: test_gemm_q4_k_f16 [M] [N] [K]   (defaults 1369 x 2048 x 1024)
 */
#include "metal/fm_metal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

static uint16_t f32_to_f16(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, out;
    if (exp == 0) { if (!mant) out = sign; else { exp = 127-15+1; while(!(mant&0x400)){mant<<=1;exp--;} mant&=0x3FF; out=sign|(exp<<23)|(mant<<13);} }
    else if (exp == 31) out = sign | 0x7F800000u | (mant<<13);
    else out = sign | ((exp-15+127)<<23) | (mant<<13);
    float f; memcpy(&f,&out,4); return f;
}
static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1000.0+t.tv_nsec/1e6; }
static uint32_t rng=0x9e3779b9u; static uint32_t ur(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static float fr(void){ return (float)(ur()%2000)/1000.0f-1.0f; }
#define QK 144u

static void deq_block(const uint8_t*blk,float*o){
    float d=f16_to_f32(((const uint16_t*)blk)[0]), dmin=f16_to_f32(((const uint16_t*)blk)[1]);
    const uint8_t*sr=blk+4,*qs=blk+16; uint8_t sc[8],mn[8];
    for(int j=0;j<4;j++){sc[j]=sr[j]&63;mn[j]=sr[j+4]&63;}
    for(int j=4;j<8;j++){sc[j]=(sr[j+4]&0xF)|((sr[j-4]>>6)<<4);mn[j]=(sr[j+4]>>4)|((sr[j]>>6)<<4);}
    for(int p=0;p<4;p++){float d1=d*sc[2*p],m1=dmin*mn[2*p],d2=d*sc[2*p+1],m2=dmin*mn[2*p+1];const uint8_t*q=qs+p*32;int o0=p*64;
        for(int l=0;l<32;l++){o[o0+l]=d1*(q[l]&0xF)-m1;o[o0+32+l]=d2*(q[l]>>4)-m2;}}
}

int main(int argc,char**argv){
    int M=argc>1?atoi(argv[1]):1369, N=argc>2?atoi(argv[2]):2048, K=argc>3?atoi(argv[3]):1024;
    if(K%256){fprintf(stderr,"K must be multiple of 256\n");return 1;}
    if(fm_init()||!fm_available()){fprintf(stderr,"Metal unavailable\n");return 77;}
    int nblk=K/256; size_t rs=(size_t)nblk*QK, wb=(size_t)N*rs;
    uint8_t*W=malloc(wb);
    for(int r=0;r<N;r++)for(int b=0;b<nblk;b++){uint8_t*blk=W+(size_t)r*rs+(size_t)b*QK;
        ((uint16_t*)blk)[0]=f32_to_f16(0.02f+0.01f*(float)(ur()%8));((uint16_t*)blk)[1]=f32_to_f16(0.01f*(float)(ur()%8));
        for(int i=4;i<144;i++)blk[i]=(uint8_t)ur();}
    float*X=malloc((size_t)M*K*4); for(int i=0;i<M*K;i++)X[i]=fr()*0.3f;   /* post-RMSNorm scale */

    float*ref=malloc((size_t)M*N*4);
    double c0=now_ms(); float blk[256];
    for(int r=0;r<N;r++)for(int b=0;b<nblk;b++){deq_block(W+(size_t)r*rs+(size_t)b*QK,blk);
        for(int m=0;m<M;m++){float acc=(b==0)?0.f:ref[(size_t)m*N+r];const float*xm=X+(size_t)m*K+b*256;
            for(int l=0;l<256;l++)acc+=blk[l]*xm[l]; ref[(size_t)m*N+r]=acc;}}
    double cpu_ms=now_ms()-c0;

    fm_buffer*bW=fm_buffer_from_ptr(W,wb), *bX=fm_buffer_from_ptr(X,(size_t)M*K*4), *bO=fm_buffer_alloc((size_t)M*N*4);
    fm_gemm_q4_k_f16(bW,bX,bO,M,N,K,(int)rs); fm_sync();  /* warm */
    int iters=20; double g0=now_ms();
    for(int i=0;i<iters;i++) fm_gemm_q4_k_f16(bW,bX,bO,M,N,K,(int)rs);
    fm_sync(); double gpu_ms=(now_ms()-g0)/iters;

    float*out=fm_buffer_data(bO); double maxabs=0,sumsq=0;
    for(int i=0;i<M*N;i++){double e=fabs((double)out[i]-ref[i]);if(e>maxabs)maxabs=e;sumsq+=(double)ref[i]*ref[i];}
    double rms=sqrt(sumsq/((double)M*N)), rel=maxabs/(rms+1e-9);
    printf("gemm_q4_k_f16 M=%d N=%d K=%d\n",M,N,K);
    printf("  err/RMS=%.4g (RMS=%.3g) -> %s\n", rel, rms, rel<2e-2?"PASS":"FAIL");
    printf("  GPU %.2f ms/call   CPU-ref(F32) %.0f ms\n", gpu_ms, cpu_ms);
    fm_buffer_free(bW);fm_buffer_free(bX);fm_buffer_free(bO);free(W);free(X);free(ref);
    return rel<2e-2?0:1;
}
