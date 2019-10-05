#include <stdio.h>
#include <stdlib.h>
#include <thrust/sort.h>
#include <assert.h>
#include "radix_select.h"


__device__ __forceinline__
int hash(long long a, long long b, long long c, long long shift){
    long long output = b * c + 584783629162121ll;
    output += (a + shift * 584771016211002119ll) * 3223372036854775909ll;
    output += (b + shift * 361415955631027267ll) * 5215415955631027207ll;
    output += (c + shift * 223372036854775859ll) * 8438543016211002467ll;
    // if (output == 0){ // 0 value is reserved for debug. 
    //     output += 1;
    // }
    return output;
}

__global__ void randd(int * p, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    while(idx < n){
        p[idx] = hash(clock(), threadIdx.x, idx, gridDim.x); 
        // p[idx] %= 300; // to make data more dense;
        idx += blockDim.x * gridDim.x;
    }
}


static const int N = 1<<28;
static const int topk = 32;
int origin()
{
    printf("start\n");
    typedef int T; // unsigned
    assert(sizeof(T) == 4); // change this need to change func:randd.
    T *p_d;
    cudaMalloc((void **)&p_d, N * sizeof(T)); 
    T *answer;  
    answer =(T *)malloc(topk * sizeof(T)); 

    randd<<<1024, 1024>>>((int *)p_d, N);
    

    /////////////////////////////////////////////////////////////////////////////////////////
    cudaEvent_t ev1, ev2;
	cudaEventCreate(&ev1);
	cudaEventCreate(&ev2);
    cudaEventRecord(ev1);
    
    radix_select(p_d, N, answer, topk);

    cudaGetLastError();
	cudaEventRecord(ev2);
	cudaDeviceSynchronize();
    float elapse = 0.0f;
    cudaEventElapsedTime(&elapse, ev1, ev2);
    /////////////////////////////////////////////////////////////////////////////////////////
    printf("time for gpu radixfind is %.3f ms\n",elapse);
    // printf("my result:\n");
    // for (int i = 0; i < topk; ++i){ 
    //     printf("%d, %d\n", i, answer[i]);
    // }


    T *p_h;
    cudaEventCreate(&ev1);
	cudaEventCreate(&ev2);
    cudaEventRecord(ev1);
    thrust::sort(thrust::device, p_d, p_d + N); // compare to thrust
    cudaGetLastError();
	cudaEventRecord(ev2);    
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapse, ev1, ev2);
    printf("time for gpu radixsort is %.3f ms\n",elapse);
    p_h = (T *)malloc(N * sizeof(T));  
    cudaMemcpy(p_h, p_d, N * sizeof(T), cudaMemcpyDeviceToHost);
    // printf("thrust result:\n");
    // for (int i = 0; i < topk; ++i){ 
    //     printf("%d, %d\n", i, p_h[i]);
    // }

    free(p_h);   
    cudaFree(p_d);
    printf("err: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 0;                
}

int main(){
    origin();
}