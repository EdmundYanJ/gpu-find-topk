#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include<time.h>
#include <algorithm>
#include <iostream>
namespace cg = cooperative_groups;

#ifndef QUICKSORT_H
#define QUICKSORT_H

#define QSORT_BLOCKSIZE_SHIFT   9
#define QSORT_BLOCKSIZE         (1 << QSORT_BLOCKSIZE_SHIFT)
#define BITONICSORT_LEN         1024            // Must be power of 2!
#define QSORT_MAXDEPTH          32              // Will force final bitonic stage at depth QSORT_MAXDEPTH+1


////////////////////////////////////////////////////////////////////////////////
// The algorithm uses several variables updated by using atomic operations.
////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(128) qsortAtomicData_t
{
	volatile unsigned int lt_offset;    // Current output offset for <pivot
	volatile unsigned int gt_offset;    // Current output offset for >pivot
	volatile unsigned int sorted_count; // Total count sorted, for deciding when to launch next wave
	volatile unsigned int index;        // Ringbuf tracking index. Can be ignored if not using ringbuf.
} qsortAtomicData;

////////////////////////////////////////////////////////////////////////////////
// A ring-buffer for rapid stack allocation
////////////////////////////////////////////////////////////////////////////////
typedef struct qsortRingbuf_t
{
	volatile unsigned int head;         // Head pointer - we allocate from here
	volatile unsigned int tail;         // Tail pointer - indicates last still-in-use element
	volatile unsigned int count;        // Total count allocated
	volatile unsigned int max;          // Max index allocated
	unsigned int stacksize;             // Wrap-around size of buffer (must be power of 2)
	volatile void *stackbase;           // Pointer to the stack we're allocating from
} qsortRingbuf;

// Stack elem count must be power of 2!
#define QSORT_STACK_ELEMS   8*1024*1024 // One million stack elements is a HUGE number.

__global__ void qsort_warp(unsigned *indata, unsigned *outdata, unsigned int len, qsortAtomicData *atomicData, qsortRingbuf *ringbuf, unsigned int source_is_indata, unsigned int depth, int k, int *target_gpu);

#endif // QUICKSORT_H



////////////////////////////////////////////////////////////////////////////////
// Inline PTX call to return index of highest non-zero bit in a word
////////////////////////////////////////////////////////////////////////////////
static __device__ __forceinline__ unsigned int __qsflo(unsigned int word)
{
	unsigned int ret;
	asm volatile("bfind.u32 %0, %1;" : "=r"(ret) : "r"(word));
	return ret;
}

template< typename T >
static __device__ T *ringbufAlloc(qsortRingbuf *ringbuf)
{
	// Wait for there to be space in the ring buffer. We'll retry only a fixed
	// number of times and then fail, to avoid an out-of-memory deadlock.
	unsigned int loop = 10000;

	while (((ringbuf->head - ringbuf->tail) >= ringbuf->stacksize) && (loop-- > 0));

	if (loop == 0)
		return NULL;

	// Note that the element includes a little index book-keeping, for freeing later.
	unsigned int index = atomicAdd((unsigned int *)&ringbuf->head, 1);
	T *ret = (T *)(ringbuf->stackbase) + (index & (ringbuf->stacksize - 1));
	ret->index = index;

	return ret;
}

template< typename T >
static __device__ void ringbufFree(qsortRingbuf *ringbuf, T *data)
{
	unsigned int index = data->index;       // Non-wrapped index to free
	unsigned int count = atomicAdd((unsigned int *)&(ringbuf->count), 1) + 1;
	unsigned int max = atomicMax((unsigned int *)&(ringbuf->max), index + 1);

	// Update the tail if need be. Note we update "max" to be the new value in ringbuf->max
	if (max < (index + 1)) max = index + 1;

	if (max == count)
		atomicMax((unsigned int *)&(ringbuf->tail), count);
}

__global__ void qsort_warp(unsigned *indata,
	unsigned *outdata,
	unsigned int offset,
	unsigned int len,
	qsortAtomicData *atomicData,
	qsortRingbuf *atomicDataStack,
	unsigned int source_is_indata,
	unsigned int depth,
	int k,
	unsigned *target_gpu)
{
	// Handle to thread block group
	cg::thread_block cta = cg::this_thread_block();
	// Find my data offset, based on warp ID
	unsigned int thread_id = threadIdx.x + (blockIdx.x << QSORT_BLOCKSIZE_SHIFT);
	//unsigned int warp_id = threadIdx.x >> 5;   // Used for debug only
	unsigned int lane_id = threadIdx.x & (warpSize - 1);

	// Exit if I'm outside the range of sort to be done
	if (thread_id >= len)
		return;

	unsigned pivot = indata[offset + len / 2];
	unsigned data = indata[offset + thread_id];

	cg::coalesced_group active = cg::coalesced_threads();
	unsigned int greater = (data > pivot);
	//�Ƚϱ�ѡȡֵ��pivot��Ҫ�����
	unsigned int gt_mask = active.ballot(greater);
	//ballot ���� __ballot_sync �������߳����е�ÿ���߳�
	//__ballot_sync(0xFFFFFFFF, predicate)ÿ���߳�����λ ��mask(0xFFFFFFFF)�������� ����Ϊ1��

	if (gt_mask == 0)//˵�����е�ֵ����pivot ��<=��
	{
		greater = (data >= pivot);
		gt_mask = active.ballot(greater);    // Must re-ballot for adjusted comparator
	}

	unsigned int lt_mask = active.ballot(!greater);
	unsigned int gt_count = __popc(gt_mask);//����64λ����������Ϊ1��λ������������߳����д���piovt������
	unsigned int lt_count = __popc(lt_mask);//����64λ����������Ϊ1��λ������������߳�����С��piovt������

	// Atomically adjust the lt_ and gt_offsets by this amount. Only one thread need do this. Share the result using shfl
	unsigned int lt_offset, gt_offset;

	if (lane_id == 0)//�߳�����ƫ��Ϊ0���߳�
	{
		if (lt_count > 0)//����ƫ��ֵ
			lt_offset = atomicAdd((unsigned int *)&atomicData->lt_offset, lt_count);

		if (gt_count > 0)
			gt_offset = len - (atomicAdd((unsigned int *)&atomicData->gt_offset, gt_count) + gt_count);
	}
	//��ϴ�� ÿ����ͨ���߳�0�Ľ���ƫ��ֵ
	lt_offset = active.shfl((int)lt_offset, 0);   // Everyone pulls the offsets from lane 0
	gt_offset = active.shfl((int)gt_offset, 0);

	unsigned lane_mask_lt;
	//����߳���warp�ڵ�λ�õ�����
	//��λ��ǰ�Ķ����ƶ���1 ��lane_id=5�����õĶ����ƣ�11111
	asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
	unsigned int my_mask = greater ? gt_mask : lt_mask;
	unsigned int my_offset = __popc(my_mask & lane_mask_lt);
	//����64λ����������Ϊ1��λ���� �����������ڱ��߳�ǰ����һ��'>'��<=��pivot�ĸ���
	
	// Move data.
	my_offset += greater ? gt_offset : lt_offset;
	outdata[offset + my_offset] = data;

	// Count up if we're the last warp in. If so, then Kepler will launch the next
	// set of sorts directly from here.
	if (lane_id == 0)
	{
		// Count "elements written". If I wrote the last one, then trigger the next qsorts
		unsigned int mycount = lt_count + gt_count;

		if (atomicAdd((unsigned int *)&atomicData->sorted_count, mycount) + mycount == len)
		{
			// We're the last warp to do any sorting. Therefore it's up to us to launch the next stage.
			unsigned int lt_len = atomicData->lt_offset;
			unsigned int gt_len = atomicData->gt_offset;

			cudaStream_t lstream, rstream;
			cudaStreamCreateWithFlags(&lstream, cudaStreamNonBlocking);
			cudaStreamCreateWithFlags(&rstream, cudaStreamNonBlocking);

			// Begin by freeing our atomicData storage. It's better for the ringbuffer algorithm
			// if we free when we're done, rather than re-using (makes for less fragmentation).
			ringbufFree<qsortAtomicData>(atomicDataStack, atomicData);

			// Exceptional case: if "lt_len" is zero, then all values in the batch
			// are equal. We are then done (may need to copy into correct buffer, though)
			if (lt_len == 0)
			{
				if (source_is_indata)
					cudaMemcpyAsync(indata + offset, outdata + offset, gt_len * sizeof(unsigned), cudaMemcpyDeviceToDevice, lstream);

				return;
			}

			if (gt_len == k) {
				return;
			}
			// Start with lower half first
			else if (gt_len < k) {
				if (depth == 24) return;
				if (lt_len > 1) {
					// Launch another quicksort. We need to allocate more storage for the atomic data.
					if ((atomicData = ringbufAlloc<qsortAtomicData>(atomicDataStack)) == NULL)
						printf("Stack-allocation error. Failing left child launch.\n");
					else
					{
						atomicData->lt_offset = atomicData->gt_offset = atomicData->sorted_count = 0;
						unsigned int numblocks = (unsigned int)(lt_len + (QSORT_BLOCKSIZE - 1)) >> QSORT_BLOCKSIZE_SHIFT;
						qsort_warp << < numblocks, QSORT_BLOCKSIZE, 0, lstream >> > (outdata, indata, offset, lt_len, atomicData, atomicDataStack, !source_is_indata, depth + 1, k - gt_len, target_gpu);
					}

				}

				else if (source_is_indata && (lt_len == 1))
					indata[offset] = outdata[offset];

				if (cudaPeekAtLastError() != cudaSuccess)
					printf("Left-side launch fail: %s\n", cudaGetErrorString(cudaGetLastError()));
			}
			else {
				target_gpu[0] = pivot;
				//printf("%d\n", target_gpu[0]);
				if (depth == 24) return;
				if (gt_len > 1) {
					// Allocate new atomic storage for this launch
					if ((atomicData = ringbufAlloc<qsortAtomicData>(atomicDataStack)) == NULL)
						printf("Stack allocation error! Failing right-side launch.\n");
					else
					{
						atomicData->lt_offset = atomicData->gt_offset = atomicData->sorted_count = 0;
						unsigned int numblocks = (unsigned int)(gt_len + (QSORT_BLOCKSIZE - 1)) >> QSORT_BLOCKSIZE_SHIFT;
						qsort_warp << < numblocks, QSORT_BLOCKSIZE, 0, rstream >> > (outdata, indata, offset + lt_len, gt_len, atomicData, atomicDataStack, !source_is_indata, depth + 1, k, target_gpu);
					}
				}

				else if (source_is_indata && (gt_len == 1))
					indata[offset + lt_len] = outdata[offset + lt_len];

				if (cudaPeekAtLastError() != cudaSuccess)
					printf("Right-side launch fail: %s\n", cudaGetErrorString(cudaGetLastError()));
			}
		}
	}
}

float run_quicksort_cdp(unsigned *gpudata, unsigned *scratchdata, unsigned int count, cudaStream_t stream, unsigned *target_gpu, unsigned *target_cpu, int k)
{
	unsigned int stacksize = QSORT_STACK_ELEMS;

	// This is the stack, for atomic tracking of each sort's status
	qsortAtomicData *gpustack;
	cudaMalloc((void **)&gpustack, stacksize * sizeof(qsortAtomicData));
	cudaMemset(gpustack, 0, sizeof(qsortAtomicData));     // Only need set first entry to 0

	// Create the memory ringbuffer used for handling the stack.
	// Initialise everything to where it needs to be.
	qsortRingbuf buf;
	qsortRingbuf *ringbuf;
	cudaMalloc((void **)&ringbuf, sizeof(qsortRingbuf));
	buf.head = 1;           // We start with one allocation
	buf.tail = 0;
	buf.count = 0;
	buf.max = 0;
	buf.stacksize = stacksize;
	buf.stackbase = gpustack;
	cudaMemcpy(ringbuf, &buf, sizeof(buf), cudaMemcpyHostToDevice);

	// Timing events...
	cudaEvent_t ev1, ev2;
	cudaEventCreate(&ev1);
	cudaEventCreate(&ev2);
	cudaEventRecord(ev1);

	// Now we trivially launch the qsort kernel
	unsigned int numblocks = (unsigned int)(count + (QSORT_BLOCKSIZE - 1)) >> QSORT_BLOCKSIZE_SHIFT;
	qsort_warp << < numblocks, QSORT_BLOCKSIZE, 0, stream >> > (gpudata, scratchdata, 0U, count, gpustack, ringbuf, true, 0, k, target_gpu);

	cudaGetLastError();
	cudaEventRecord(ev2);
	cudaDeviceSynchronize();
	cudaMemcpy(target_cpu, target_gpu, 3 * sizeof(unsigned), cudaMemcpyDeviceToHost);

	float elapse = 0.0f;


	cudaEventElapsedTime(&elapse, ev1, ev2);

	// Sanity check that the stack allocator is doing the right thing
	cudaMemcpy(&buf, ringbuf, sizeof(*ringbuf), cudaMemcpyDeviceToHost);
	// Release our stack data once we're done
	cudaFree(ringbuf);
	cudaFree(gpustack);

	return elapse;
}

int cpu_quickfind(unsigned *a, int left, int right, int k) {
	int i = left;
	int j = right;
	int mark = a[(left + right) / 2];

	//���ŵķ���ʵ���ڱ�mark��߶���С�����ģ��ұ߶��Ǵ������� 
	while (i < j) {
		while (i < j && a[j] >= mark)
			--j;
		if (i < j)
			a[i++] = a[j];

		while (i < j && a[i] <= mark)
			++i;
		if (i < j)
			a[j--] = a[i];
	}
	a[i] = mark;

	//�ڱ��Ҳ����������ָ���
	int big_num = right - i;

	//����ڱ��պ��ǵ�K�����
	if (k - big_num - 1 == 0)
		return mark;
	else if (k - big_num - 1 > 0) {
		//����Ҳ����ָ�������K�����������ҵ�k-big_num-1�����
		return cpu_quickfind(a, left, i - 1, k - big_num - 1);
	}
	else {
		//����Ҳ����ָ�����K�࣬�����Ҳ��ҵ�K�����
		return cpu_quickfind(a, i + 1, right, k);
	}
}

void rand(unsigned * arr, int count) {
	srand(time(NULL));
	for (int i = 0; i < count; i++)
		arr[i] = 100000000 / RAND_MAX * rand();
}
int cmp(const void *a, const void *b)
{
	return *(int *)a - *(int *)b;
}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int run_qsort(unsigned int size,  int debug, int loop, int verbose)
{
	// Create and set up our test
	unsigned *gpudata, *scratchdata;
	cudaMalloc((void **)&gpudata, size * sizeof(unsigned));
	cudaMalloc((void **)&scratchdata, size * sizeof(unsigned));

	// Create CPU data.
	unsigned *data = new unsigned[size];
	unsigned *data1 = new unsigned[size];
	unsigned int min = loop ? loop : size;
	unsigned int max = size;
	loop = (loop == 0) ? 1 : loop;
	int k = 50;
	
	//rand(data, size);
	for (int i = 0; i < size; i++)
		data1[i] = data[i]=i;

	cudaMemcpy(gpudata, data, size * sizeof(unsigned), cudaMemcpyHostToDevice);

	// So we're now populated and ready to go! We size our launch as
	// blocks of up to BLOCKSIZE threads, and appropriate grid size.
	// One thread is launched per element.
	unsigned *target_gpu, *target_cpu;
	target_cpu = (unsigned *)malloc(3 * sizeof(unsigned));
	cudaMalloc((void **)&target_gpu, 3 * sizeof(unsigned));
	clock_t start, end;
	start = clock();
	int target = cpu_quickfind(data, 0, size - 1, k);
	end = clock();
	printf("cpu target:%d\n", target);
	double time = (double)(end - start) / CLOCKS_PER_SEC*1000;
	printf("time for cpu quicksort is %.3f ms\n", time);

	float elapse;
	elapse = run_quicksort_cdp(gpudata, scratchdata, size, NULL, target_gpu, target_cpu, k);
	cudaDeviceSynchronize();
	// Copy back the data and verify correct sort
	//cudaMemcpy(data, gpudata, size * sizeof(unsigned), cudaMemcpyDeviceToHost);
	int point[2000], i, j = 0;
	//printf("%d\n", target_cpu[0]);
	for (i = 0; i < size; i++) {
		if (data1[i] >= target_cpu[0])
			point[j++] = data1[i];
	}
	std::qsort(point, j, sizeof(int), cmp);
	printf("gpu target:%d\n", point[j-k+1]);
	printf("time for gpu quicksort is %.3f ms (%.3f Melems/sec)\n", elapse, (float)size / (elapse*1000.0f));
	fflush(stdout);
	

	// Release everything and we're done
	cudaFree(scratchdata);
	cudaFree(gpudata);
	delete(data);
	return 0;
}

static void usage()
{
	printf("Syntax: qsort [-size=<num>] [-seed=<num>] [-debug] [-loop-step=<num>] [-verbose]\n");
	printf("If loop_step is non-zero, will run from 1->array_len in steps of loop_step\n");
}


// Host side entry
int main(int argc, char *argv[])
{
	int size = 2e8;     // TODO: make this 1e6
	int debug = 0;
	int loop = 0;
	int verbose = 0;

	printf("Running qsort on %d elements\n", size);
	run_qsort(size, debug, loop, verbose);

	exit(EXIT_SUCCESS);
}