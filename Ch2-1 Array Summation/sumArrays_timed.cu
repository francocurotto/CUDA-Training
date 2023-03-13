#include <stdio.h>
#include <sys/time.h>

double cpuSeconds() {
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec + t.tv_usec*1e-6;
}

void sumArrayOnHost(float *A, float *B, float *C, const int N) {
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<N) C[i] = A[i] + B[i];
}

void initialData(float *ip, int size) {
    // generate different seed for random number
    time_t t;
    srand ((unsigned int) time(&t));
    
    for (int i=0; i<size; i++) {
        ip[i] = (float) (rand() & 0xFF)/10.0f;
    }
}

void checkResult(float *C, float *D, const int N) {
    bool match = 1;
    double epsilon = 1.0e-8;
    
    for (int i=0; i<N; i++){
        if (abs(C[i]-D[i]) > epsilon) {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", C[i], D[i], i);
            break;
        }
    }
    if (match) printf("Arrays match.\n\n");
}

int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // define array size
    //int nElem = 1 << 24;
    int nElem = 1 << 29;
    printf("Vector size %d\n", nElem);

    // allocate host memory
    float *h_A, *h_B, *h_res_cpu, *h_res_gpu;
    size_t nBytes = nElem * sizeof(float);
    h_A       = (float*) malloc(nBytes);
    h_B       = (float*) malloc(nBytes);
    h_res_cpu = (float*) malloc(nBytes);
    h_res_gpu = (float*) malloc(nBytes);

    // initialize arrays
    double ti, dt;
    ti = cpuSeconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    dt = cpuSeconds() - ti;
    printf("Time elapsed in initialData: %f[s]\n", dt);

    // allocate GPU memory
    float *d_A, *d_B, *d_res_gpu;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_res_gpu, nBytes);

    // transfer data from host to device
    ti = cpuSeconds();
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
    dt = cpuSeconds() - ti;
    printf("Time elapsed in cudaMemcpy: %f[s]\n", dt);

    // sum array on host
    ti = cpuSeconds();
    sumArrayOnHost(h_A, h_B, h_res_cpu, nElem);
    dt = cpuSeconds() - ti;
    printf("Time elapsed in sumArrayOnHost: %f[s]\n", dt);

    // define grid layout
    int nThreads = 1024;
    dim3 block (nThreads);
    dim3 grid (nElem/block.x);

    // sum array on GPU
    ti = cpuSeconds();
    sumArrayOnGPU<<<grid,block>>>(d_A, d_B, d_res_gpu, nElem);
    cudaDeviceSynchronize();
    dt = cpuSeconds() - ti;
    printf("Time elapsed in sumArrayOnGPU<<<%d,%d>>>: %f[s]\n", grid.x, block.x, dt);

    // transfer data from device to host
    cudaMemcpy(h_res_gpu, d_res_gpu, nBytes, cudaMemcpyDeviceToHost);

    // check result
    checkResult(h_res_cpu, h_res_gpu, nElem);
    
    // free host memory
    free(h_A);
    free(h_B);
    free(h_res_cpu);
    free(h_res_gpu);

    // free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_res_gpu);

    cudaDeviceReset();
    return(0);
}
