#include <stdio.h>

void sumArrayOnHost(float *A, float *B, float *C, const int N) {
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C, const int N) {
    int i = threadIdx.x;
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
    // define array size
    int nElem = 1024;

    // allocate host memory
    float *h_A, *h_B, *h_res_cpu, *h_res_gpu;
    size_t nBytes = nElem * sizeof(float);
    h_A = (float*) malloc(nBytes);
    h_B = (float*) malloc(nBytes);
    h_res_cpu = (float*) malloc(nBytes);
    h_res_gpu = (float*) malloc(nBytes);

    // initialize arrays
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // allocate GPU memory
    float *d_A, *d_B, *d_res_gpu;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_res_gpu, nBytes);

    // transfer data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // sum array on host
    sumArrayOnHost(h_A, h_B, h_res_cpu, nElem);

    // sum array on GPU
    sumArrayOnGPU<<<1,1024>>>(d_A, d_B, d_res_gpu, nElem);

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
