#include <stdio.h>

__global__ void helloFromGPU(void) {
    printf("Hello World from GPU! thread %i\n", threadIdx.x);
}

int main(void) {
    // hello from cpu
    printf("Hello World from CPU!\n");

    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    //cudaDeviceSynchronize(); // what is the diference between reset and synchronize?
    return 0;
}
