// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Green, and Blue is in it.
//The 'A' stands for Alpha and is used for transparency; it will be
//ignored in this homework.

//Each channel Red, Blue, Green, and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

//#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>
#include "device_launch_parameters.h"

// Global functions are also called "kernels". 
// It's the functions that you may call from the 
// host side using CUDA kernel call semantics
__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_x < numRows && idx_y < numCols) {
        int idx_1d = idx_x * numCols + idx_y;
        const uchar4 rgba = rgbaImage[idx_1d];
        float fpix = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
        greyImage[idx_1d] = fpix;
    }
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{

    const int thread = 16;
    const dim3 blockSize(thread, thread, 1);
    const dim3 gridSize(ceil(numRows /(float)thread), ceil(numCols /(float)thread), 1);
    rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

// Notes:
// https://stackoverflow.com/questions/4391162/cuda-determining-threads-per-block-blocks-per-grid
// https://stackoverflow.com/questions/12373940/difference-between-global-and-device-functions