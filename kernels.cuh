/*
 * kernels.h
 *
 *  Created on: 17.10.2011
 *      Author:  www.VarnaSoftware.com,  SYavorovsky@varnasoftware.com
 */

#ifndef KERNELS_H_
#define KERNELS_H_
#include "nvClassifier.h"
__global__ void train_kernel(struct one_fern *in_ferns,int *IIdata, int width, int height, int p_idx2, int* patch, int* tbb );
__global__ void classify_kernel(struct one_fern *in_ferns,int *IIdata, int width, int height, int p_idx, int* patch, float *ret );
__global__ void train_kernel_warp(struct one_fern *in_ferns,int *IIdata, int width, int height, int* patch, int* bb, float *m );
__global__ void patcher_kernel(int count, int incX, int minX, int incY, int minY,int currentWidth, int currentHeight, int* patch, int s);
__global__ void reduce3(float*g_idata, float*g_odata,unsigned int *g_iidx,unsigned int *g_oidx);
__global__ void detect_conf_kernel(int* patch, int* tbb, int p_len, float *ret);

#endif /* KERNELS_H_ */
