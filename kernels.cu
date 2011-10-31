/*
 * kernels.cu
 *
 *  Created on: 17.10.2011
 *      Author: www.VarnaSoftware.com,  SYavorovsky@varnasoftware.com
 */

#include "kernels.cuh"


__device__  double bbOverlap(int *bb1, int *bb2) {
    // Check whether the bounding-boxes overlap at all
    if (bb1[0] > bb2[0] + bb2[2]) {
        return 0;
    }
    else if (bb1[1] > bb2[1] + bb2[3]) {
        return 0;
    }
    else if (bb2[0] > bb1[0] + bb1[2]) {
        return 0;
    }
    else if (bb2[1] > bb1[1] + bb1[3]) {
        return 0;
    }


    // If we got this far, the bounding-boxes overlap
    double overlapWidth = min(bb1[0] + bb1[2], bb2[0] + bb2[2]) - max(bb1[0], bb2[0]);
    double overlapHeight = min(bb1[1] + bb1[3], bb2[1] + bb2[3]) - max(bb1[1], bb2[1]);
    double overlapArea = overlapWidth * overlapHeight;
    double bb1Area = bb1[2] * bb1[3];
    double bb2Area = bb2[2] * bb2[3];

    return overlapArea / (bb1Area + bb2Area - overlapArea);
}


__device__ int sumRect(int *IIdata, int width, int height, int x, int y, int w, int h)
{
	int retval=0;

    if (x >= 0 && w > 0 && x + w < width && y >= 0 && h > 0 && y + h < height) {

    	int dx=IIdata[x+y*(width)];
    	int dy=IIdata[(x+w)+(y+h)*(width)];
    	int dw=IIdata[(x+w)+y*(width)];
    	int dh=IIdata[x+(y+h)*(width)];

    	retval=dx+dy-dw-dh;

         // IIdata[x][y] + IIdata[x + w][y + h] - IIdata[x + w][y] - IIdata[x][y + h];

     } else {
    	 printf("ERROR: SUM RECT OUT OF BOUNDS! (%d, %d, %d, %d)\n", x, y, w, h);
     }
        return retval;
}


__device__ int getWarpXY(int *IIdata, int width, int height, int wx,int wy,int ww,int wh,int*bb,float *m)
{
    int ox = -(int)((float)(ww) * 0.5);
    int oy = -(int)((float)(wh) * 0.5);
    int cx = (int)(bb[0] - ox);
    int cy = (int)(bb[1] - oy);

	int x=ox+wx;
	int y=oy+wy;
    int xp = (int)(m[0] * (float)x + m[1] * (float)y + cx);
    int yp = (int)(m[2] * (float)x + m[3] * (float)y + cy);

    // Limit pixels to those in the given bounding-box
    xp = max(min(xp, bb[0] + ww), bb[0]);
    yp = max(min(yp, bb[1] + wh), bb[1]);

    return IIdata[xp + yp*width];
}

__device__ int sumRectWarp(int *IIdata, int width, int height, int x, int y, int w, int h, int *bb, float *m)
{
	int retval=0;

    if (x >= 0 && w > 0 && x + w < width && y >= 0 && h > 0 && y + h < height) {

    	int dx=getWarpXY(IIdata,width, height, x,y,w,h,bb,m);
    	int dy=getWarpXY(IIdata,width, height, x+w,y+h,w,h,bb,m);
    	int dw=getWarpXY(IIdata,width, height, x+w,y,w,h,bb,m);
    	int dh=getWarpXY(IIdata,width, height, x,y+h,w,h,bb,m);

    	retval=dx+dy-dw-dh;

         // IIdata[x][y] + IIdata[x + w][y + h] - IIdata[x + w][y] - IIdata[x][y + h];

     } else {
    	 printf("ERROR: SUM RECT OUT OF WARP BOUNDS! (%d, %d, %d, %d)\n", x, y, w, h);
     }
        return retval;
}


/*
 * in_ferns  r/w лес деревьев
 * IIdata Integral Image
 * w,h size of Integral Image
 * patch[]={x,y,w,h,class} - Patch parametrs
 */

__global__ void train_kernel(struct one_fern *in_ferns,int *IIdata, int width, int height, int p_idx2, int* patch, int* tbb )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y; //y - this a p_idx = 0..6*30*30

	int leafIdx=0;
	if(idx>=TOTAL_FERNS)return;
	//найдем индекс ветки
	if(idy>=p_idx2)return;
	int p_idx=idy;

	int patchX=patch[p_idx*5+0];
	int patchY=patch[p_idx*5+1];
	int patchW=patch[p_idx*5+2];
	int patchH=patch[p_idx*5+3];
	if(!(patchX+patchY+patchW+patchH))
	{
		printf("error in %d",p_idx);
		return;
	}
    // Clamp x and y values between 0 and width and height respectively
    patchX = max(min(patchX, width - 2), 0);
    patchY = max(min(patchY, height - 2), 0);

    // Limit width and height values to (width - patchX) and (height - patchY)
    // respectively
    patchW = min(patchW, width - patchX);
    patchH = min(patchH, height - patchY);

    // Apply all tests to find the leaf index this patch falls into
    int leaf = 0;
    int test=0;

    /*for (int i = 0; i < nodeCount; i++) {
        leaf = leaf | (nodes[i]->test(image, patchX, patchY, patchW, patchH) << i * (int)POWER);
    }*/
    int i;
	for(i=0;i<TOTAL_NODES;i++)
	{

	    int x = (int)(in_ferns[idx].nodes[i][0] * (float)patchW) + patchX;
	    int y = (int)(in_ferns[idx].nodes[i][1] * (float)patchH) + patchY;
	    int w = (int)(in_ferns[idx].nodes[i][2] * (float)patchW * 0.5f);
	    int h = (int)(in_ferns[idx].nodes[i][3] * (float)patchH * 0.5f);

	    // Compare the various halfs of the feature on the patch
	    int left,right,top,bottom;

	    left = sumRect(IIdata, width, height, x, y, w, h * 2);
	    right = sumRect(IIdata, width, height, x + w, y, w, h * 2);
	    top = sumRect(IIdata, width, height, x, y, w * 2, h);
	    bottom = sumRect(IIdata, width, height, x, y + h, w * 2, h);

	    if (left > right) {
	        if (top > bottom) {
	        	test=0;
	        }
	        else {
	        	test=1;
	        }
	    }
	    else {
	        if (top > bottom) {
	        	test=2;
	        }
	        else {
	        	test=3;
	        }
	    }


		leaf = leaf | (test << (i * (int)2));

	}

	int is_positive=patch[p_idx*5+4];

	if(is_positive<0) return;

	if(p_idx2>1)
	{
		if (is_positive==1 && bbOverlap(tbb, &patch[p_idx*5]) < MIN_LEARNING_OVERLAP)
		{
			is_positive=0;
			//printf(" %d; %d \n",blockIdx.x,blockIdx.y);
		}
		//double fff
		//printf(" %f; ",bbOverlap(tbb, &patch[p_idx*5]));
		//printf(" %d; %d \n",blockIdx.x,blockIdx.y);

	}

	//if(tbb!=NULL && is_positive)printf(" %d; ",is_positive);

    if (is_positive == 0) {
    	atomicAdd(&in_ferns[idx].n[leaf],1);
    }

	if (is_positive == 1)
	{
		atomicAdd(&in_ferns[idx].p[leaf],1);
		//atomicExch(&in_ferns[idx].n[leaf],0);
	}
   // int p=in_ferns[idx].p[leaf];
   // int n=in_ferns[idx].n[leaf];
    // Compute the posterior likelihood of a positive class for this leaf
  /*  if (p > 0) {

    	in_ferns[idx].posteriors[leaf] = (float)p / (float)(p + n);
    }*/

}


__global__ void patcher_kernel(int count, int incX, int minX, int incY, int minY,int currentWidth, int currentHeight, int* patch, int s)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if(idx>=count || idy>=count)return;

	int p_len=idy+idx*count+s*count*count; //int p_len=idy+idx*county+s*countx*county;
	int x=idx*incX+minX;
	int y=idy*incY+minY;

	patch[(p_len)*5]=x;
	patch[(p_len)*5+1]=y;
	patch[(p_len)*5+2]=currentWidth;
	patch[(p_len)*5+3]=currentHeight;
	__syncthreads();
}


__global__ void detect_conf_kernel(int* patch, int* tbb, int p_len, float *ret)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx>=p_len)return;
    double myanswer=ret[idx]/(float)TOTAL_FERNS;
    int train_b=0;
   // printf("About: %f,%f,%f,%f\n ",bb[0],bb[1],bb[2],bb[3]);

    if (tbb[5] != -1 && bbOverlap(&patch[idx*5], tbb) > MIN_LEARNING_OVERLAP) {
    	train_b=1;
    } else {
    	train_b=0;
    }

    // If positive, or negative and overlapping with the tracked
    // bounding-box, add this bounding-box to our return list
    if (myanswer > 0.5f || train_b==1) {
    	train_b=1;
    } else {
    	train_b=-1; //we not wont to use it for train

    }

//   if(train_b==1)
//     printf("About: %f: %d\n ",myanswer, blockIdx.x);

    atomicExch(&patch[idx*5+4],train_b);
}

#define BLOCK_SIZE 128

__shared__ float sdata[BLOCK_SIZE];
__shared__ float sidxdata[BLOCK_SIZE];
__global__ void reduce3(float*g_idata, float*g_odata,unsigned int *g_iidx,unsigned int *g_oidx)
{
//	__shared__ float sdata[BLOCK_SIZE];
//	__shared__ float sidxdata[BLOCK_SIZE];
// each thread loadsone element from global to shared mem
unsigned int tid = threadIdx.x;
unsigned int i= blockIdx.x*blockDim.x+ threadIdx.x;
sdata[tid]=-1;
if(i>=5400)return;
if(tid>=BLOCK_SIZE)return;
if(g_iidx[i]==0)g_iidx[i]=i;
sdata[tid] = g_idata[i];
sidxdata[tid] = g_iidx[i];
__syncthreads();
//printf("%d %f %f\n",blockIdx.x,sdata[tid],g_idata[i]);

// do reduction in shared mem
for(unsigned int s=blockDim.x/2; s>0; s>>=1) {
if (tid < s) {
	if(sdata[tid] < sdata[tid + s])
		{
		sdata[tid] = sdata[tid + s];
		sidxdata[tid] = sidxdata[tid + s];
		}

}
__syncthreads();
}

// write result for this block to global mem
if(tid == 0) {
	g_odata[blockIdx.x] = sdata[0];
	g_oidx[blockIdx.x] = sidxdata[0];
//if(sdata[0]>0)
//	printf("%d %f\n",blockIdx.x,sdata[0]);
}
}


__global__ void classify_kernel(struct one_fern *in_ferns,int *IIdata, int width, int height, int p_idx2, int* patch, float *ret )
{


	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y; //y - this a p_idx = 0..6*30*30

	int leafIdx=0;

	if(idx>=TOTAL_FERNS)return;

	if(idy>=p_idx2)return;
	int p_idx=idy;
	//if(p_idx2>1)printf("ok %d %d; ",idy,p_idx2);
	int patchX=patch[p_idx*5+0];
	int patchY=patch[p_idx*5+1];
	int patchW=patch[p_idx*5+2];
	int patchH=patch[p_idx*5+3];
	if(!(patchX+patchY+patchW+patchH))
	{
		printf("error in %d",p_idx);
		return;
	}
    // Clamp x and y values between 0 and width and height respectively
    patchX = max(min(patchX, width - 2), 0);
    patchY = max(min(patchY, height - 2), 0);

    // Limit width and height values to (width - patchX) and (height - patchY)
    // respectively
    patchW = min(patchW, width - patchX);
    patchH = min(patchH, height - patchY);

    // Apply all tests to find the leaf index this patch falls into
    int leaf = 0;
    int test=0;

    /*for (int i = 0; i < nodeCount; i++) {
        leaf = leaf | (nodes[i]->test(image, patchX, patchY, patchW, patchH) << i * (int)POWER);
    }*/
    int i;
	for(i=0;i<TOTAL_NODES;i++)
	{

	    int x = (int)(in_ferns[idx].nodes[i][0] * (float)patchW) + patchX;
	    int y = (int)(in_ferns[idx].nodes[i][1] * (float)patchH) + patchY;
	    int w = (int)(in_ferns[idx].nodes[i][2] * (float)patchW * 0.5f);
	    int h = (int)(in_ferns[idx].nodes[i][3] * (float)patchH * 0.5f);

	    // Compare the various halfs of the feature on the patch
	    int left,right,top,bottom;

	    left = sumRect(IIdata, width, height, x, y, w, h * 2);
	    right = sumRect(IIdata, width, height, x + w, y, w, h * 2);
	    top = sumRect(IIdata, width, height, x, y, w * 2, h);
	    bottom = sumRect(IIdata, width, height, x, y + h, w * 2, h);

	    if (left > right) {
	        if (top > bottom) {
	        	test=0;
	        }
	        else {
	        	test=1;
	        }
	    }
	    else {
	        if (top > bottom) {
	        	test=2;
	        }
	        else {
	        	test=3;
	        }
	    }


		leaf = leaf | (test << (i * (int)2));

	}
	//printf("Hello thread %d, f=%d\n", threadIdx.x, leaf);
	//wait for all treads
	//ret[idx+p_idx*TOTAL_FERNS]=in_ferns[idx].posteriors[leaf];
    int p=in_ferns[idx].p[leaf];
    int n=in_ferns[idx].n[leaf];
    if(p || n)
	atomicAdd(&ret[p_idx],(float)p / (float)(p + n));
	//atomicAdd(&ret[p_idx],in_ferns[idx].posteriors[leaf]);

	__syncthreads();
	//
	/*if(idx==0)
	{
		*ret=c_retval[0];//(float)TOTAL_FERNS;
	}*/


}

__global__ void train_kernel_warp(struct one_fern *in_ferns,int *IIdata, int width, int height, int* patch, int* bb, float *m )
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int leafIdx=0;
	if(idx>=TOTAL_FERNS)return;
	//найдем индекс ветки
	//int p_idx=0;

	int patchX=patch[0];
	int patchY=patch[1];
	int patchW=patch[2];
	int patchH=patch[3];

    // Clamp x and y values between 0 and width and height respectively
    // Clamp x and y values between 0 and width and height respectively
    patchX = max(min(patchX, width - 2), 0);
    patchY = max(min(patchY, height - 2), 0);

    // Limit width and height values to (width - patchX) and (height - patchY)
    // respectively
    patchW = min(patchW, width - patchX);
    patchH = min(patchH, height - patchY);
    //__syncthreads ();
    // Apply all tests to find the leaf index this patch falls into
    int leaf = 0;
    int test=0;

    /*for (int i = 0; i < nodeCount; i++) {
        leaf = leaf | (nodes[i]->test(image, patchX, patchY, patchW, patchH) << i * (int)POWER);
    }*/
    int i;
	for(i=0;i<TOTAL_NODES;i++)
	{

	    int x = (int)(in_ferns[idx].nodes[i][0] * (float)patchW) + patchX;
	    int y = (int)(in_ferns[idx].nodes[i][1] * (float)patchH) + patchY;
	    int w = (int)(in_ferns[idx].nodes[i][2] * (float)patchW * 0.5f);
	    int h = (int)(in_ferns[idx].nodes[i][3] * (float)patchH * 0.5f);


	    // Compare the various halfs of the feature on the patch
	    int left,right,top,bottom;
	    left = sumRectWarp(IIdata, width, height, x, y, w, h * 2,bb,m);
	    right = sumRectWarp(IIdata, width, height, x + w, y, w, h * 2,bb,m);
	    top = sumRectWarp(IIdata, width, height, x, y, w * 2, h,bb,m);
	    bottom = sumRectWarp(IIdata, width, height, x, y + h, w * 2, h,bb,m);


	    if (left > right) {
	        if (top > bottom) {
	        	test=0;
	        }
	        else {
	        	test=1;
	        }
	    }
	    else {
	        if (top > bottom) {
	        	test=2;
	        }
	        else {
	        	test=3;
	        }
	    }


		leaf = leaf | (test << (i * (int)2));

	}

/*
	int is_positive=patch[p_idx*5+4];
	if(tbb!=NULL)
	{
		if (bbOverlap(tbb, &patch[p_idx*5]) >= MIN_LEARNING_OVERLAP)
			is_positive=1;
	}


    if (is_positive == 0) {
    	atomicAdd(&in_ferns[idx].n[leaf],1);
    }
    else {
    	atomicAdd(&in_ferns[idx].p[leaf],1);
    }
*/

    if (patch[4] == 0) {
    	atomicAdd(&in_ferns[idx].n[leaf],1);
    }
    else {
    	atomicAdd(&in_ferns[idx].p[leaf],1);
    }
/*  int p=in_ferns[idx].p[leaf];
    int n=in_ferns[idx].n[leaf];
    // Compute the posterior likelihood of a positive class for this leaf
    if (p > 0) {

    	in_ferns[idx].posteriors[leaf] = (float)p / (float)(p + n);
    }
*/
}

