/*
 * nvClassifier.cpp
 *
 *  Created on: 17.10.2011
 *      Author:  www.VarnaSoftware.com,  SYavorovsky@varnasoftware.com
 */

#include "nvClassifier.h"
#include "kernels.cuh"
#include "mt19937ar.h"

namespace std {

nvClassifier::nvClassifier(float minScale, float maxScale) {
	// TODO Auto-generated constructor stub
	struct one_fern* forest_ferns=(struct one_fern *)malloc(sizeof(struct one_fern)*TOTAL_FERNS);
	memset(forest_ferns,0,sizeof(struct one_fern)*TOTAL_FERNS);
 //Init FEARNS
	 for (int f = 0; f < TOTAL_FERNS; f++)
	 {
		for (int i = 0; i < TOTAL_NODES; i++)
		{
			forest_ferns[f].nodes[i][2]=//(maxScale - minScale) * ((float)rand() / (float)RAND_MAX) + minScale;
					(maxScale - minScale) * (float)genrand_int31()/(float)RAND_MAX + minScale;
			forest_ferns[f].nodes[i][3]= //(maxScale - minScale) * ((float)rand() / (float)RAND_MAX) + minScale;
					 (maxScale - minScale) * (float)genrand_int31()/(float)RAND_MAX + minScale;
			forest_ferns[f].nodes[i][0]= //(1.0f - forest_ferns[f].nodes[i][2]) * ((float)rand() / (float)RAND_MAX);
					(1.0f - forest_ferns[f].nodes[i][2]) * (float)genrand_int31()/(float)RAND_MAX;
			forest_ferns[f].nodes[i][1]= //(1.0f - forest_ferns[f].nodes[i][3]) * ((float)rand() / (float)RAND_MAX);
					(1.0f - forest_ferns[f].nodes[i][3]) * (float)genrand_int31()/(float)RAND_MAX;
		}

	 }


	//Will be save our array on the card (about 4096*TOTAL_FERNS)
	cudaMalloc((void**) &c_forest_ferns, sizeof(struct one_fern)*TOTAL_FERNS);
	cudaMemcpy(c_forest_ferns, forest_ferns, sizeof(struct one_fern)*TOTAL_FERNS, cudaMemcpyHostToDevice);
	free(forest_ferns);

	c_IMGdata=NULL;
	c_IIdata=NULL;
	planar_data=NULL;

}

nvClassifier::~nvClassifier() {
	// TODO Auto-generated destructor stub
	if(c_forest_ferns!=NULL)
		cudaFree(c_forest_ferns);
	if(c_IIdata)
		cudaFree(c_IIdata);
	if(c_IMGdata)
		cudaFree(c_IMGdata);
	if(planar_data)
		free(planar_data);
}


float nvClassifier::classify(int* patch)
{
	// Host variables
	int i,j;
	unsigned int num_threads;
	// Initialize CPU variables and allocate required memory
	num_threads = (unsigned int) TOTAL_FERNS;
	if(patch[2]<=0 || patch[2]<=0) return 0.0f;
	// Allocate GPU (device) memory and variables
	float h_answer[TOTAL_FERNS];
	float* d_answer;
	int *d_patch;
	cudaMalloc((void**) &d_answer, sizeof(float));
	cudaMalloc((void**) &d_patch, sizeof(int)*5);

	// Copy vectors from host memory to device memory

	cudaMemcpy(d_patch, patch, sizeof(int)*5, cudaMemcpyHostToDevice);
	cudaMemset(d_answer, 0,  sizeof(float));


	// Setup kernel execution parameters
	//dim3 grid(1,1,1);
	//dim3 threads(num_threads,1,1);

	int blocksize=16;
	dim3 dimBlock (1,1);
	dim3 dimGrid( ceil( (float)num_threads / (float)dimBlock.x),1 );
//	if(patch[2]<=0)
//	 printf("w:%d; ",patch[2]);
	// Execute the kernel on the GPU
	classify_kernel<<< dimGrid, dimBlock >>>(c_forest_ferns,c_IIdata,width,height,1,d_patch,d_answer);
	cudaDeviceSynchronize();

	// Copy result from GPU to CPU
	cudaMemcpy(&h_answer, d_answer, sizeof(float), cudaMemcpyDeviceToHost);

/*	 for(i=1;i<TOTAL_FERNS;i++)
	 {
		 h_answer[0]+=h_answer[i];
	 }*/
	//printf("about: %f",h_answer[0]);
	//////////////////////////////////////////
	// All done - clean up and exit
	//////////////////////////////////////////
	// Free up CPU memory

	// Free up GPU memory
	cudaFree(d_answer);
	cudaFree(d_patch);
	return h_answer[0]/(float)TOTAL_FERNS;
}


void nvClassifier::train(int* patch)
{

	// Host variables
	int i,j;
	unsigned int num_threads;
	struct one_fern *frn;
	// Initialize CPU variables and allocate required memory
	num_threads = (unsigned int) TOTAL_FERNS;

	// Allocate GPU (device) memory and variables
	int *d_patch;

	cudaMalloc((void**) &d_patch, sizeof(int)*5);
	cudaMemcpy(d_patch, patch, sizeof(int)*5, cudaMemcpyHostToDevice);

	// Copy vectors from host memory to device memory

	// Setup kernel execution parameters
	int blocksize=16;
	dim3 dimBlock (blocksize);
	dim3 dimGrid( ceil( (float)num_threads / (float)blocksize) );

    int *d_tbb;
    cudaMalloc((void**) &d_tbb, sizeof(int)*6);

	// Execute the kernel on the GPU
	train_kernel<<< dimGrid, dimBlock >>>(c_forest_ferns,c_IIdata,width,height,1,d_patch,d_tbb);
	cudaDeviceSynchronize();
	// Copy result from GPU to CPU

	//////////////////////////////////////////
	// All done - clean up and exit
	//////////////////////////////////////////
	// Free up CPU memory

	// Free up GPU memory

	cudaFree(d_patch);
	cudaFree(d_tbb);


}

void nvClassifier::detect(double *tbb, double confidence)
{
    // Set the width and height that are used as 1 * scale.
    // If tbb is NULL, we are not tracking and use the first-frame
    // bounding-box size, otherwise we use the tracked bounding-box size

	// Host variables
	int i,j;
	unsigned int num_threads;
	// Initialize CPU variables and allocate required memory
	num_threads = (unsigned int) TOTAL_FERNS;

    float baseWidth, baseHeight;

    //int *patch=NULL;

        baseWidth = (float)tbb[2];
        baseHeight = (float)tbb[3];


    if (baseWidth < 40 || baseHeight < 40) {
        return;
    }

    // Using the sliding-window approach, find positive matches to our object
    // Vector of positive patch matches' bounding-boxes


    // Minimum and maximum scales for the bounding-box, the number of scale
    // iterations to make, and the amount to increment scale by each iteration
    float minScale = 0.5f;
    float maxScale = 1.5f;
    int iterationsScale = 6;
    float scaleInc = (maxScale - minScale) / (iterationsScale - 1);


	// Allocate GPU (device) memory and variables
    int *d_patch;
    int p_len=iterationsScale*30*30;
    cudaMalloc((void**) &d_patch, sizeof(int)*5*p_len);
    cudaMemset(d_patch, 0,  sizeof(int)*5*p_len);

    int blocksize=16;
	dim3 dimBlock2 (1,1);
	dim3 dimGrid2( ceil( (float)32 / (float)dimBlock2.x), ceil( (float)32 / (float)dimBlock2.y)  );

    // Loop through a range of bounding-box scales
    //make massive of scales
    for (int s = 0; s < iterationsScale; s ++) {
    	//i=0;i<6;i++
    	float scale=s*scaleInc+minScale;
        int minX = 0;
        int currentWidth = (int)(scale * (float)baseWidth);
        if(currentWidth>=width)currentWidth=width-1;
        int maxX = width - currentWidth;
        float iterationsX = 30.0;
        int incX = (int)floor((float)(maxX - minX) / (iterationsX - 1.0));
        if(incX<=0)incX=1;

        // Same for y
        int minY = 0;
        int currentHeight = (int)(scale * (float)baseHeight);
        if(currentHeight>=height)currentHeight=height-1;
        int maxY = height - currentHeight;
        float iterationsY = 30.0;
        int incY = (int)floor((float)(maxY - minY) / (iterationsY - 1.0));
        if(incY<=0)incY=1;
       // printf("w:%d; ",currentWidth);
        patcher_kernel<<< dimGrid2, dimBlock2 >>>(30,incX,minX,incY,minY,currentWidth ,currentHeight,d_patch,s);
       // printf("h:%d; ",currentHeight);
    }
    cudaDeviceSynchronize();
	//float* h_answer;
	//h_answer=(float*)malloc(sizeof(float)*p_len);

	// Allocate GPU (device) memory and variables
	float* d_answer;


	cudaMalloc((void**) &d_answer, sizeof(float)*p_len);

	// Copy vectors from host memory to device memory
	//cudaMemcpy(d_patch, patch, sizeof(int)*4*p_len, cudaMemcpyHostToDevice);
	cudaMemset(d_answer, 0,  sizeof(float)*p_len);

	// Setup kernel execution parameters
	dim3 dimBlock3 (8,1);
	dim3 dimGrid3 ( ceil( (float)num_threads / (float)dimBlock3.x),ceil( (float)p_len / (float)dimBlock3.y) );


	//for(int f=0;f<p_len;f++) //нужно уменьшить количество вызовов распределением
	//{
	// Execute the kernel on the GPU
	classify_kernel<<< dimGrid3, dimBlock3 >>>(c_forest_ferns,c_IIdata,width,height,p_len,d_patch,d_answer);
	//}
	cudaDeviceSynchronize();

	dim3 dimBlock5 (128);
	dim3 dimGrid5 ( ceil( (float)p_len / (float)dimBlock5.x) );
	int tbb_tmp[6];
	for(int p=0;p<6;p++)
		tbb_tmp[p]=(int)tbb[p];
    int *d_tbb;
    cudaMalloc((void**) &d_tbb, sizeof(int)*6);
    cudaMemcpy(d_tbb,tbb_tmp,sizeof(int)*6,cudaMemcpyHostToDevice);

	detect_conf_kernel<<< dimGrid5, dimBlock5 >>>(d_patch, d_tbb, p_len, d_answer);
	cudaDeviceSynchronize();
	// Copy result from GPU to CPU
	//cudaMemcpy(h_answer, d_answer, sizeof(float)*p_len, cudaMemcpyDeviceToHost);


	//int *patch=(int*)malloc(sizeof(int)*p_len*5);
	//cudaMemcpy(patch, d_patch, sizeof(int)*5*p_len, cudaMemcpyDeviceToHost);

	/*
	 * Get MAX confidence from answerd area and it index.
	 */

    #define BLOCK_SIZE 128
    // allocate device memory and data
    float* d_idata[2] = {NULL,NULL};
    unsigned int* d_idx[2] =  {NULL,NULL};

    int n=p_len;//  ceil( (float)p_len / (float)BLOCK_SIZE);

    cudaMalloc((void**) &d_idata[0], sizeof(float)*p_len) ;
    cudaMalloc((void**) &d_idata[1], sizeof(float)*p_len) ;
    cudaMemset(d_idata[1],0,sizeof(float)*p_len);

   cudaMalloc((void**) &d_idx[0], sizeof(unsigned int)*p_len) ;
   cudaMalloc((void**) &d_idx[1], sizeof(unsigned int)*p_len) ;

    cudaMemset(d_idx[0],0,sizeof(unsigned int)*p_len);
    cudaMemcpy(d_idata[0],d_answer,sizeof(float)*p_len,cudaMemcpyDeviceToDevice);
    int k=2;

    for ( i = 0; k >= 1; k--,i^=1 ){
    	    dim3 dimBlock (BLOCK_SIZE, 1, 1);
            dim3 dimGrid (ceil( (float)n / (float)dimBlock.x), 1, 1);
            reduce3<<<dimGrid,dimBlock>>>(d_idata[i],d_idata[i^1],d_idx[i],d_idx[i^1]);
            n=22;
            //n=ceil( (float)n / (float)dimBlock.x);
    }
    cudaDeviceSynchronize();
    float MaxConf;
    int dbbMaxConf_idx;

    cudaMemcpy(&MaxConf, d_idata[0], sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&dbbMaxConf_idx, d_idx[0], sizeof(int), cudaMemcpyDeviceToHost);

    double dbbMaxConf=MaxConf/(float)TOTAL_FERNS;
    //printf("MAX_VALUE_P is %d: %f\n", dbbMaxConf_idx, dbbMaxConf);
#define MIN_REINIT_CONF 0.7f
#define MIN_LEARNING_CONF 0.8f

	if (dbbMaxConf > tbb[4] && dbbMaxConf > MIN_REINIT_CONF) {
//		delete [] tbb;
//		tbb = new double[5];
		//double *dbb = dbbs->at(dbbMaxConfIndex);

		int patchtest[5];
		cudaMemcpy(&patchtest, d_patch+dbbMaxConf_idx*5, sizeof(int)*5, cudaMemcpyDeviceToHost);
		//printf("PATCH is %d: %d\n", patchtest[0], patch[dbbMaxConf_idx*5]);
        tbb[0] = (double)patchtest[0];
	    tbb[1] = (double)patchtest[1];
	    tbb[2] = (double)patchtest[2];
	    tbb[3] = (double)patchtest[3];
	    tbb[4] = dbbMaxConf;

	}
	else
		if (tbb[4] > dbbMaxConf && confidence > MIN_LEARNING_CONF)
		{
			// Train the classifier on positive (overlapping with tracked
			// patch) and negative (classed as positive but non-overlapping)
			// patches
        	num_threads = (unsigned int) TOTAL_FERNS;
        	// Setup kernel execution parameters
        	dim3 dimBlock8 (8,64);
        	dim3 dimGrid8( ceil( (float)num_threads / (float)dimBlock8.x),ceil( (float)p_len / (float)dimBlock8.y) );
        	//int tbb_tmp[5];
        	/*for(int p=0;p<4;p++)
        		tbb_tmp[p]=(int)tbb[p];
            cudaMemcpy(d_tbb,tbb_tmp,sizeof(int)*4,cudaMemcpyHostToDevice);*/

        	// Execute the kernel on the GPU
        	train_kernel<<< dimGrid8, dimBlock8 >>>(c_forest_ferns,c_IIdata,width,height,p_len,d_patch,d_tbb);
        	cudaDeviceSynchronize();

		}


/*
    test=0;
    test_idx=0;
	for(int f=0;f<p_len;f++)
	{

	 float myanswer=h_answer[0+f];
	 if(test<myanswer){ test=myanswer; test_idx=f;}

	}
	printf("MAX_VALUE is %d: %f\n", test_idx, test);
*/



    //////////////////////////////////////////
	// All done - clean up and exit
	//////////////////////////////////////////
	// Free up CPU memory
	//free(h_answer);
	//free(patch);


	// Free up GPU memory
	cudaFree(d_tbb);
	cudaFree(d_answer);
	cudaFree(d_patch);
	cudaFree(d_idata[0]);
	cudaFree(d_idata[1]);
	cudaFree(d_idx[0]);
	cudaFree(d_idx[1]);



}

/*  Trains the classifier on negative training patches, i.e. patches from the
    first frame that don't overlap the bounding-box patch.
    frame: frame to take warps from
    tbb: first-frame bounding-box [x, y, width, height] */
void nvClassifier::trainNegative(double *tbb)
{
    // Minimum and maximum scales for the bounding-box, the number of scale
    // iterations to make, and the amount to increment scale by each iteration
    float minScale = 0.5f;
    float maxScale = 1.5f;
    int iterationsScale = 5;
    float scaleInc = (maxScale - minScale) / (iterationsScale - 1);

    int *d_patch;
    int p_len=iterationsScale*30*30;
    cudaMalloc((void**) &d_patch, sizeof(int)*5*p_len);
    cudaMemset(d_patch, 0,  sizeof(int)*5*p_len);

    int blocksize=16;
	dim3 dimBlock2 (blocksize,blocksize);
	dim3 dimGrid2( ceil( (float)32 / (float)blocksize), ceil( (float)32 / (float)blocksize)  );



    // Loop through a range of bounding-box scales
    //make massive of scales
    for (int s = 0; s < iterationsScale; s ++) {
    	//i=0;i<6;i++
    	float scale=s*scaleInc+minScale;
        int minX = 0;
        int currentWidth = (int)(scale * (float)tbb[2]);
        if(currentWidth>=tbb[2])currentWidth=tbb[2]-1;
        int maxX = tbb[2] - currentWidth;
        float iterationsX = 30.0;
        int incX = (int)floor((float)(maxX - minX) / (iterationsX - 1.0));
        if(incX<=0)incX=1;

        // Same for y
        int minY = 0;
        int currentHeight = (int)(scale * (float)tbb[3]);
        if(currentHeight>=tbb[3])currentHeight=tbb[3]-1;
        int maxY = tbb[3] - currentHeight;
        float iterationsY = 30.0;
        int incY = (int)floor((float)(maxY - minY) / (iterationsY - 1.0));
        if(incY<=0)incY=1;
       // printf("w:%d; ",currentWidth);
        patcher_kernel<<< dimGrid2, dimBlock2 >>>(30,incX,minX,incY,minY,currentWidth ,currentHeight,d_patch,s);
       // printf("h:%d; ",currentHeight);
    }
    cudaDeviceSynchronize();



    int *bb_patch = new int[6];
    for(int i=0;i<4;i++)
    	bb_patch[i]=tbb[i];
    bb_patch[4]=0;
    bb_patch[5]=0;

    int *d_tbb;
    cudaMalloc((void**) &d_tbb, sizeof(int)*6);
    cudaMemcpy(d_tbb,bb_patch,sizeof(int)*6,cudaMemcpyHostToDevice);


	dim3 dimBlock3 (8,64);
	dim3 dimGrid3 ( ceil( (float)TOTAL_FERNS / (float)dimBlock3.x),ceil( (float)p_len / (float)dimBlock3.y) );

	// Execute the kernel on the GPU
	train_kernel<<< dimGrid3, dimBlock3 >>>(c_forest_ferns,c_IIdata,width,height,p_len,d_patch,d_tbb);
	cudaDeviceSynchronize();

    delete [] bb_patch;

    cudaFree(d_patch);
    cudaFree(d_tbb);

}

void nvClassifier::bbWarpPatch(double *bb)
{
    // Transformation matrix
    float *m = new float[4];

    // Loop through various rotations and skews
    for (float r = -0.1f; r < 0.1f; r += 0.005f) {
        float sine = sin(r);
        float cosine = cos(r);

        for (float sx = -0.1f; sx < 0.1f; sx += 0.05f) {
            for (float sy = -0.1f; sy < 0.1f; sy += 0.05f) {
                // Set transformation
                /*  Rotation matrix * skew matrix =

                    | cos r   sin r | * | 1   sx | =
                    | -sin r  cos r |   | sy   1 |

                    | cos r + sy * sin r   sx * cos r + sin r |
                    | sy * cos r - sin r   cos r - sx * sin r | */
                m[0] = cosine + sy * sine;
                m[1] = sx * cosine + sine;
                m[2] = sy * cosine - sine;
                m[3] = cosine - sx * sine;

                // Create warp and train classifier
                int *bb_patch=new int[5];
                bb_patch[0]=0;
                bb_patch[1]=0;
                bb_patch[2]=(int)bb[2];
                bb_patch[3]=(int)bb[3];
                bb_patch[4]=1;
                train_warp( bb_patch, bb, m);
                delete [] bb_patch;
            }
        }
    }

    delete m;
}


void nvClassifier::train_warp( int *patch, double *bb_in, float *m)
{
	// Host variables
	int i,j;
	unsigned int num_threads;
	struct one_fern *frn;
	// Initialize CPU variables and allocate required memory
	num_threads = (unsigned int) TOTAL_FERNS;
	int bb[4];
	for(int i=0;i<4;i++)
		bb[i]=(int)bb_in[i];

	// Allocate GPU (device) memory and variables
	int *d_patch;
	int *d_bb;
	float *d_m;


	cudaMalloc((void**) &d_patch, sizeof(int)*5);
	cudaMalloc((void**) &d_bb, sizeof(int)*4);
	cudaMalloc((void**) &d_m, sizeof(float)*4);

	cudaMemcpy(d_patch, patch, sizeof(int)*5, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bb, bb, sizeof(int)*4, cudaMemcpyHostToDevice);
	cudaMemcpy(d_m, m, sizeof(float)*4, cudaMemcpyHostToDevice);

	// Copy vectors from host memory to device memory

	// Setup kernel execution parameters
	int blocksize=16;
	dim3 dimBlock (blocksize);
	dim3 dimGrid( ceil( (float)num_threads / (float)blocksize) );

	// Execute the kernel on the GPU
	train_kernel_warp<<< dimGrid, dimBlock >>>(c_forest_ferns,c_IIdata,width,height,d_patch,d_bb,d_m);
	cudaDeviceSynchronize();
	// Copy result from GPU to CPU

	//////////////////////////////////////////
	// All done - clean up and exit
	//////////////////////////////////////////
	// Free up CPU memory

	// Free up GPU memory

	cudaFree(d_patch);
	cudaFree(d_m);
	cudaFree(d_bb);

}

void nvClassifier::set_II(const unsigned char *imageData, int Width, int Height)
{
	width=Width;
	height=Height;
	/*if(c_IMGdata==NULL)
		cudaMalloc((void**) &c_IMGdata, (width*height)*sizeof(int));
	cudaMemcpy(c_IMGdata, iImage->imageData, (width*height)*sizeof(int), cudaMemcpyHostToDevice);*/
	width++;
	height++;
	if(c_IIdata==NULL)
		cudaMalloc((void**) &c_IIdata, (width*height)*sizeof(int));

	if(planar_data==NULL)
		planar_data=(int*)malloc((width)*(height)*sizeof(int));
    memset(planar_data,0,(width)*(height)*sizeof(int));

    for (int i = 1; i < width; i++) {
        for (int j = 1; j < height; j++) {
        	planar_data[i + j*width] = (int)imageData[(i-1) + (j-1)*(width-1)] +  planar_data[(i-1) + (j)*width] + planar_data[(i) + (j-1)*width] - planar_data[(i-1) + (j-1)*width];

        }
    }

    cudaMemcpy(c_IIdata, planar_data, (width*height)*sizeof(int), cudaMemcpyHostToDevice);
}


double nvClassifier::bbOverlap(double *bb1, double *bb2) {
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


} /* namespace std */
