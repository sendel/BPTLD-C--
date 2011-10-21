/*
 * nvClassifier.cpp
 *
 *  Created on: 17.10.2011
 *      Author: sadmin
 */

#include "nvClassifier.h"
#include "kernels.cuh"


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
			forest_ferns[f].nodes[i][2]=(maxScale - minScale) * ((float)rand() / (float)RAND_MAX) + minScale;
					//(maxScale - minScale) * (float)genrand_int31()/(float)RAND_MAX + minScale;
			forest_ferns[f].nodes[i][3]=(maxScale - minScale) * ((float)rand() / (float)RAND_MAX) + minScale;
			forest_ferns[f].nodes[i][0]= (1.0f - forest_ferns[f].nodes[i][2]) * ((float)rand() / (float)RAND_MAX);
					//(1.0f - forest_ferns[f].nodes[i][2]) * (float)genrand_int31()/(float)RAND_MAX;
			forest_ferns[f].nodes[i][1]=(1.0f - forest_ferns[f].nodes[i][3]) * ((float)rand() / (float)RAND_MAX);
					//(1.0f - forest_ferns[f].nodes[i][3]) * (float)genrand_int31()/(float)RAND_MAX;
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

	// Execute the kernel on the GPU
	train_kernel<<< dimGrid, dimBlock >>>(c_forest_ferns,c_IIdata,width,height,1,d_patch,NULL);
	cudaDeviceSynchronize();
	// Copy result from GPU to CPU

	//////////////////////////////////////////
	// All done - clean up and exit
	//////////////////////////////////////////
	// Free up CPU memory

	// Free up GPU memory

	cudaFree(d_patch);


}

vector<double *> *nvClassifier::detect(double *tbb)
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
        return new vector<double *>();
    }

    // Using the sliding-window approach, find positive matches to our object
    // Vector of positive patch matches' bounding-boxes
    vector<double *> *bbs = new vector<double *>();

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
	float* h_answer;
	h_answer=(float*)malloc(sizeof(float)*p_len);

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
	// Copy result from GPU to CPU
	cudaMemcpy(h_answer, d_answer, sizeof(float)*p_len, cudaMemcpyDeviceToHost);


	int *patch=(int*)malloc(sizeof(int)*p_len*5);
	cudaMemcpy(patch, d_patch, sizeof(int)*5*p_len, cudaMemcpyDeviceToHost);

	for(int f=0;f<p_len;f++)
	{

	 float myanswer=h_answer[0+f];
	 /*for(i=1;i<TOTAL_FERNS;i++)
	 {
		 myanswer+=h_answer[i+f*TOTAL_FERNS];
	 }*/
     double *bb = new double[6];
     bb[0] = (double)patch[f*5];
     bb[1] = (double)patch[f*5+1];
     bb[2] = (double)patch[f*5+2];
     bb[3] = (double)patch[f*5+3];
     bb[4] = (double)myanswer/(float)TOTAL_FERNS;
    // printf("About: %f,%f,%f,%f\n ",bb[0],bb[1],bb[2],bb[3]);
     //if(bb[4]>0.6)
     //printf("About: %f: %f\n ",myanswer, bb[4]);
     if (tbb[5] != -1 && bbOverlap(bb, tbb) > MIN_LEARNING_OVERLAP) {
         bb[5] = 1;
     } else {
         bb[5] = 0;
     }

     // If positive, or negative and overlapping with the tracked
     // bounding-box, add this bounding-box to our return list
     if (bb[4] > 0.5f || bb[5] == 1) {
         bbs->push_back(bb);
     } else {
         delete [] bb;
     }
	}

	//////////////////////////////////////////
	// All done - clean up and exit
	//////////////////////////////////////////
	// Free up CPU memory
	free(h_answer);
	free(patch);

	// Free up GPU memory
	cudaFree(d_answer);
	cudaFree(d_patch);

    return bbs;
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



    int *bb_patch = new int[5];
    for(int i=0;i<4;i++)
    	bb_patch[i]=tbb[i];
    bb_patch[4]=0;

    int *d_tbb;
    cudaMalloc((void**) &d_tbb, sizeof(int)*4);
    cudaMemcpy(d_tbb,bb_patch,sizeof(int)*4,cudaMemcpyHostToDevice);


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
