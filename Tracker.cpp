/*  Copyright 2011 Ben Pryke.
    This file is part of Ben Pryke's TLD Implementation available under the
    terms of the GNU General Public License as published by the Free Software
    Foundation. This software is provided without warranty of ANY kind. */

#include "Tracker.h"


Tracker::Tracker(int frameWidth, int frameHeight, CvSize *frameSize, IplImage *firstFrame, Classifier *classifier) {
    width = frameWidth;
    height = frameHeight;
    prevFrame = cvCloneImage(firstFrame);
    prevPyramid = cvCreateImage(*frameSize, IPL_DEPTH_8U, 1);
    nextPyramid = cvCreateImage(*frameSize, IPL_DEPTH_8U, 1);
    prevPoints = (CvPoint2D32f *)cvAlloc(TOTAL_POINTS * sizeof(CvPoint2D32f));
    nextPoints = (CvPoint2D32f *)cvAlloc(TOTAL_POINTS * sizeof(CvPoint2D32f));
    //predPoints = (CvPoint2D32f *)cvAlloc(TOTAL_POINTS * sizeof(CvPoint2D32f));
    windowSize = (CvSize *)malloc(sizeof(CvSize));
    *windowSize = cvSize(WINDOW_SIZE, WINDOW_SIZE);
    status = (char *)cvAlloc(TOTAL_POINTS);
    //predStatus = (char *)cvAlloc(TOTAL_POINTS);
    termCriteria = (TermCriteria *)malloc(sizeof(TermCriteria));
    *termCriteria = TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
    this->classifier = classifier;
}


float Tracker::median(float *A, int length) {
    int index = (int)(length / 2);
    std::sort(A, A + length);
    
    if (length % 2 == 0) {
        return (A[index] + A[index + 1]) / 2;
    }
    else {
        return A[index];
    }
}


double *Tracker::track(IplImage *nextFrame, IntegralImage *nextFrameIntImg, double *bb) {
    // Perform Lucas-Kanade Tracking -----------------------------------------
    // Distribute points to track uniformly over the bounding-box
    double bbWidth = bb[2];
    double bbHeight = bb[3];
    double stepX = bbWidth / (DIM_POINTS + 1);
    double stepY = bbHeight / (DIM_POINTS + 1);
    int i, x, y;
    
    for (i = 0, x = 1; x <= DIM_POINTS; x++) {
        for (y = 1; y <= DIM_POINTS; y++, i++) {
            prevPoints[i].x = (float)(bb[0] + x * stepX);
            prevPoints[i].y = (float)(bb[1] + y * stepY);
            nextPoints[i].x = prevPoints[i].x;
            nextPoints[i].y = prevPoints[i].y;
        }
    }
    
    // Calculate optical flow with the iterative Lucas-Kanade method in pyramids
    // Last parameter flag meanings:
    // CV_LKFLOW_PYR_A_READY: pyramid A is precalculated before the call
    // CV_LKFLOW_PYR_B_READY: pyramid B is precalculated before the call
    // CV_LKFLOW_INITIAL_GUESSES: array B contains initial coordinates of features before the function call
    cvCalcOpticalFlowPyrLK(prevFrame, nextFrame, prevPyramid, nextPyramid, prevPoints, nextPoints, TOTAL_POINTS, *windowSize, LEVEL, status, 0, *termCriteria, CV_LKFLOW_INITIAL_GUESSES);
    //cvCalcOpticalFlowPyrLK(nextFrame, prevFrame, nextPyramid, prevPyramid, nextPoints, predPoints, TOTAL_POINTS, *windowSize, LEVEL, predStatus, 0, *termCriteria, CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY);
    
    
    // Calculate Bounding-Box Displacement -----------------------------------
    // Calculate the median displacement of the bounding-box in each dimension
    // We individually calculate the displacement of each point that was
    // successfully tracked and then find the median values in each dimension
    float *dxs = (float *)malloc(TOTAL_POINTS * sizeof(float));
    float *dys = (float *)malloc(TOTAL_POINTS * sizeof(float));
    int successful = 0;
    
    for (i = 0, x = 0; x < DIM_POINTS; x++) {
        for (y = 0; y < DIM_POINTS; y++, i++) {
            if (status[i] == 1) {
                dxs[successful] = nextPoints[i].x - prevPoints[i].x;
                dys[successful] = nextPoints[i].y - prevPoints[i].y;
                successful++;
            }
        }
    }
    
    // Get the median displacements
    double dispX = (double)median(dxs, successful);
    double dispY = (double)median(dys, successful);
    
    
    // Calculate Bounding-Box Scale Change -----------------------------------
    // For each possible pair of points we compute the ratio of distance apart
    // at time t + 1 to distance apart at time t
    // We then take the median ratio as the scale factor for the bounding-box
    // First, compute the maximum number of pairs in order to allocate our
    // ratios array
    int comparisons = 0;
    
    for (i = 1; i < TOTAL_POINTS; i++) {
        comparisons += i;
    }
    
    // Allocate the ratios array for storing each ratio and reset comparisons
    // to zero so we can count the number of successful comparisons
    float *scales = (float *)malloc(comparisons * sizeof(float));
    comparisons = 0;
    
    for (i = 0; i < TOTAL_POINTS; i++) {
        for (int j = i + 1; j < TOTAL_POINTS; j++) {
            if (status[i] == 1 && status[j] == 1) {
                float dxPrev = prevPoints[j].x - prevPoints[i].x;
                float dyPrev = prevPoints[j].y - prevPoints[i].y;
                float dxNext = nextPoints[j].x - nextPoints[i].x;
                float dyNext = nextPoints[j].y - nextPoints[i].y;
                float sPrev = sqrt(dxPrev * dxPrev + dyPrev * dyPrev);
                float sNext = sqrt(dxNext * dxNext + dyNext * dyNext);
                
                if (sPrev != 0 && sNext != 0) {
                    scales[comparisons] = sNext / sPrev;
                    comparisons++;
                }
            }
        }
    }
    
    // Get the median scale factor
    double scale = (double)median(scales, comparisons);
    
    // Calculate the offset of the bounidng-box in x and y directions
    // We half the result because the bounding-box is made to expand about its
    // centre
    double offsetX = 0.5f * bbWidth * (scale - 1);
    double offsetY = 0.5f * bbHeight * (scale - 1);
    
    // Free memory
    cvReleaseImage(&prevFrame);
    prevFrame = cvCloneImage(nextFrame);
    free(dxs);
    free(dys);
    free(scales);
    
    
    // Set output ------------------------------------------------------------
    // We output the estimated new top-left and bottom-right coordinates of
    // the bounding-box. [top-left x, top-left y, width, height]
    double *bbNew = new double[5];
    bbNew[0] = bb[0] - offsetX + dispX;
    bbNew[1] = bb[1] - offsetY + dispY;
    bbNew[2] = bb[2] + offsetX * 2;
    bbNew[3] = bb[3] + offsetY * 2;
   
	//避免边框盒左上角坐标出现负值或者右下角坐标大于帧宽高
	if((bbNew[0] < 0)||(bbNew[1] < 0)||((bbNew[0]+bbNew[2]) > nextFrame->width)||((bbNew[1]+bbNew[3]) > nextFrame->height))
	{
		bbNew[0] = 0;
		bbNew[1] = 0;
		bbNew[2] = 0;
		bbNew[3] = 0;
		bbNew[4] = 0;
	}
	else
	{
		//只有在跟踪器输出坐标有效的情况下,才有必要计算信任度
		bbNew[4] = (double)classifier->classify(nextFrameIntImg, (int)bbNew[0], (int)bbNew[1], (int)bbNew[2], (int)bbNew[3]);
	}
    
	
	//cout<<"跟踪器输出的边框盒的信任度:"<<bbNew[4]<<endl;
	//cout<<"bbNew[0] = "<<bbNew[0]<<endl;
	//cout<<"bbNew[1] = "<<bbNew[1]<<endl;
	//cout<<"bbNew[2] = "<<bbNew[2]<<endl;
	//cout<<"bbNew[3] = "<<bbNew[3]<<endl;
    
	return bbNew;
}


void Tracker::setPrevFrame(IplImage *frame) {
    cvReleaseImage(&prevFrame);
    prevFrame = cvCloneImage(frame);
}


Tracker::~Tracker() {
    free(prevFrame);
    cvReleaseImage(&prevPyramid);
    cvReleaseImage(&nextPyramid);
    cvFree(&prevPoints);
    cvFree(&nextPoints);
    //cvFree(&predPoints);
    free(windowSize);
    cvFree(&status);
    //cvFree(&predStatus);
    free(termCriteria);
}
