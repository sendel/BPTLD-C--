/*  Copyright 2011 Ben Pryke.
    This file is part of Ben Pryke's TLD Implementation available under the
    terms of the GNU General Public License as published by the Free Software
    Foundation. This software is provided without warranty of ANY kind. */

#include "cv.h"
#include "highgui.h"
#include "Detector.h"
#include "Classifier.h"
#include "Tracker.h"
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <iostream>
using namespace std;


///全局变量 ==================================================================
// 常量      -----------------------------------------------------------------
// 分类器中蕨的数量
#define TOTAL_FERNS 13

// 每颗蕨的节点数
#define TOTAL_NODES 7

// 特征小块与整幅图像的最小的高度宽度百分比
#define MIN_FEATURE_SCALE 0.1f

// 特征小块与整幅图像的最大的高度宽度百分比
#define MAX_FEATURE_SCALE 0.5f

// 上一帧轨迹上的小块的最小的信任度，用于本帧中学习
#define MIN_LEARNING_CONF 0.8

// 当探测出的小块中有一个信任度高于跟踪器跟踪的小块,
// 它必须也高于这个值，然后才能用于重新初始化跟踪器
// 注意: MIN_REINIT_CONF 应当 <= MIN_LEARNING_CONF
#define MIN_REINIT_CONF 0.7

// 上一帧跟踪器跟踪出的小块的最小的信任度,用于下一帧中继续跟踪
#define MIN_TRACKING_CONF 0.15


// 变量 -----------------------------------------------------------------
// 我们的分类器，跟踪器和探测器
static Classifier *classifier;
static Tracker *tracker;
static Detector *detector;

// 让我们知道TLD是否被初始化过
static bool initialised = false;

// 初始化边框盒的大小
static float initBBWidth;
static float initBBHeight;

// 每一帧的大小
int frameWidth;
int frameHeight;
CvSize *frameSize;

// 上一帧的轨迹上的小块的信任度
double confidence;



/// 方法 ==================================================================
/*  将Matlab图像转换为IplImage
    Returns: the converted image.
    mxImage: the image straight from Matlab */
//IplImage *imageFromMatlab(const mxArray *mxImage) {
//    // Get pointer
//    unsigned char *values = (unsigned char *)mxGetPr(mxImage);
//    
//    // Create our return image
//    IplImage *image = cvCreateImage(*frameSize, IPL_DEPTH_8U, 1);
//    
//    // Loop through the new image getting values from the old one
//    // Note: values are effectively rotated 90 degrees
//    for (int i = 0; i < frameWidth; i++) {
//        for (int j = 0; j < frameHeight; j++) {
//            image->imageData[j * frameWidth + i] = values[i * frameHeight + j];
//        }
//    }
//    
//    return image;
//}


//  用边框盒围住的小块的仿射变换来训练分类器
//  frame: 输入的用于仿射变换的图像
//  bb: 第一帧中的边框盒[x,y,width,height]
void bbWarpPatch(IntegralImage *frame, double *bb) {
    // 变换矩阵
    float *m = new float[4];
    
	// 循环穿过各种旋转和斜交参数
    for (float r = -0.1f; r < 0.1f; r += 0.005f) {
        float sine = sin(r);
        float cosine = cos(r);
        
        for (float sx = -0.1f; sx < 0.1f; sx += 0.05f) {
            for (float sy = -0.1f; sy < 0.1f; sy += 0.05f) {
                // 设置转换矩阵
                /*  Rotation matrix * skew matrix =
                    
                    | cos r   sin r | * | 1   sx | = 
                    | -sin r  cos r |   | sy   1 |
                    
                    | cos r + sy * sin r   sx * cos r + sin r |
                    | sy * cos r - sin r   cos r - sx * sin r | */
                m[0] = cosine + sy * sine;
                m[1] = sx * cosine + sine;
                m[2] = sy * cosine - sine;
                m[3] = cosine - sx * sine;
                
				// 创建仿射变换，然后训练分类器
                IntegralImage *warp = new IntegralImage();
                warp->createWarp(frame, bb, m);
                classifier->train(warp, 1, 1, (int)bb[2], (int)bb[3], 1);
                delete warp;
            }
        }
    }
    
    delete m;
}


//  用负向训练小块训练分类器,也就是第一帧中与初始边框盒覆盖率小于一定程度的小块
//  frame: 输入的用于仿射变换的图像
//  tbb: 第一帧中的边框盒[x,y,width,height]
void trainNegative(IntegralImage *frame, double *tbb) {
	// 边框盒的最小和最大尺度
    float minScale = 0.5f;
    float maxScale = 1.5f;
	// 尺度迭代的数量
    int iterationsScale = 5;
	// 每次迭代的增量
    float scaleInc = (maxScale - minScale) / (iterationsScale - 1);
    
	// 循环通过一系列的边框盒尺度
    for (float scale = minScale; scale <= maxScale; scale += scaleInc) {
        int minX = 0;
        int currentWidth = (int)(scale * initBBWidth);
        if(currentWidth>=initBBWidth)currentWidth=initBBWidth-1;
        int maxX = initBBWidth - currentWidth;
        float iterationsX = 20.0;
        int incX = (int)floor((float)(maxX - minX) / (iterationsX - 1.0));
        if(incX==0)incX=1;
        
	            // Same for y
            int minY = 0;
            int currentHeight = (int)(scale * (float)initBBHeight);
            if(currentHeight>=initBBHeight)currentHeight=initBBHeight-1;
            int maxY = initBBHeight - currentHeight;
            float iterationsY = 20.0;
            int incY = (int)floor((float)(maxY - minY) / (iterationsY - 1));
            if(incY==0)incY=1;
        // Loop through all bounding-box top-left x-positions
        for (int x = minX; x <= maxX; x += incX) {

            
            // Loop through all bounding-box top-left x-positions
            for (int y = minY; y <= maxY; y += incY) {
				// 定义小块，测试它与初始小块的覆盖率是否少于MIN_LEARNING_OVERLAP,
				// 如果是，那么就当做负向样本用来训练
                double *bb = new double[4];
                bb[0] = (double)x;
                bb[1] = (double)y;
                bb[2] = (double)currentWidth;
                bb[3] = (double)currentHeight;
                
                if (Detector::bbOverlap(tbb, bb) < MIN_LEARNING_OVERLAP) {
                    classifier->train(frame, x, y, currentWidth, currentHeight, 0);
                } else {
                    //classifier->train(frame, x, y, currentWidth, currentHeight, 1);
                }
                
                delete [] bb;
            }
        }
    }
}

// 初始化 --------------------------------------------------------
// 参数1帧宽,参数2帧高,参数3初始第一帧,参数4初始边框盒
void BpTld_Init(int Width,int Height,IplImage * firstImage,double * firstbb)
{
	// 获取图像参数
	frameWidth = Width;
	frameHeight = Height;
	frameSize = (CvSize *)malloc(sizeof(CvSize));
	*frameSize = cvSize(frameWidth, frameHeight);
	IntegralImage *firstFrame = new IntegralImage();
	firstFrame->createFromIplImage(firstImage);
	IplImage *firstFrameIplImage = firstImage;
	double *bb = firstbb;
	initBBWidth = (float)bb[2];
	initBBHeight = (float)bb[3];
	//初始化信任度
	confidence = 1.0f;

	// 初始化分类器,跟踪器和探测器
	srand((unsigned int)time(0));
	classifier = new Classifier(TOTAL_FERNS, TOTAL_NODES, MIN_FEATURE_SCALE, MAX_FEATURE_SCALE);
	tracker = new Tracker(frameWidth, frameHeight, frameSize, firstFrameIplImage, classifier);
	detector = new Detector(frameWidth, frameHeight, bb, classifier);

	// 用初始图像小块和它的仿射变形来训练分类器
	classifier->train(firstFrame, (int)bb[0], (int)bb[1], (int)initBBWidth, (int)initBBHeight, 1);
	bbWarpPatch(firstFrame, bb);
	trainNegative(firstFrame, bb);

	// 释放内存
	delete firstFrame;
	// 设置bool值initialised
	initialised = true;

	return;
}

/*  MEX入口点.
    两种使用方法:
    初始化:
        TLD(帧宽, 帧高, 第一帧, 选择出的边框盒)
    处理每一帧:
        新的轨迹边框盒 = TLD(当前帧, 轨迹边框盒)
*/
void BpTld_Process(IplImage * NewImage,double * ttbb,double * outPut) {
    
    // 获得输入 -------------------------------------------------------------
    // 当前帧
    IplImage *nextFrame = NewImage;
    IntegralImage *nextFrameIntImg = new IntegralImage();
    nextFrameIntImg->createFromIplImage(NewImage);
    
	// 轨迹边框盒[x, y, width, height]
    double *bb = ttbb;
    
    
    // 跟踪 + 探测 ------------------------------------------------------
	// 只有在上个迭代中我们足够自信的时候,才跟踪
	// 从这开始，跟踪器处理下一帧内存
    double *tbb;
    vector<double *> *dbbs;
    
    if (confidence > MIN_TRACKING_CONF) {
		//cout<<"跟踪前."<<endl;
        tbb = tracker->track(nextFrame, nextFrameIntImg, bb);
		//cout<<"跟踪后,探测前."<<endl;
		if (tbb[4] > MIN_TRACKING_CONF)
		{
			dbbs = detector->detect(nextFrameIntImg, tbb);
		} 
		else
		{
			dbbs = detector->detect(nextFrameIntImg, NULL);
		}
        
		//cout<<"探测后."<<endl;
    } else {
        dbbs = detector->detect(nextFrameIntImg, NULL);
        tracker->setPrevFrame(nextFrame);
        tbb = new double[5];
        tbb[0] = 0;
        tbb[1] = 0;
        tbb[2] = 0;
        tbb[3] = 0;
        tbb[4] = MIN_TRACKING_CONF;
    }
    
    
    // 学习 -----------------------------------------------------------------
    // 获得最好的探测出的小块的信任度
    double dbbMaxConf = 0.0f;
    int dbbMaxConfIndex = -1;
    
    for (int i = 0; i < dbbs->size(); i++) {
        double dbbConf = dbbs->at(i)[4];
        
        if (dbbConf > dbbMaxConf) {
            dbbMaxConf = dbbConf;
            dbbMaxConfIndex = i;
        }
    }
    
	// 如果探测出的小块中有一个信任度最高，并且大于MIN_REINIT_CONF
	// 那么就重置跟踪器的边框盒
    if (dbbMaxConf > tbb[4] && dbbMaxConf > MIN_REINIT_CONF) {
        delete tbb;
        tbb = new double[5];
        double *dbb = dbbs->at(dbbMaxConfIndex);
        tbb[0] = dbb[0];
        tbb[1] = dbb[1];
        tbb[2] = dbb[2];
        tbb[3] = dbb[3];
        tbb[4] = dbb[4];
    }
    
	// 如果跟踪出的小块的信任度最高并且最后一帧时足够自信，那么就启动约束。
    else if (tbb[4] > dbbMaxConf && confidence > MIN_LEARNING_CONF) {
        for (int i = 0; i < dbbs->size(); i++) {
			// 用正向和负向小块训练分类器
			// 正向-与被跟踪的小块重合
			// 负向-被分类为正向但与被跟踪的小块不重合
            double *dbb = dbbs->at(i);
            
            if (dbb[5] == 1) {
                classifier->train(nextFrameIntImg, (int)dbb[0], (int)dbb[1], (int)dbb[2], (int)dbb[3], 1);
            }
            else if (dbb[5] == 0) {
                classifier->train(nextFrameIntImg, (int)dbb[0], (int)dbb[1], (int)dbb[2], (int)dbb[3], 0);
            }
        }
    }
    
	// 为下个迭代设置信任度
    confidence = tbb[4];

	//设置输出
	outPut[0] = tbb[0];
	outPut[1] = tbb[1];
	outPut[2] = tbb[2];
	outPut[3] = tbb[3];

    
    
	//////////////////////////////////////////////////////////////////////////
	//此处tbb[0],tbb[1],tbb[2],tbb[3]即是最终的边框盒
	//如果tbb[2],tbb[3]全都大于0,,就表明估算出物体位置
	//////////////////////////////////////////////////////////////////////////
    // 设置输出 ------------------------------------------------------------
	// 我们输出一系列边框盒;第一个是跟踪出的小块，剩下的是探测出的小块
    // Rows correspond to individual bounding boxes
    // Columns correspond to [x, y, width, height, confidence, overlapping]
    
    // 释放内存
    free(tbb);
    dbbs->clear();
    delete nextFrameIntImg;
}


// 保留，用于将来完全C++化时的入口
//////////////////////////////////////////////////////////////////////////
Rect box;
bool drawing_box = false;
bool gotBB = false;

//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
	switch( event ){
  case CV_EVENT_MOUSEMOVE:
	  if( drawing_box ){
		  box.width = x-box.x;
		  box.height = y-box.y;
	  }
	  break;
  case CV_EVENT_LBUTTONDOWN:
	  drawing_box = true;
	  box = Rect( x, y, 0, 0 );
	  break;
  case CV_EVENT_LBUTTONUP:
	  drawing_box = false;
	  if( box.width < 0 ){
		  box.x += box.width;
		  box.width *= -1;
	  }
	  if( box.height < 0 ){
		  box.y += box.height;
		  box.height *= -1;
	  }
	  gotBB = true;
	  break;
	}
}
void drawBox(IplImage * image, CvRect box, cv::Scalar color = cvScalarAll(255), int thick=1); 
//////////////////////////////////////////////////////////////////////////
int main() 
{
	CvSize dst_size;
	dst_size.width = 640;
	dst_size.height = 480;
	//////////////////////////////////////////////////////
	CvCapture* m_pCapture = cvCreateCameraCapture(CV_CAP_ANY);
	if( NULL == m_pCapture)
	{
		assert(!"摄像头初始化失败");
		return false;
	}
	//////////////////////////////////////////////////////
	//Register mouse callback to draw the bounding box
	cvNamedWindow("TLD",CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback( "TLD", mouseHandler, NULL );

	//读取第一帧
	IplImage *pNewCapImg = cvQueryFrame(m_pCapture);
	IplImage *dst = cvCreateImage(dst_size,pNewCapImg->depth,pNewCapImg->nChannels);
	//灰度图,用于运算
	IplImage *dst_gray = cvCreateImage(dst_size,IPL_DEPTH_8U,1);
	cvResize(pNewCapImg,dst);


GETBOUNDINGBOX:
	while(!gotBB)
	{

		pNewCapImg = cvQueryFrame(m_pCapture);
		cvResize(pNewCapImg,dst);
		drawBox(dst,box);
		cvShowImage("TLD",dst);

		if (cvWaitKey(33) == 'q')
			return 0;
	}
	if (min(box.width,box.height)<24){
		cout << "Bounding box too small, try again." << endl;
		gotBB = false;
		goto GETBOUNDINGBOX;
	}
	//Remove callback
	cvSetMouseCallback( "TLD", NULL, NULL );
	printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);

	//////////////////////////////////////////////////////////////////////////
	//TLD的初始化工作
	double * BBox = new double[4];
	double * BBox_out = new double[4];
	BBox[0] = box.x;BBox[1] = box.y;BBox[2] = box.width;BBox[3] = box.height;
	BBox_out[0] = 0;BBox_out[1] = 0;BBox_out[2] = 0;BBox_out[3] = 0;

	cvCvtColor(dst,dst_gray,CV_RGB2GRAY);
	BpTld_Init(dst_size.width,dst_size.height,dst_gray,BBox);
	//////////////////////////////////////////////////////////////////////////
	

	///Run-time
	//BoundingBox pbox;
	int frames = 0;
	int detections = 0;
	while(true)
	{
		IplImage *dst_loop = cvCreateImage(dst_size,IPL_DEPTH_8U,1);
		//获取当前时间
		double t = (double)getTickCount();
		//获取图像
		//用于运算
		pNewCapImg = cvQueryFrame(m_pCapture);
		cvResize(pNewCapImg,dst);
		cvCvtColor(dst,dst_loop,CV_RGB2GRAY);
		//处理每一帧
		BpTld_Process(dst_loop,BBox,BBox_out);
		BBox[0] = BBox_out[0];
		BBox[1] = BBox_out[1];
		BBox[2] = BBox_out[2];
		BBox[3] = BBox_out[3];

		if (BBox_out[2]>0 && BBox_out[3]>0)
		{
			CvRect pBox;
			pBox.x = BBox_out[0];
			pBox.y = BBox_out[1];
			pBox.width = BBox_out[2];
			pBox.height = BBox_out[3];
			drawBox(dst,pBox);
			detections++;
		}

		//Display
		cvShowImage("TLD",dst);
		frames++;
		printf("Detection rate: %d/%d\n",detections,frames);
		//获取时间差
		t=(double)getTickCount()-t;
		t=getTickFrequency()/t;
		printf("帧速: %g \n", t);
		if (cvWaitKey(33) == 'q')
			break;
		cvReleaseImage(&dst_loop);
	}
	cvReleaseImage(&pNewCapImg);
	cvReleaseImage(&dst);
	cvReleaseImage(&dst_gray);
	return 0;
}

void drawBox(IplImage * image, CvRect box, Scalar color, int thick){
	cvRectangle( image, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height),color, thick);
} 