/*  Copyright 2011 Ben Pryke.
    This file is part of Ben Pryke's TLD Implementation available under the
    terms of the GNU General Public License as published by the Free Software
    Foundation. This software is provided without warranty of ANY kind. */

#include "IntegralImage.h"


IntegralImage::IntegralImage() {planar_data=NULL;}


void IntegralImage::createFromIplImage(const IplImage *mxImage) {
    // Get pointer
    unsigned char *values = (unsigned char *)mxImage->imageData;
    
    // Get width and height
    width = (int)mxImage->width;
    height = (int)mxImage->height;
    int ii_w=width+1;

    //planar_data = new int [width*height];
    planar_data=(int*)malloc((width+1)*(height+1)*sizeof(int));
    memset(planar_data,0,(width+1)*(height+1)*sizeof(int));
    
    for (int i = 1; i <= width; i++) {
        for (int j = 1; j <= height; j++) {

        	//printf("%d; ",planar_data[i + j*ii_w]);

        	planar_data[i + j*ii_w] = (int)values[(i-1) + (j-1)*width] +  planar_data[(i-1) + (j)*ii_w] + planar_data[(i) + (j-1)*ii_w] - planar_data[(i-1) + (j-1)*ii_w];
        	//planar_data[i + j*ii_w] = 130;
        	//data[i][j] = values[(i-1) + (j-1)*(width-1)] + data[i - 1][j] + data[i][j - 1] - data[i - 1][j - 1];

        			//data[i - 1][j] + data[i][j - 1] - data[i - 1][j - 1];
            //II(x,y) = I(x,y) — II(x-1,y-1) + II(x,y-1) + II(x-1,y)
        }
    }
    width++;
    height++;

  //  printf("\n");

}


void IntegralImage::createFromIntegralImage(IntegralImage *image, int x, int y, int w, int h) {
    // Check we don't exceed image dimensions
    // Note: assumes all parameters are positive
    if (x + w <= image->getWidth() && y + h <= image->getHeight()) {
        width = w;
        height = h;


        //planar_data = new int [width*height];
        planar_data=(int*)malloc((width)*(height)*sizeof(int));
        memset(planar_data,0,width*height*sizeof(int));

        int *imageData = image->getData();

        
        // Assign each value in our data array to a pointer to the correct
        // element in our original array

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
            	planar_data[i + j*width] = imageData[x + y*image->getWidth()];
            }
        }

  /*      for (int i = 0; i <= w; i++) {

            data[i] = &(imageData[x + i][y]);
        }*/
    } else {
        printf("ERROR: IMAGE SIZE OUT OF BOUNDS! (%d, %d, %d, %d)\n", x, y, w, h);
    }
}


void IntegralImage::createWarp(IntegralImage *image, double *bb, float *m) {
    // Initialise variables
    // Remember an IntegralImage has dimensions (width + 1)x(height + 1)
    width = (int)bb[2];
    height = (int)bb[3];

    //planar_data = new int [(width+1)*(height+1)];
    planar_data=(int*)malloc((width)*(height)*sizeof(int));
    memset(planar_data,0,(width)*(height)*sizeof(int));

    int *imageData = image->getData();
    
    // Get centre of bounding-box (cx, cy) and the offset relative to this of
    // the top-left of the bounding-box (ox, oy)
    int ox = -(int)((width) * 0.5);
    int oy = -(int)((height) * 0.5);
    int cx = (int)(bb[0] - ox);
    int cy = (int)(bb[1] - oy);
    
    // Loop through pixels of this image, width then height, calculating the
    // position of corresponding pixels in the source image
    for (int x = ox; x < ox + width; x++) {
        for (int y = oy; y < oy + height; y++) {
            int xp = (int)(m[0] * (float)x + m[1] * (float)y + cx);
            int yp = (int)(m[2] * (float)x + m[3] * (float)y + cy);
            
            // Limit pixels to those in the given bounding-box
            xp = std::max(std::min(xp, (int)bb[0] + width), (int)bb[0]);
            yp = std::max(std::min(yp, (int)bb[1] + height), (int)bb[1]);
            
            planar_data[(x - ox) + (y - oy)*(width)] = imageData[xp + yp*image->getWidth()];
        }
    }
    //width++;
    //height++;

}


int IntegralImage::sumRect(int x, int y, int w, int h) {
    // Note: assumes all parameters are positive and within the image bounds
    if (x >= 0 && w > 0 && x + w < width && y >= 0 && h > 0 && y + h < height) {
    	int dx=planar_data[x+y*width];
    	int dy=planar_data[(x+w)+(y+h)*width];
    	int dw=planar_data[(x+w)+y*width];
    	int dh=planar_data[x+(y+h)*width];
    	return dx+dy-dw-dh;
        //return data[x][y] + data[x + w][y + h] - data[x + w][y] - data[x][y + h];
    } else {
        printf("ERROR: SUM RECT OUT OF BOUNDS! (%d, %d, %d, %d)\n", x, y, w, h);
        return 0;
    }

/*
    if (x >= 0 && w > 0 && x + w < width && y >= 0 && h > 0 && y + h < height) {
        return planar_data[x+y*width] + planar_data[(x+w)+(y+h)*width] - planar_data[(x+w)+y*width] - planar_data[x+(y+h)*width];
    } else {
        printf("ERROR: SUM RECT OUT OF BOUNDS! (%d, %d, %d, %d)\n", x, y, w, h);
        return 0;
    }
*/

}


int IntegralImage::getWidth() {
    return width;
}


int IntegralImage::getHeight() {
    return height;
}


int *IntegralImage::getData() {
    return planar_data;
}


void IntegralImage::setData(int *d) {
	planar_data = d;
}


IntegralImage::~IntegralImage() {
    // Only delete all the data if it was created from Matlab

    if(planar_data!=NULL)
    	free(planar_data);
}
