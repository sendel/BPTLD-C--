/*
 * nvClassifier.h
 *
 *  Created on: 17.10.2011
 *      Author: www.VarnaSoftware.com,  SYavorovsky@varnasoftware.com
 */

#ifndef NVCLASSIFIER_H_
#define NVCLASSIFIER_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
//#include <cuda_runtime_api.h>

//int leafNodes = (int)pow(2.0f * (float)POWER, nodeCount);

//TOTAL_FERNS < MultiProcessors
#define TOTAL_FERNS 32
#define TOTAL_NODES 6
#define LEAF_NODES 4096

// Constants -----------------------------------------------------------------
// The minimum percentage overlap between a tracked and detected patch that
// counts as an overlap
#define MIN_LEARNING_OVERLAP 0.6

struct one_fern {
	int p[LEAF_NODES];
	int n[LEAF_NODES];
	float posteriors[LEAF_NODES];
	float nodes[TOTAL_NODES][4];
};

namespace std {

class nvClassifier {
private:
	struct one_fern *c_forest_ferns;
	int *c_IMGdata;
	int *c_IIdata;
	int width,height;
	int *c_warpII;
	int warpII[4];
	int *planar_data;
public:
	nvClassifier(float minScale, float maxScale);
	virtual ~nvClassifier();

	float classify(int* patch);
	void train(int* patch);
	void detect(double *tbb, double confidence);
	void set_II(const unsigned char *iImage, int Width, int Height); //get IPL and set II to cuda
	void bbWarpPatch(double *bb); //train for init positive
	void trainNegative(double *tbb); //train for init negative
	double bbOverlap(double *bb1, double *bb2);
	void train_warp( int *patch, double *bb, float *m);

};

} /* namespace std */
#endif /* NVCLASSIFIER_H_ */
