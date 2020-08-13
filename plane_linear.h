#ifndef _PLANELINEAR_H
#define _PLANELINEAR_H

#include <Eigen/Dense>
#include "cuda_svd.h"

#define DATA_AXIS 3

class PlaneLinear
{
public:
	PlaneLinear(uint16_t m, bool mode);
	~PlaneLinear();	
	bool plane_fit(float *points, uint16_t num, float *params);

private:
	bool mode_; // 0-cpu 1-gpu
	uint16_t m_;
	cudaSVD *cuda_svd;
	float *host_A;
};

#endif