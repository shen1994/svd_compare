#ifndef _CUDASVD_H
#define _CUDASVD_H

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cusolverDn.h>
#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>
#include "cuda_runtime.h"

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

class cudaSVD
{
public:
	cudaSVD();
	~cudaSVD();
	void malloc_space(uint16_t m, uint16_t n); // matrix m * n	
	void set_matrix(float *matrix, uint16_t num);
	void core_compute();
	float *get_host_u() { return host_U; }

private:
	float *host_U; // m-by-m unitary matrix, left singular vectors
	float *host_V; // n-by-n unitary matrix, right singular vectors
	float *host_S; // numerical singular value
	float *host_A;
	float *dev_U;
	float *dev_V;
	float *dev_S;
	float *dev_A; // device copy from matrix
	int host_info;
	int *dev_info;
	int lwork; // sie of workspace
	float *dev_work;

	uint16_t m_;
	uint16_t n_;
	uint16_t lda;

	double tol;
	int max_sweeps;
	cusolverEigMode_t jobz; // compute eigenvectors
	int econ; // econ = 1 for economy size

	cusolverDnHandle_t cusolverH;
	cudaStream_t stream;
	gesvdjInfo_t gesvdj_params;
};//class	

#endif