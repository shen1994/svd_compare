#include <iostream>
#include "cuda_svd.h"
#include <Eigen/Dense>

int main(int argc, char *argv[])
{
	const long int m = 10, n = 3;
	float *host_A = (float*)malloc(m * n * sizeof(float));
	float *host_B = (float*)malloc(m * n * sizeof(float));

	// set points value
	host_A[0] = 1.130000; host_A[1] = 0.135000; host_A[2] = -0.608000;
	host_A[3] = 2.807000; host_A[4] = 0.892000; host_A[5] = -0.640000;
	host_A[6] = 2.212000; host_A[7] = 1.252000; host_A[8] = -0.537000;
	host_A[9] = 4.268000; host_A[10] = 0.560000; host_A[11] = -0.647000;
	host_A[12] = 0.965000; host_A[13] = 0.238000; host_A[14] = -0.615000;
	host_A[15] = 2.344000; host_A[16] = -0.477000; host_A[17] = -0.574000;
	host_A[18] = 2.019000; host_A[19] = 0.564000; host_A[20] = -0.629000;
	host_A[21] = 2.704000; host_A[22] = 1.248000; host_A[23] = -0.655000;
	host_A[24] = 2.935000; host_A[25] = 1.131000; host_A[26] = -0.650000;
	host_A[27] = 3.529000; host_A[28] = -1.193000; host_A[29] = -0.596000;

	// time print	
	cudaEvent_t cuda_time_s;
	cudaEvent_t cuda_time_e;
	cudaEventCreate(&cuda_time_s);
	cudaEventCreate(&cuda_time_e);
	float cuda_time_ms;

	cudaSVD cuda_svd;
	cuda_svd.malloc_space(n, m);

	for(int loop = 0; loop < 1000; loop++)
	{
		Eigen::MatrixXf pc_operate(n, m);
		for(uint32_t i = 0; i < m; i ++)
		{
			pc_operate(0, i) = host_A[i * 3];
			host_B[i] = host_A[i * 3];
			pc_operate(1, i) = host_A[i * 3 + 1];
			host_B[m + i] = host_A[i * 3 + 1];
			pc_operate(2, i) = host_A[i * 3 + 2];
			host_B[m * 2 + i] = host_A[i * 3 + 2];
		}

    	Eigen::Vector3f centroid = pc_operate.rowwise().mean(); 
    	pc_operate.colwise() -= centroid; 

		cudaEventRecord(cuda_time_s, 0);

		Eigen::JacobiSVD<Eigen::MatrixXf> svd(pc_operate, Eigen::DecompositionOptions::ComputeFullU);

		cudaEventRecord(cuda_time_e,0);
		cudaEventSynchronize(cuda_time_e);
		cudaEventElapsedTime(&cuda_time_ms, cuda_time_s, cuda_time_e);
		printf("%s's Time: %fms\n", "Eigen", cuda_time_ms);

		Eigen::Vector3f normal = svd.matrixU().col(2); 
		std::cout<< normal <<std::endl;

		float d = -normal.dot(centroid);

		printf("1d: %f\n", d);

		cudaEventRecord(cuda_time_s, 0);

		cuda_svd.set_matrix(pc_operate.data());
		cuda_svd.core_compute();
		float *host_U = cuda_svd.get_host_u();

		cudaEventRecord(cuda_time_e,0);
		cudaEventSynchronize(cuda_time_e);
		cudaEventElapsedTime(&cuda_time_ms, cuda_time_s, cuda_time_e);
		printf("%s's Time: %fms\n", "Cuda", cuda_time_ms);

		normal(0) = host_U[6];
		normal(1) = host_U[7];
		normal(2) = host_U[8];

		printf("%f  %f  %f\n", host_U[0], host_U[1], host_U[2]);
		printf("%f  %f  %f\n", host_U[3], host_U[4], host_U[5]);
		printf("%f  %f  %f\n", host_U[6], host_U[7], host_U[8]);

		d = -normal.dot(centroid);

		printf("2d: %f\n", d);
	}


	cudaEventDestroy(cuda_time_s);
	cudaEventDestroy(cuda_time_e);

	free(host_A); host_A = nullptr;
	free(host_B); host_B = nullptr;
}