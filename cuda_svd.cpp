#include "cuda_svd.h"

cudaSVD::cudaSVD()
{

}	

cudaSVD::~cudaSVD()
{
	if(nullptr != host_U) cudaFreeHost(host_U);
	if(nullptr != host_V) cudaFreeHost(host_V);
	if(nullptr != host_S) cudaFreeHost(host_S);
	if(nullptr != dev_U) cudaFree(dev_U);
	if(nullptr != dev_V) cudaFree(dev_V);
	if(nullptr != dev_S) cudaFree(dev_S);
	if(nullptr != dev_A) cudaFree(dev_A);
	if(nullptr != dev_info) cudaFree(dev_info);
	if(nullptr != dev_work) cudaFree(dev_work);
	if(nullptr != cusolverH) cusolverDnDestroy(cusolverH);
	if(nullptr != stream) cudaStreamDestroy(stream);
	if(nullptr != gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
}

void cudaSVD::malloc_space(uint16_t m, uint16_t n)
{
	// malloc space needed
	lda = std::min(m, n);
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_U, m * lda * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_V, n * lda * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMallocHost((void**)&host_S, lda * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_U, m * lda * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_V, n * lda * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_S, lda * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_A, m * n * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_info, sizeof(int)));

	// configuration of gesvdj
	tol = 1.e-7;
	max_sweeps = 100;
	jobz = CUSOLVER_EIG_MODE_VECTOR;
	econ = 1;
	m_ = m;
	n_ = n;

	// create cusolver handle, bind a stream
	cusolverH = nullptr;
	stream = nullptr;
	gesvdj_params = nullptr;
	cusolverDnCreate(&cusolverH);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cusolverDnSetStream(cusolverH, stream);

	// configuration of gesvdj
	cusolverDnCreateGesvdjInfo(&gesvdj_params);

	// default value of tolerance is machine zero
	cusolverDnXgesvdjSetTolerance(gesvdj_params, tol);

	// default value of max. sweeps is 100
	cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps);

	// query workspace of svd
	cusolverDnSgesvdj_bufferSize(
		cusolverH,
		jobz,  // CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only
			   // CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors
		econ,  // econ = 1 for economy size
		m,     // nubmer of rows of A, 0 <= m
		n,     // number of columns of A, 0 <= n
		dev_A, // m-by-n
		lda,   // leading dimension of A
		dev_S, // min(m, n)
			   // the singular values in descending order
		dev_U, // m-by-m if econ = 0
			   // m-by-min(m,n) if econ = 1
		lda,   // leading dimension of U, ldu >= max(1,m)
		dev_V, // n-by-n if econ = 0
			   // n-by-min(m,n) if econ = 1
		n,     // leading dimension of V, ldv >= max(1,n)
		&lwork, gesvdj_params);
	//cusolverDnSgesvd_bufferSize(cusolverH, m, n, &lwork);

	CUDA_CHECK_RETURN(cudaMalloc((void**)&dev_work, lwork * sizeof(float)));
}

void cudaSVD::set_matrix(float *matrix, uint16_t num)
{
	host_A = matrix;
	n_ = num;
	lda = std::min(m_, n_);
}

void cudaSVD::core_compute()
{
	CUDA_CHECK_RETURN(cudaMemcpy(dev_A, host_A, m_ * n_ * sizeof(float), cudaMemcpyHostToDevice));

	cusolverDnSgesvdj(cusolverH, jobz, econ, m_, n_,
		dev_A, lda, dev_S, dev_U, lda, dev_V, n_,
		dev_work, lwork, dev_info, gesvdj_params);
	cudaDeviceSynchronize();

	CUDA_CHECK_RETURN(cudaMemcpy(host_U, dev_U, m_ * lda * sizeof(float), cudaMemcpyDeviceToHost));
	//CUDA_CHECK_RETURN(cudaMemcpy(host_V, dev_V, n_ * lda * sizeof(float), cudaMemcpyDeviceToHost));
	//CUDA_CHECK_RETURN(cudaMemcpy(host_S, dev_S, lda * sizeof(float), cudaMemcpyDeviceToHost));
	//CUDA_CHECK_RETURN(cudaMemcpy(&host_info, dev_info, sizeof(int), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	//if (0 == host_info) printf("gesvdj converges \n");
	//else if(0 > host_info) printf("%d-th parameter is wrong \n", -host_info);
	//else printf("WARNING: info = %d : gesvdj does not converge \n", host_info);
}
