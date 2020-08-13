#include "plane_linear.h"

PlaneLinear::PlaneLinear(uint16_t m, bool mode)
{
	mode_ = mode;
	if(mode_)
	{
		cuda_svd = new cudaSVD();
		cuda_svd->malloc_space(DATA_AXIS, m);
	}
	else
	{

	}
	
}

PlaneLinear::~PlaneLinear()
{
	if(mode_)
	{
		delete cuda_svd;
		cuda_svd = nullptr;
	}
}

bool PlaneLinear::plane_fit(float *points, uint16_t num, float *params)
{
	uint16_t operate_m;
	if(num > m_) operate_m = m_;
	else operate_m = num;
	Eigen::MatrixXf operate_data(DATA_AXIS, operate_m);
	Eigen::Vector3f normal;
	for(uint16_t i = 0; i < operate_m; i ++)
	{
		operate_data(0, i) = points[i * DATA_AXIS];
		operate_data(1, i) = points[i * DATA_AXIS + 1];
		operate_data(2, i) = points[i * DATA_AXIS + 2];
	}
	Eigen::Vector3f centroid = operate_data.rowwise().mean(); 
	operate_data.colwise() -= centroid; 

	if(mode_)
	{
		cuda_svd->set_matrix(operate_data.data(), operate_m);
		cuda_svd->core_compute();
		float *host_U = cuda_svd->get_host_u();
		params[0] = normal(0) = host_U[6];
		params[1] = normal(1) = host_U[7];
		params[2] = normal(2) = host_U[8];
		params[3] = -normal.dot(centroid);
	}
	else
	{
		Eigen::JacobiSVD<Eigen::MatrixXf> svd(operate_data, Eigen::DecompositionOptions::ComputeFullU);
		normal = svd.matrixU().col(2); 
		params[0] = normal(0);
		params[1] = normal(1);
		params[2] = normal(2);
		params[3] = -normal.dot(centroid);
	}
	
	return true;
}