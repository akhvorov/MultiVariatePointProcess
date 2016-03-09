#ifndef PLAIN_HAWKES_H
#define PLAIN_HAWKES_H
#include <vector>
#include <string>
#include "Process.h"


/*
	
	This class defines the Hawkes process which implements the general process internface IPorcess.
	 
*/

class PlainHawkes : public IProcess
{

protected:

	Eigen::MatrixXd Beta_;

//  This variable is process-specific. It stores the temporal features associated with the intensity function of the multivariate hawkes process.
//  for each sequence c, given a pair of dimension n and m, and a point t^n{c,i} on the dimension n, all_exp_kernel_recursive_sum_[c][m][n][i] stores the exponential sum of all the past events t^m_{c,j} < t^n_{c,i} on the dimension m in the sequence c
	std::vector<std::vector<std::vector<Eigen::VectorXd> > > all_exp_kernel_recursive_sum_;

//	This variable is process-specific. It stores the temporal features associated with the integral intensity function of the multivariate hawkes process. intensity_itegral_features_[c][m][n] stores the summation \sum_{t^m_{c,i} < T_c} (1 - exp(-\beta^mn(T_c - t^m_{c,i})))
	std::vector<Eigen::MatrixXd> intensity_itegral_features_;

	Eigen::VectorXd observation_window_T_;

//  This variable is process-specific. It records totoal number of sequences used to fit the process.
	unsigned num_sequences_;

//  This function requires process-specific implementation. It initializes the temporal features used to calculate the negative loglikelihood and the gradient. 
	void Initialize(const std::vector<Sequence>& data);


public:

//  Constructor : n is the number of parameters in total; num_dims is the number of dimensions in the process;
	PlainHawkes(const unsigned& n, const unsigned& num_dims, const Eigen::MatrixXd& Beta) : IProcess(n, num_dims), Beta_(Beta), num_sequences_(0) {}

//  MLE esitmation of the parameters
	void fit(const std::vector<Sequence>& data, const std::string& opt);

//  This virtual function requires process-specific implementation. It calculates the negative loglikelihood of the given data. This function must be called after the Initialize method to return the negative loglikelihood of the data with respect to the current parameters. 
//	The returned negative loglikelihood is stored in the variable objvalue;
//	The returned gradient vector wrt the current parameters is stored in the variable Gradient; 
	virtual void NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient);

	//  Return the stochastic gradient on the random sample k.
	virtual void Gradient(const unsigned &k, Eigen::VectorXd& gradient);

//  This virtual function requires process-specific implementation. It returns the intensity value on each dimension in the variable intensity_dim for the given sequence stored in data and the given time t;
//  This function returns the summation of the individual intensity function on all dimensions. 
	virtual double Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim);

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the summation of the individual intensity upper bound on all dimensions. 
	virtual double IntensityUpperBound(const double& t, const Sequence& data, Eigen::VectorXd& intensity_upper_dim);

//  This virtual function requires process-specific implementation. It returns the upper bound of the intensity function on each dimension at time t given the history data in the variable intensity_upper_dim;
//	This function returns the integral of the intensity from a to b
	virtual double IntensityIntegral(const double& lower, const double& upper, const Sequence& data);

//  This function predicts the next event by simulation;
	virtual double PredictNextEventTime(const Sequence& data, const unsigned& num_simulations);

};

#endif