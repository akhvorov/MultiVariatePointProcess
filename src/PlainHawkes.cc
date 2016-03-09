#include <vector>
#include <cmath>
#include <iostream>
#include "../include/PlainHawkes.h"
#include "../include/Sequence.h"
#include "../include/Optimizer.h"
#include "../include/OgataThinning.h"

void PlainHawkes::Initialize(const std::vector<Sequence>& data)
{
	num_sequences_ = data.size();

	all_exp_kernel_recursive_sum_ = std::vector<std::vector<std::vector<Eigen::VectorXd> > >(num_sequences_, std::vector<std::vector<Eigen::VectorXd> >(
          num_dims_, std::vector<Eigen::VectorXd>(num_dims_, Eigen::VectorXd())));

	// all_timestamp_per_dimension_ = std::vector<std::vector<std::vector<double> > >(num_sequences_, std::vector<std::vector<double> > (num_dims_, std::vector<double> ()));
	// for(unsigned c = 0; c < num_sequences_; ++ c)
	// {
	// 	const std::vector<Event>& seq = data[c].GetEvents();

	// 	for(unsigned i = 0; i < seq.size(); ++ i)
	// 	{
	// 		all_timestamp_per_dimension_[c][seq[i].DimentionID].push_back(seq[i].time);
	// 	}

	// }

	InitializeDimension(data);

	for (unsigned k = 0; k < num_sequences_; ++k) 
	{
	    for (unsigned m = 0; m < num_dims_; ++m) 
	    {
	      for (unsigned n = 0; n < num_dims_; ++n) 
	      {

	      	all_exp_kernel_recursive_sum_[k][m][n] = Eigen::VectorXd::Zero(all_timestamp_per_dimension_[k][n].size());

	        if (m != n) {

	          for (unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size(); ++i) 
	          {

	            double value = exp(-Beta_(m,n) * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][n][i - 1])) * all_exp_kernel_recursive_sum_[k][m][n](i - 1);

	            for (unsigned j = 0; j < all_timestamp_per_dimension_[k][m].size(); ++j) {
	              if ((all_timestamp_per_dimension_[k][n][i - 1] <= all_timestamp_per_dimension_[k][m][j]) &&
	                  (all_timestamp_per_dimension_[k][m][j] < all_timestamp_per_dimension_[k][n][i])) 
	              {
	                value += exp(-Beta_(m,n) * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][m][j]));
	              }
	            }

	            all_exp_kernel_recursive_sum_[k][m][n](i) = value;
	          }
	        } else 
	        {
	          for (unsigned i = 1; i < all_timestamp_per_dimension_[k][n].size(); ++i) 
	          {
	            all_exp_kernel_recursive_sum_[k][m][n](i) = exp(-Beta_(m,n) * (all_timestamp_per_dimension_[k][n][i] - all_timestamp_per_dimension_[k][n][i - 1])) * (1 + all_exp_kernel_recursive_sum_[k][m][n](i - 1));
	          }
	        }
	      }
	    }
  	}

	observation_window_T_ = Eigen::VectorXd::Zero(num_sequences_);

  	intensity_itegral_features_ = std::vector<Eigen::MatrixXd> (num_sequences_, Eigen::MatrixXd::Zero(num_dims_, num_dims_));

  	for (unsigned c = 0; c < num_sequences_; ++c) {

  		observation_window_T_(c) = data[c].GetTimeWindow();

	    for (unsigned m = 0; m < num_dims_; ++ m) {

	      for (unsigned n = 0; n < num_dims_; ++ n) {

	      	Eigen::Map<Eigen::VectorXd> event_dim_m = Eigen::Map<Eigen::VectorXd>(all_timestamp_per_dimension_[c][m].data(), all_timestamp_per_dimension_[c][m].size());

	      	intensity_itegral_features_[c](m,n) = (1 - (-Beta_(m,n) * (observation_window_T_(c) - event_dim_m.array())).exp()).sum();

	      }
	  	}
	}

}


double PlainHawkes::Intensity(const double& t, const Sequence& data, Eigen::VectorXd& intensity_dim)
{

	intensity_dim = Eigen::VectorXd::Zero(num_dims_);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	intensity_dim = Lambda0_;

	const std::vector<Event>& seq = data.GetEvents();

	for(unsigned i = 0; i < seq.size(); ++ i)
	{
		if (seq[i].time < t)
		{
			for(unsigned d = 0; d < num_dims_; ++ d)
			{
				intensity_dim(d) += Alpha_(seq[i].DimentionID, d) * exp(-Beta_(seq[i].DimentionID, d) * (t - seq[i].time));
			}	
		}
		else
		{
			break;
		}
	}

	return intensity_dim.array().sum();
	
}

double PlainHawkes::IntensityUpperBound(const double& t, const Sequence& data, Eigen::VectorXd& intensity_upper_dim)
{

	intensity_upper_dim = Eigen::VectorXd::Zero(num_dims_);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	intensity_upper_dim = Lambda0_;

	const std::vector<Event>& seq = data.GetEvents();

	for(unsigned i = 0; i < seq.size(); ++ i)
	{
		if (seq[i].time <= t)
		{
			for(unsigned d = 0; d < num_dims_; ++ d)
			{
				intensity_upper_dim(d) += Alpha_(seq[i].DimentionID, d) * exp(-Beta_(seq[i].DimentionID, d) * (t - seq[i].time));
			}	
		}
		else
		{
			break;
		}
	}

	return intensity_upper_dim.array().sum();
}

void PlainHawkes::NegLoglikelihood(double& objvalue, Eigen::VectorXd& gradient)
{

	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	gradient = Eigen::VectorXd::Zero(num_dims_ * (1 + num_dims_));

	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> grad_alpha_matrix = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	// first num_of_dimensions of the parameters are the base intensity
	// the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	objvalue = 0;

	for (unsigned k = 0; k < num_sequences_; ++k) 
	{
	    const std::vector<std::vector<double> > &timestamp_per_dimension = all_timestamp_per_dimension_[k];

	    const std::vector<std::vector<Eigen::VectorXd> > &exp_kernel_recursive_sum = all_exp_kernel_recursive_sum_[k];

	    for (unsigned n = 0; n < num_dims_; ++n) 
	    {

	      double obj_n = 0;

	      for (unsigned i = 0; i < timestamp_per_dimension[n].size(); ++i) 
	      {
	        double local_sum = Lambda0_(n);
	        
	        for (unsigned m = 0; m < num_dims_; ++m) 
	        {
	          local_sum += Alpha_(m,n) * exp_kernel_recursive_sum[m][n](i);
	        }

	        obj_n += log(local_sum);

	        grad_lambda0_vector(n) += (1 / local_sum);

	        for (unsigned m = 0; m < num_dims_; ++m) 
	        {
	          grad_alpha_matrix(m, n) += exp_kernel_recursive_sum[m][n](i) / local_sum;
	        }
	      }

	      obj_n -= ((Alpha_.col(n).array() / Beta_.col(n).array()) * intensity_itegral_features_[k].col(n).array()).sum();

	      grad_alpha_matrix.col(n) = grad_alpha_matrix.col(n).array() - (intensity_itegral_features_[k].col(n).array() / Beta_.col(n).array());

	      obj_n -= observation_window_T_(k) * Lambda0_(n);

	      grad_lambda0_vector(n) -= observation_window_T_(k);

	      objvalue += obj_n;
	    }
  	}

  	gradient = -gradient.array() / num_sequences_;

	objvalue = -objvalue / num_sequences_;

}

void PlainHawkes::Gradient(const unsigned &k, Eigen::VectorXd& gradient)
{
	if(all_timestamp_per_dimension_.size() == 0)
	{
		std::cout << "Process is uninitialzed with any data." << std::endl;
		return;
	}

	// // first num_of_dimensions of the parameters are the base intensity
	// // the rest num_of_dimensions * num_of_dimensions constitute the alpha matrix

	gradient = Eigen::VectorXd::Zero(num_dims_ * (1 + num_dims_));

	Eigen::Map<Eigen::VectorXd> grad_lambda0_vector = Eigen::Map<Eigen::VectorXd>(gradient.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> grad_alpha_matrix = Eigen::Map<Eigen::MatrixXd>(gradient.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	const std::vector<std::vector<double> > &timestamp_per_dimension = all_timestamp_per_dimension_[k];

	const std::vector<std::vector<Eigen::VectorXd> > &exp_kernel_recursive_sum = all_exp_kernel_recursive_sum_[k];

	for (unsigned n = 0; n < num_dims_; ++n) 
    {
      for (unsigned i = 0; i < timestamp_per_dimension[n].size(); ++i) 
      {
        double local_sum = Lambda0_(n);
        
        for (unsigned m = 0; m < num_dims_; ++m) 
        {
          local_sum += Alpha_(m,n) * exp_kernel_recursive_sum[m][n](i);
        }

        grad_lambda0_vector(n) += (1 / local_sum);

        for (unsigned m = 0; m < num_dims_; ++m) 
        {
          grad_alpha_matrix(m, n) += exp_kernel_recursive_sum[m][n](i) / local_sum;
        }
      }

      grad_alpha_matrix.col(n) = grad_alpha_matrix.col(n).array() - (intensity_itegral_features_[k].col(n).array() / Beta_.col(n).array());

      grad_lambda0_vector(n) -= observation_window_T_(k);

    }    

    gradient = -gradient.array() / num_sequences_;
}

void PlainHawkes::fit(const std::vector<Sequence>& data, const std::string& method)
{
	PlainHawkes::Initialize(data);

	Optimizer opt(this);

	if(method == "SGD")
	{	
		opt.SGD(1e-5, 5000, data);
		return;
	}

	if(method == "LBFGS")
	{	
		opt.PLBFGS(0, 1e10);
		return;
	}	

}

double PlainHawkes::PredictNextEventTime(const Sequence& data, const unsigned& num_simulations)
{
	OgataThinning ot(num_dims_);
	double t = 0;
	for(unsigned i = 0; i < num_simulations; ++ i)
	{
		Event event = ot.SimulateNext(*this, data);
		t += event.time;
	}
	return t / num_simulations;
}

double PlainHawkes::IntensityIntegral(const double& lower, const double& upper, const Sequence& data)
{
	std::vector<Sequence> sequences;
	sequences.push_back(data);

	InitializeDimension(sequences);

	Eigen::Map<Eigen::VectorXd> Lambda0_ = Eigen::Map<Eigen::VectorXd>(parameters_.segment(0, num_dims_).data(), num_dims_);

	Eigen::Map<Eigen::MatrixXd> Alpha_ = Eigen::Map<Eigen::MatrixXd>(parameters_.segment(num_dims_, num_dims_ * num_dims_).data(), num_dims_, num_dims_);

	std::vector<std::vector<double> >& timestamp_per_dimension = all_timestamp_per_dimension_[0];

	double integral_value = 0;

	for(unsigned n = 0; n < num_dims_; ++ n)
	{
		integral_value = Lambda0_(n) * (upper - lower);

		for(unsigned m = 0; m < num_dims_; ++ m)
		{
			Eigen::Map<Eigen::VectorXd> event_dim_m = Eigen::Map<Eigen::VectorXd>(timestamp_per_dimension[m].data(), timestamp_per_dimension[m].size());

			Eigen::VectorXd mask = (event_dim_m.array() < lower).cast<double>();
			double a = (mask.array() * (((-Beta_(m,n) * (lower - event_dim_m.array())) * mask.array()).exp() - ((-Beta_(m,n) * (upper - event_dim_m.array())) * mask.array()).exp())).sum();

			mask = (event_dim_m.array() >= lower && event_dim_m.array() < upper).cast<double>();
			double b = (mask.array() * (1 - ((-Beta_(m,n) * (upper - event_dim_m.array())) * mask.array()).exp())).sum();

			integral_value += (Alpha_(m,n) / Beta_(m,n)) * (a + b);

		}
	}


	return integral_value;
}