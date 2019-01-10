//
// Created by Aleksandr Khvorov on 1/9/19.
//

#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "LowRankHawkesProcess.h"

int main(const int argc, const char** argv)
{
    unsigned num_users = 4, num_items = 3083;
//    unsigned num_users = 41, num_items = 19089;
    std::vector<Sequence> data;
    std::cout << "1. Loading " << num_users << " users " << num_items << " items" << std::endl;
    ImportFromExistingUserItemSequences("data/lastfm_100k", num_users, num_items, data);
    unsigned dim = num_users * num_items;
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(dim, 1.0);
    LowRankHawkesProcess low_rank_hawkes(num_users, num_items, beta);
    LowRankHawkesProcess::OPTION options;
    options.coefficients[LowRankHawkesProcess::LAMBDA0] = 1;
    options.coefficients[LowRankHawkesProcess::LAMBDA] = 1;
    options.ini_learning_rate = 2e-5;
    options.ub_nuclear_lambda0 = 25;
    options.ub_nuclear_alpha = 25;
    options.rho = 1e1;
    options.ini_max_iter = 300;
    std::cout << "2. Fitting Parameters " << std::endl;
    low_rank_hawkes.fit(data, options);

    unsigned test_userID = 0;
    double t = 100;
    std::cout << "3. Predicted Item for User " << test_userID <<": " << low_rank_hawkes.PredictNextItem(test_userID, t, data) << std::endl;

    test_userID = 24;
    unsigned test_itemID = 6;
    double observation_window = 2000;
    std::cout << "4. Predicted next event for user " << test_userID <<" and item " << test_itemID <<": " << low_rank_hawkes.PredictNextEventTime(test_userID, test_itemID, observation_window, data) << std::endl;


    return 0;
}