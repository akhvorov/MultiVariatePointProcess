//
// Created by Aleksandr Khvorov on 2/7/19.
//

#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "LowRankHawkesProcess.h"

double return_time_mae(LowRankHawkesProcess& model, std::vector<Sequence> train_data,
                       const std::vector<Sequence>& test_data, double observation_window, int num_users) {
    double error = 0.0;
    int count = 0;
    for (const Sequence& sequence : test_data) {
        Event first_event = sequence.GetEvents()[0];
        int user_id = first_event.DimentionID % num_users;
        int item_id = first_event.DimentionID / num_users;
        for (const Event& event : sequence.GetEvents()) {
            double predicted_time = model.PredictNextEventTime(user_id, item_id, observation_window, train_data);
            train_data[event.SequenceID].Add(event);

            error += abs(event.time - predicted_time);
            count++;
        }
    }
    return error / count;
}

int main(const int argc, const char** argv)
{
    unsigned num_users = 230, num_items = 141; // 2 days, 8
//    unsigned num_users = 143, num_items = 231; // 30 days, 65
    std::string DATE = "11_01";
    std::vector<Sequence> train_data, test_data;
    std::cout << "1. Loading " << num_users << " users " << num_items << " items" << std::endl;
    ImportFromExistingUserItemSequences("data/toloka/toloka_" + DATE + "_1k_3k_0.75_train", num_users, num_items, train_data);
    unsigned dim = num_users * num_items;
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(dim, 1.0);
    LowRankHawkesProcess low_rank_hawkes(num_users, num_items, beta);
    LowRankHawkesProcess::OPTION options;
    options.coefficients[LowRankHawkesProcess::LAMBDA0] = 1;
    options.coefficients[LowRankHawkesProcess::LAMBDA] = 1;
    options.ini_learning_rate = 5e-3;  // 2e-5 for 100k, 8e-5 for 1M
    options.ub_nuclear_lambda0 = 25;
    options.ub_nuclear_alpha = 25;
    options.rho = 1e1;
    options.ini_max_iter = 9;
    std::cout << "2. Fitting Parameters " << std::endl;
    low_rank_hawkes.fit(train_data, options);

    ImportFromExistingUserItemSequences("data/toloka/toloka_" + DATE + "_1k_3k_0.75_test", num_users, num_items, test_data);

    std::cout << "Fitted. Start testing" << std::endl;
    double observation_window = 2000;
    std::cout << return_time_mae(low_rank_hawkes, train_data, test_data, observation_window, num_users) << std::endl;

    return 0;
}