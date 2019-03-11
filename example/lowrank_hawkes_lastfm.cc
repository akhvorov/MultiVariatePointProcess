//
// Created by Aleksandr Khvorov on 1/9/19.
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

//double unseen_rec(LowRankHawkesProcess& model, std::vector<Sequence> train_data,
//                       const std::vector<Sequence>& test_data, double observation_window, int num_users) {
//    std::unordered_map<int, std::unordered_set<int>> all_unseen_projects;
//    std::unordered_map<int, std::unordered_set<int>> will_seen_projects;
//    for (const Sequence& sequence : test_data) {
//        Event first_event = sequence.GetEvents()[0];
//        int user_id = first_event.DimentionID % num_users;
//        int item_id = first_event.DimentionID / num_users;
//        will_seen_projects[user_id].insert(item_id);
//    }
//    for (const Sequence& sequence : train_data) {
//        Event first_event = sequence.GetEvents()[0];
//        int user_id = first_event.DimentionID % num_users;
//        int item_id = first_event.DimentionID / num_users;
//        will_seen_projects[user_id].insert(item_id);
//    }
//    double error = 0.0;
//    int count = 0;
//    for (const Sequence& sequence : test_data) {
//        Event first_event = sequence.GetEvents()[0];
//        int user_id = first_event.DimentionID % num_users;
//        int item_id = first_event.DimentionID / num_users;
//        for (const Event& event : sequence.GetEvents()) {
//            double predicted_time = model.PredictNextEventTime(user_id, item_id, observation_window, train_data);
//            train_data[event.SequenceID].Add(event);
//
//            error += abs(event.time - predicted_time);
//            count++;
//        }
//    }
//    return error / count;
//}

int main(const int argc, const char** argv) {
//    unsigned num_users = 1, num_items = 514;
//    unsigned num_users = 4, num_items = 3083;  // 100k, MAE = 1517
//    unsigned num_users = 41, num_items = 19089;  // 1M, MAE = 960
//    unsigned num_users = 41, num_items = 3000;  // 1M, MAE = 874 (top projects), MAE = 861 (rand projects)
    unsigned num_users = 38, num_items = 3000;  // 1M, 811 (по-старому)
//    unsigned num_users = 528, num_items = 76443;  // 10M
//    unsigned num_users = 992, num_items = 107296;  // all
//    unsigned num_users = 992, num_items = 3000;  // all MAE = 581.221 (top projects), MAE = 728 (rand projects)
    std::string FILE_SIZE = "1M";
    std::vector<Sequence> train_data, test_data;
    std::cout << "1. Loading " << num_users << " users " << num_items << " items" << std::endl;
    ImportFromExistingUserItemSequences("data/lastfm/lastfm_top_" + FILE_SIZE + "_1k_3k_0.75_train", num_users, num_items, train_data);
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
    options.ini_max_iter = 30;
    std::cout << "2. Fitting Parameters " << std::endl;
    low_rank_hawkes.fit(train_data, options);

    ImportFromExistingUserItemSequences("data/lastfm/lastfm_top_" + FILE_SIZE + "_1k_3k_0.75_test", num_users, num_items, test_data);

    std::cout << "Fitted. Start testing" << std::endl;
    double observation_window = 2000;
    std::cout << "Test return time mae " << return_time_mae(low_rank_hawkes, train_data, test_data, observation_window, num_users) << std::endl;

//    unsigned test_userID = 0;
//    double t = 100;
//    std::cout << "3. Predicted Item for User " << test_userID <<": " << low_rank_hawkes.PredictNextItem(test_userID, t, data) << std::endl;
//
//    test_userID = 24;
//    unsigned test_itemID = 6;
//    double observation_window = 2000;
//    std::cout << "4. Predicted next event for user " << test_userID << " and item " << test_itemID <<": " << low_rank_hawkes.PredictNextEventTime(test_userID, test_itemID, observation_window, data) << std::endl;


    return 0;
}