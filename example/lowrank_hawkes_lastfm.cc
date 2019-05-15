//
// Created by Aleksandr Khvorov on 1/9/19.
//

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "LowRankHawkesProcess.h"

const double INF = 1e18;
const double SESSION_DIFF = .5;

double predict_user_next(LowRankHawkesProcess &model, std::vector <Sequence> &train_data, double window,
        int num_users, int user, std::unordered_map<int, std::vector<int>> &user_items) {
    double prediction = INF;
    for (int item: user_items[user]) {
        prediction = std::min(prediction, model.PredictNextEventTime(user, item, window, train_data));
    }
    return prediction;
}

std::unordered_map<unsigned, unsigned> build_dim_to_seq(std::vector<Sequence> &data) {
    std::unordered_map<unsigned, unsigned> dim_to_seq;
    for (Sequence sequence: data) {
        Event first_event = sequence.GetEvents()[0];
        dim_to_seq[first_event.DimentionID] = first_event.SequenceID;
    }
    return std::move(dim_to_seq);
}

class Predictor {
private:
    std::unordered_map<unsigned, double> predictions;
    LowRankHawkesProcess &model;
    unsigned num_users;
    std::vector<Sequence> train_data;
    std::unordered_map<int, std::vector<int>> &user_items;
    double observation_window;
    std::unordered_map<unsigned, unsigned> dim_to_seq;
public:
    Predictor(LowRankHawkesProcess &model, int num_users, std::vector<Sequence> train_data,
            std::unordered_map<int, std::vector<int>> &user_items, double observation_window):
            model(model), num_users(num_users), train_data(train_data), user_items(user_items),
            observation_window(observation_window) {
        for (auto it = user_items.begin(); it != user_items.end(); ++it) {
             int user = it->first;
             for (int item: it->second) {
                 unsigned dim_id = item * num_users + user;
                 predictions[dim_id] = model.PredictNextEventTime(user, item, observation_window, train_data);
             }
        }
        dim_to_seq = build_dim_to_seq(train_data);
    }

    void update(Event event) {
        int user = event.DimentionID % num_users;
        int item = event.DimentionID / num_users;
        unsigned sequence;
        auto dim_pos = dim_to_seq.find(event.DimentionID);
        if (dim_pos == dim_to_seq.end()) {
            sequence = train_data.size();
            train_data.push_back(Sequence());
            dim_to_seq[event.DimentionID] = sequence;
            user_items[user].push_back(item);
        } else {
            sequence = dim_pos->second;
        }
        train_data[sequence].Add(event);
        predictions[event.DimentionID] = model.PredictNextEventTime(user, item, observation_window, train_data);
    }

    double predict(unsigned user, double cur_time) {
        double prediction = INF;
        for (int item: user_items[user]) {
            unsigned dim_id = item * num_users + user;
            double item_prediction = predictions[dim_id];
            if (item_prediction >= cur_time) {
                prediction = std::min(prediction, item_prediction);
            }
        }
        return prediction;
    }
};

double mae(LowRankHawkesProcess &model, std::vector <Sequence> train_data,
           const std::vector <Sequence> &test_data, double observation_window, int num_users) {
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

double spu(LowRankHawkesProcess& model, std::vector<Sequence> train_data,
           const std::vector<Sequence>& test_data, double observation_window, int num_users) {
    double total_diff = 0.;
    long count = 0;
    for (const Sequence& sequence : test_data) {
        Event first_event = sequence.GetEvents()[0];
        int user_id = first_event.DimentionID % num_users;
        int item_id = first_event.DimentionID / num_users;
        double prev_time = -1;
        for (const Event& event : sequence.GetEvents()) {
            if (prev_time >= 0) {
                double predicted_time = model.PredictNextEventTime(user_id, item_id, observation_window, train_data);
                double delta_predicted = predicted_time - prev_time;
                double delta_real = event.time - prev_time;
                total_diff += abs(1 / delta_predicted - 1 / delta_real);
                ++count;
            }
            train_data[event.SequenceID].Add(event);
            prev_time = event.time;
        }
    }
    return total_diff / count;
}

double mae_user(LowRankHawkesProcess &model, std::unordered_map<int, Sequence> &user_sequences,
                std::vector <Sequence> train_data, std::unordered_map<int, std::vector<int>> &user_items,
                double observation_window, int num_users) {
    double error = 0.0;
    long count = 0;
    int step = 1;
    for (auto it = user_sequences.begin(); it != user_sequences.end(); ++it) {
        std::cout << "Step " << step++ << std::endl;
        int user_id = it->first;
        for (const Event& event : it->second.GetEvents()) {
            double predicted_time =
                    predict_user_next(model, train_data, observation_window, num_users, user_id, user_items);
            train_data[event.SequenceID].Add(event);
            error += abs(event.time - predicted_time);
            count++;
        }
    }
    return error / count;
}

double spu_user(LowRankHawkesProcess &model, std::unordered_map<int, Sequence> &user_sequences,
                std::vector <Sequence> train_data, std::unordered_map<int, std::vector<int>> &user_items,
                double observation_window, int num_users) {
    double total_diff = 0.;
    long count = 0;
    int step = 1;
    for (auto it = user_sequences.begin(); it != user_sequences.end(); ++it) {
        std::cout << "Step " << step++ << std::endl;
        int user_id = it->first;
        double prev_time = -1;
        for (const Event& event : it->second.GetEvents()) {
            if (prev_time >= 0) {
                double delta_predicted = predict_user_next(model, train_data, observation_window, num_users,
                                                           user_id, user_items) - prev_time;
                double delta_real = event.time - prev_time;
                total_diff += abs(1 / delta_real - 1 / delta_predicted);
                count++;
            }
            train_data[event.SequenceID].Add(event);
            prev_time = event.time;
        }
    }
    return total_diff / count;
}

void group_by_users(int num_users, std::vector<Sequence> &data, std::unordered_map<int, Sequence> &user_sequences,
                    std::unordered_map<int, std::vector<int>> &user_items) {
    std::unordered_map<int, std::vector<Event>> user_events;
    for (const Sequence& sequence : data) {
        const std::vector<Event> &seq_events = sequence.GetEvents();
        Event first_event = seq_events[0];
        int user_id = first_event.DimentionID % num_users;
        int item_id = first_event.DimentionID / num_users;
        if (user_items.find(user_id) == user_items.end()) {
            user_items.emplace(user_id, std::vector<int>());
            user_events.emplace(user_id, std::vector<Event>());
        }
        user_items[user_id].push_back(item_id);
        user_events[user_id].insert(user_events[user_id].end(), seq_events.begin(), seq_events.end());
    }
    for (auto it = user_events.begin(); it != user_events.end(); ++it) {
        std::sort(it->second.begin(), it->second.end(), [](Event a, Event b) { return a.time < b.time; });
        Sequence user_seq;
        for (Event event: it->second) {
            user_seq.Add(event);
        }
        user_sequences.emplace(it->first, user_seq);
    }
}

std::pair<double, double> user_metrics(LowRankHawkesProcess &model, std::vector <Sequence> &train_data,
        std::vector <Sequence> &test_data, double observation_window, int num_users) {
    std::unordered_map<int, Sequence> user_sequences;
    std::unordered_map<int, std::vector<int>> user_items;
    group_by_users(num_users, test_data, user_sequences, user_items);
    double total_spu_diff = 0., error = 0.;
    long count_spu = 0, count_mae = 0;
    int step = 1;
    Predictor predictor(model, num_users, train_data, user_items, observation_window);
    for (auto it = user_sequences.begin(); it != user_sequences.end(); ++it) {
        std::cout << "Processing user " << step++ << " of " << user_sequences.size() << std::endl;
        int user_id = it->first;
        double prev_time = -1;
        for (const Event& event : it->second.GetEvents()) {
            if (prev_time >= 0 && event.time - prev_time > SESSION_DIFF) {
                double predicted_time = predictor.predict(user_id, prev_time);
                error += abs(event.time - predicted_time);
                count_mae++;

                double delta_predicted = predicted_time - prev_time;
                double delta_real = event.time - prev_time;
                if (event.time != prev_time) {
                    total_spu_diff += abs(1 / delta_real - 1 / delta_predicted);
                    count_spu++;
                }
            }
            predictor.update(event);
            prev_time = event.time;
        }
    }
    return std::make_pair(error / count_mae, total_spu_diff / count_spu);
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
//    unsigned num_users = 4, num_items = 620;  // 100k (1000, 1000)
//    unsigned num_users = 41, num_items = 19089;  // 1M, MAE = 960
//    unsigned num_users = 41, num_items = 3000;  // 1M, MAE = 874 (top projects), MAE = 861 (rand projects)
//    unsigned num_users = 38, num_items = 3000;  // 1M, 811 (по-старому)
//    unsigned num_users = 38, num_items = 2424;  // 1M (1000, 3000),
//    unsigned num_users = 528, num_items = 76443;  // 10M
//    unsigned num_users = 992, num_items = 107296;  // all
//    unsigned num_users = 992, num_items = 3000;  // all MAE = 581.221 (top projects), MAE = 728 (rand projects)
    unsigned num_users = 193, num_items = 1000;
//    std::string FILE_SIZE = "_all";
    std::string FILE_SIZE = "5000000";
    std::string USERS_NUM = "1000";
    std::string ITEMS_NUM = "1000";
    std::string DATA_FORMAT = "lastfm";
    std::string FILTRATION_TYPE = "top";
    std::string FOLDER = "data/";
    std::string FILENAME_PREFIX = FOLDER + DATA_FORMAT + "/"
            + DATA_FORMAT + "_" + FILTRATION_TYPE + "_" + FILE_SIZE + "_" + USERS_NUM + "_" + ITEMS_NUM;
    std::string TRAIN_FILENAME = FILENAME_PREFIX + "_0.75_train";
    std::string TEST_FILENAME = FILENAME_PREFIX + "_0.75_test";

    std::vector<Sequence> train_data, test_data;
    std::cout << "Using path:" << FILENAME_PREFIX << std::endl;
    std::cout << "1. Loading " << num_users << " users " << num_items << " items" << std::endl;

    ImportFromExistingUserItemSequences(TRAIN_FILENAME, num_users, num_items, train_data);
    ImportFromExistingUserItemSequences(TEST_FILENAME, num_users, num_items, test_data);

    unsigned dim = num_users * num_items;
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(dim, 1);
    LowRankHawkesProcess low_rank_hawkes(num_users, num_items, beta);
    LowRankHawkesProcess::OPTION options;
    options.coefficients[LowRankHawkesProcess::LAMBDA0] = 1;
    options.coefficients[LowRankHawkesProcess::LAMBDA] = 1;
    options.ini_learning_rate = 1e-3;  // 2e-5 for 100k, 8e-5 for 1M
    options.ub_nuclear_lambda0 = 25;
    options.ub_nuclear_alpha = 25;
    options.rho = 1e1;
    options.ini_max_iter = 300;
    std::cout << "2. Fitting Parameters " << std::endl;
    low_rank_hawkes.fit(train_data, options);

    std::cout << "Fitted. Start testing" << std::endl;
    double observation_window = 2000;
    auto metrics = user_metrics(low_rank_hawkes, train_data, test_data, observation_window, num_users);
    std::cout << "Test return time mae: " << metrics.first << std::endl;
//        mae(low_rank_hawkes, train_data, test_data, observation_window, num_users) << std::endl;
//        mae_user(low_rank_hawkes, user_sequences, train_data, user_items, observation_window, num_users) << std::endl;
    std::cout << "Test SPU: " << metrics.second << std::endl;
//        spu(low_rank_hawkes, train_data, test_data, observation_window, num_users) << std::endl;
//        spu_user(low_rank_hawkes, user_sequences, train_data, user_items, observation_window, num_users) << std::endl;

    return 0;
}