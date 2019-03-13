#include <iostream>
#include <Eigen/Dense>
#include "Sequence.h"
#include "OgataThinning.h"
#include "LowRankHawkesProcess.h"

const double OBSERVATION_WINDOW = 10000;
const unsigned NUM_USERS = 1;
const unsigned NUM_ITEMS = 1;
const unsigned DIM = NUM_USERS * NUM_ITEMS;

double return_time_mae(LowRankHawkesProcess& model, std::vector<Sequence> train_data,
                       const std::vector<Sequence>& test_data, double observation_window, int NUM_USERS) {
    double error = 0.0;
    int count = 0;
    for (const Sequence& sequence : test_data) {
        Event first_event = sequence.GetEvents()[0];
        int user_id = first_event.DimentionID % NUM_USERS;
        int item_id = first_event.DimentionID / NUM_USERS;
        for (const Event& event : sequence.GetEvents()) {
            double predicted_time = model.PredictNextEventTime(user_id, item_id, observation_window, train_data);
            auto events = train_data[0].GetEvents();
            std::cout << events[events.size() - 1].time << " " << predicted_time << ' ' << event.time << '\n';
            train_data[event.SequenceID].Add(event);
            error += abs(event.time - predicted_time);
            count++;
        }
    }
    return error / count;
}

Event make_event(int id, int sequence_id, double time) {
    Event event;
    event.EventID = id;
    event.SequenceID = sequence_id;
    event.DimentionID = sequence_id;
    event.time = time;
    event.marker = -1;
    return event;
}

std::vector<Sequence> make_seq(double start, double step, int n) {
    Sequence seq;
    for (int i = 0; i < n; ++i) {
        seq.Add(make_event(0, 0, start + step * i));
//        std::cout << "add event " << start + step * i << std::endl;
    }
    return std::move(std::vector<Sequence> { seq });
}

void show_seq(Sequence &seq) {
    int N = 10;
    for (int i = 0; i < N; ++i) {
        Event e = seq.GetEvents()[i];
        std::cout << e.EventID << ' ' << e.SequenceID << ' ' << e.DimentionID << ' '
                << e.time << ' ' << e.marker << std::endl;
    }
}

void test_short() {
    int diff = 5;
    auto data_train = make_seq(0, diff, 2);
    auto data_test = make_seq(2 * diff, diff, 2);
    Eigen::VectorXd beta = Eigen::VectorXd::Constant(DIM, 1e-2);
    LowRankHawkesProcess low_rank_hawkes(NUM_USERS, NUM_ITEMS, beta);
    LowRankHawkesProcess::OPTION options;
    options.coefficients[LowRankHawkesProcess::LAMBDA0] = 1;
    options.coefficients[LowRankHawkesProcess::LAMBDA] = 1;
    options.ini_learning_rate = 7e-10;  // 2e-5 for 100k, 8e-5 for 1M
    options.ub_nuclear_lambda0 = 2;
    options.ub_nuclear_alpha = 2;
    options.rho = 1e1;
    options.ini_max_iter = 100;
    low_rank_hawkes.fit(data_train, options);

    std::cout << return_time_mae(low_rank_hawkes, data_train, data_test, OBSERVATION_WINDOW, NUM_USERS) << std::endl;
}

void test_long() {
    double diff = 250;
    int train_size = 3000;
    int test_size = 40;
    auto data_train = make_seq(0, diff, train_size);
    auto data_test = make_seq(train_size * diff, diff, test_size);

    Eigen::VectorXd beta = Eigen::VectorXd::Constant(DIM, 1000);
    LowRankHawkesProcess low_rank_hawkes(NUM_USERS, NUM_ITEMS, beta);
    LowRankHawkesProcess::OPTION options;
    options.coefficients[LowRankHawkesProcess::LAMBDA0] = 1e-3;
    options.coefficients[LowRankHawkesProcess::LAMBDA] = 1e-3;
    options.ini_learning_rate = 1e-6;  // 2e-5 for 100k, 8e-5 for 1M
    options.ub_nuclear_lambda0 = 20000;
    options.ub_nuclear_alpha = 20000;
    options.rho = 1e1;
    options.ini_max_iter = 100000;
    low_rank_hawkes.fit(data_train, options);
    std::cout << low_rank_hawkes.GetParameters();

    std::cout << return_time_mae(low_rank_hawkes, data_train, data_test, OBSERVATION_WINDOW, NUM_USERS) << std::endl;
//    std::cout << low_rank_hawkes.GetParameters() << std::endl;
}

int main(const int argc, const char** argv)
{
//    test_short();
    test_long();

    return 0;
}
