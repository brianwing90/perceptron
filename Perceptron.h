/*
 * Created by brianwing90@gmail.com on 4/6/2018.
 */

#ifndef PERCEPTRON_PERCEPTRON_H
#define PERCEPTRON_PERCEPTRON_H

#include <vector>
#include <cmath>

using namespace std;

class Perceptron {
public:
    const double INITIAL_WEIGHT = 0.0; // This is the starting value of the weights for each input.
    explicit Perceptron(int number_of_inputs);
    ~Perceptron();
    void train(vector<vector<double>> training_data, vector<int> y_actual); // This trains the weights of the perceptron to properly classify inputs.
    vector<bool> test(vector<vector<double>> testing_data, vector<int> y_actual); // This tests the perceptron on the given input data and scores itself based on the percentage of correct classifications.
    double get_classification_rate();
    vector<double> get_weights();
    double get_learning_rate();
    void set_learning_rate(double rate);

private:
    int num_inputs; // The number of inputs and weights for this perceptron.
    vector<double> weights; // The weights assigned to the inputs.
    double learning_rate; // How aggressively to change the weights of the inputs.
    double classification_rate; // How accurate the perceptron is at any given time.
    int activation_function(double value); // A simple activation function that returns 1 if the value to activate is >= 0 and -1 otherwise.
};


#endif //PERCEPTRON_PERCEPTRON_H