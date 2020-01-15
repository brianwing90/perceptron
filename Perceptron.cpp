/*
 * Created by brianwing90@gmail.com on 4/6/2018.
 */

#include <cstdio>
#include "Perceptron.h"

/**
 * A single perceptron representation for use in a layered network or as a single classifier. Input and output data
 * can be given to the train() method for training the weights of the input values or to the test() method for using
 * the perceptron like a classifier.
 * @param number_of_inputs The number of inputs this perceptron can expect. This influences the number of weights the
 * perceptron keeps track of.
 */
Perceptron::Perceptron(int number_of_inputs) {
    num_inputs = number_of_inputs;
    learning_rate = 0.01; // Default learning rate.
    for(int i = 0; i < num_inputs; i++){
        weights.push_back(INITIAL_WEIGHT); // Initialize the input weights.
    }
}

Perceptron::~Perceptron() {}

/**
 * Online training of the weights of this perceptron using the given data set and results.
 * @param training_data The data set to train the perceptron on.
 * @param y_actual The correct results the perceptron is trying to achieve.
 */
void Perceptron::train(vector<vector<double>> training_data, vector<int> y_actual) {
    for(int i = 0; i < training_data.size(); i++){
        int epoch = 0; // This keeps track of how many times the weights have been trained on this one observation.
        int max_epochs = 50; // This is the maximum number of times the weights will train on each observation.
        bool next_obs = false; // This is true when it is time to move to the next observation in the data set.

        while(!next_obs) {
            vector<double> inputs = training_data[i]; // This observation's features.
            double sum = 0.0;
            for (int j = 0; j < inputs.size(); j++) {
                sum += inputs[j] * weights[j]; // This is the jth feature of the ith input times the jth weight.
            }

            // Calculate the predicted value for y based on the activation function. Here the activation function is simply 1 if sum >= 0 and -1 otherwise.
            int y_predicted = activation_function(sum);

            if (y_predicted != y_actual[i]) {
                //printf("The predicted y and actual y were not equal!\n"); // This shows when the weights had to be readjusted to prove that the perceptron really is learning.

                // Redistribute the weights of the inputs.
                for (int j = 0; j < weights.size(); j++) {
                    weights[j] = weights[j] + learning_rate * (y_actual[i] - y_predicted) * inputs[j]; // The new weight is the actual output minus the predicted output times the input plus the old weight.
                }
            } else {
                next_obs = true;
            }

            if (epoch >= max_epochs) {
                next_obs = true;
            }
        }
    }
}

/**
 * A test for the perceptron to see if it classifies data correctly. Afterward the classification rate can be seen with get_classification_rate().
 * @param testing_data The inputs to test the perceptron on.
 * @param y_actual The expected outputs of the perceptron.
 * @return A vector of booleans where true indicates that the predicted and actual outputs were equal and false that they were not.
 */
vector<bool> Perceptron::test(vector<vector<double>> testing_data, vector<int> y_actual) {
    vector<bool> responses;
    double error_sum = 0.0;
    for(int i = 0; i < testing_data.size(); i++){
        vector<double> inputs = testing_data[i]; // This observation's features.
        double sum = 0.0;
        for (int j = 0; j < inputs.size(); j++) {
            sum += inputs[j] * weights[j]; // This is the jth feature of the ith input times the jth weight.
        }

        // Calculate the predicted value for y based on the activation function. Here the activation function is simply 1 if sum >= 0 and -1 otherwise.
        int y_predicted = activation_function(sum);

        if (y_predicted == y_actual[i]) {
            // Prediction was successful!
            responses.push_back(true);
        }else{
            // Prediction failed.
            error_sum += abs(y_actual[i] - y_predicted);
            responses.push_back(false);
        }
    }

    // Figure out how well the classifier worked.
    classification_rate = 100 - error_sum / y_actual.size() * 100;

    return responses;
}

/**
 * This function takes the output of the perceptron and activates it to be either 1 or -1 depending on what class the perceptron is estimating the inputs blong to.
 * @param value The output of the perceptron to activate.
 * @return The activated result of the given value. This should be 1 if value >= 0 and -1 otherwise.
 */
int Perceptron::activation_function(double value) {
    if(value >= 0.0){
        return 1;
    }else{
        return -1;
    }
}

/**
 * Returns the classification rate of the last test of classifying data.
 * @return The classification rate of the last set of testing data as a double.
 */
double Perceptron::get_classification_rate() {
    return classification_rate;
}

/**
 * Returns the current weights of the perceptron.
 * @return The current weights of the perceptron as a vector of doubles.
 */
vector<double> Perceptron::get_weights() {
    return weights;
}

/**
 * Returns the learning rate the algorithm is currently learning at. This rate influences how quickly or slowly the algorithm converges on a set of weights that produce accurate results.
 * @return The current learning rate of the perceptron as a double.
 */
double Perceptron::get_learning_rate() {
    return learning_rate;
}

/**
 * Changes the learning rate to the given double.
 * @param rate The double to set to use as the new learning rate.
 */
void Perceptron::set_learning_rate(double rate) {
    if(rate <= 0.0){
        return; // Do not allow a negative or 0 learning rate.
    }
    learning_rate = rate;
}