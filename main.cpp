/*
 * Created by brianwing90@gmail.com on 4/6/2018.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include "Perceptron.h"

using namespace std;

const string CLASS1_NAME = "Iris-setosa";
const string CLASS2_NAME = "Iris-versicolor";

/**
 *
 * @return A normal run returns 0 as a no-error completion code.
 */
int main(){

    /*******************************************
     ***** Read in the data from the file. *****
     *******************************************/
    ifstream file("data.txt");
    string row;
    vector<vector<double>> observations; // This will hold the inputs of the data set once parsed from the file.
    vector<double> max; // This holds the greatest value of each feature for all observations of the data set.
    while(getline(file, row)){ // Get each row of data from the file.
        string col;
        stringstream ss(row);
        vector<double> line;
        line.push_back(1.0); // This ensures weight[0] acts as a bias instead of a weight on an input.
        while(getline(ss, col, ',')){ // Get each feature of data from each row of the data set.
            if(col == CLASS1_NAME){
                line.push_back(1);
            }else if(col == CLASS2_NAME){
                line.push_back(-1);
            }else{
                double feature = atof(col.c_str());
                line.push_back(feature);
            }
        }

        // Keep track of the greatest values of each feature for normalization later.
        for(int i = 0; i < line.size() - 1; i++){ // -1 excludes the class data from the maximums since we will not be normalizing the class data.
            if(max.size() <= i){
                max.push_back(line[i]); // If there is no value for the maximum for this feature so far then add this value as the max for this feature.
            }else if(max[i] < line[i]){
                max[i] = line[i]; // If the current max is less than this value of the feature then add this value as the max for this feature.
            }
        }

        observations.push_back(line); // Add this observation to the list of observations.
    }

    /************************************************************************************
     ***** Randomize the input data and strip the class data off the features. This *****
     ***** ensures that the perceptron must really learn what inputs are present in *****
     ***** a given class and not just follow a pattern.                             *****
     ************************************************************************************/
    random_shuffle(observations.begin(), observations.end()); // Randomize the data.
    vector<int> y_actual; // This is the actual class values of the data set. This is the actual output.
    for(int i = 0; i < observations.size(); i++){
        y_actual.push_back(observations[i].back()); // Get the class ID for this input.
        observations[i].pop_back(); // Strip off the class data from the input.
    }

    /************************************************************************************
     ***** Normalize the input data. This is good practice when starting to work on *****
     ***** a data set as it makes for easier processing of the data and allows the  *****
     ***** results of the process to be displayed in a more efficient manner.       *****
     ************************************************************************************/
    for(vector<double> observation : observations){
        for(int i = 0; i < observation.size() - 1; i++){ // -1 excludes the class data from the maximums since we will not be normalizing the class data.
            observation[i] = observation[i] / max[i]; // Normalize the data by dividing each element of each feature by the maximum value of that feature.
        }
    }

    /********************************************************************************
     ***** Create a perceptron and train it on a sample of the randomized data. *****
     ********************************************************************************/
    Perceptron perceptron(observations[0].size());
    vector<vector<double>> training_inputs;
    vector<int> training_outputs;
    for(int i = 0; i < observations.size() / 3; i++){ // Train the perceptron using the first 1/3 of the data set.
        training_inputs.push_back(observations[i]);
        training_outputs.push_back(y_actual[i]);
    }
    perceptron.train(training_inputs, training_outputs);
    vector<vector<double>> testing_inputs;
    vector<int> testing_outputs;
    for(int i = observations.size() / 3; i < observations.size(); i++){ // Test the perceptron using the last 2/3 of the data set.
        testing_inputs.push_back(observations[i]);
        testing_outputs.push_back(y_actual[i]);
    }
    vector<bool> responses = perceptron.test(testing_inputs, testing_outputs);

    /**********************************
     ***** Print out the results. *****
     **********************************/
    for(int i = 0; i < responses.size(); i++){
        printf("Correct classification for input %i? %s\n", (i + 1), responses[i] ? "yes" : "no");
    }
    printf("\nClassification Rate: %f%%\n", perceptron.get_classification_rate());
    printf("Learning Rate: %f\n", perceptron.get_learning_rate());
    vector<double> weights = perceptron.get_weights();
    printf("\nWeight 0 (Bias): %f\n", weights[0]);
    for(int i = 1; i < weights.size(); i++){
        printf("Weight %i: %f\n", i, weights[i]);
    }

    return 0;
}