#include <iostream>
#include <Eigen/Dense>
#include "Dataset/Dataset.hpp"
#include <vector>
#include "logistic_regression/LogisticRegression.hpp"
#include "logistic_regression/MulticlassClassifier.hpp"
#include <string>

using namespace std;

int main() {
    string process;
    cout << "Enter 'binary' for binary classification or 'multiclass' for multiclass classification: ";
    cin >> process;
    
    if (process == "binary")
    {
        // Load the dataset
        Dataset X_train("data/representation.eng.train.csv");
        Dataset y_train("data/true_labels.eng.train.csv", true, false);
        
        // Parameters for logistic regression
        double lr = 0.001;
        long m_epochs = 500;

        // Create an instance of LogisticRegression and fit the model.
        cout << "\nFitting the model..." << endl;
        LogisticRegression logReg(&X_train, &y_train, lr, m_epochs);
        cout << "Model fitted.\n" << endl;

        // metrics on the training set
        cout << "Confusion matrix on the training set:" << endl;
        logReg.show_confusion_matrix(X_train, y_train);
        cout << "Accuracy on training set: " << logReg.accuracy(X_train, y_train) << endl;
        cout << "Precision on training set: " << logReg.precision(X_train, y_train) << endl;
        cout << "Recall on training set: " << logReg.recall(X_train, y_train) << endl;
        cout << "F1 score on training set: " << logReg.f1_score(X_train, y_train) << endl;
        
        // Accuracy on the test set a
        Dataset X_test_a("data/representation.eng.testa.csv");
        Dataset y_test_a("data/true_labels.eng.testa.csv", true);
        cout << "\nEvaluating the model on the test set a..." << endl;
        
        cout << "Confusion matrix on the test set a:" << endl;
        logReg.show_confusion_matrix(X_test_a, y_test_a);
        cout << "Accuracy on test set a: " << logReg.accuracy(X_test_a, y_test_a) << endl;
        cout << "Precision on test set a: " << logReg.precision(X_test_a, y_test_a) << endl;
        cout << "Recall on test set a: " << logReg.recall(X_test_a, y_test_a) << endl;
        cout << "F1 score on test set a: " << logReg.f1_score(X_test_a, y_test_a) << endl;

        // Accuracy on the test set b
        Dataset X_test_b("data/representation.eng.testb.csv");
        Dataset y_test_b("data/true_labels.eng.testb.csv", true);
        cout << "\nEvaluating the model on the test set b..." << endl;

        cout << "Confusion matrix on the test set b:" << endl;
        logReg.show_confusion_matrix(X_test_b, y_test_b);
        cout << "Accuracy on test set b: " << logReg.accuracy(X_test_b, y_test_b) << endl;
        cout << "Precision on test set b: " << logReg.precision(X_test_b, y_test_b) << endl;
        cout << "Recall on test set b: " << logReg.recall(X_test_b, y_test_b) << endl;
        cout << "F1 score on test set b: " << logReg.f1_score(X_test_b, y_test_b) << endl;
    } 
    else if (process == "multiclass")
    {
        string choice;
        cout << "Enter 'one_vs_all' for one vs all or 'one_vs_one' for one vs one: ";
        cin >> choice;

        {
            // Parameters for logistic regression for multiclass classification
            double lr = 0.001;
            long m_epochs = 500;

            // Load the dataset
            Dataset X_train1("data/representation.eng.train.csv");
            Dataset y_train1("data/true_labels.eng.train.csv", false, true);
            

            // Create an instance of MulticlassClassifier and fit the model
            cout << "\nFitting the model..." << endl;
            MulticlassClassifier classifier(&X_train1, &y_train1, lr, m_epochs, choice);
            cout << "Model fitted using " << choice << " Strategy\n" << endl;

            // metrics on the training set
            classifier.show_confusion_matrix(X_train1, y_train1);
            cout << "Accuracy on training set: " << classifier.accuracy(X_train1, y_train1) << endl;
            cout << "Precision on training set: " << classifier.precision(X_train1, y_train1) << endl;
            cout << "Recall on training set: " << classifier.recall(X_train1, y_train1) << endl;
            cout << "F1 score on training set: " << classifier.f1_score(X_train1, y_train1) << endl;

            // metrics on the test set a
            cout << "\nEvaluating the model on the test set a" << endl;
            Dataset X_test_a("data/representation.eng.testa.csv");
            Dataset y_test_a("data/true_labels.eng.testa.csv", false, true);

            classifier.show_confusion_matrix(X_test_a, y_test_a);
            cout << "Accuracy on test set a: " << classifier.accuracy(X_test_a, y_test_a) << endl;
            cout << "Precision on test set a: " << classifier.precision(X_test_a, y_test_a) << endl;
            cout << "Recall on test set a: " << classifier.recall(X_test_a, y_test_a) << endl;
            cout << "F1 score on test set a: " << classifier.f1_score(X_test_a, y_test_a) << endl;

            // metrics on the test set b
            cout << "\nEvaluating the model on the test set b " << endl;
            Dataset X_test_b("data/representation.eng.testb.csv");
            Dataset y_test_b("data/true_labels.eng.testb.csv", false, true);

            classifier.show_confusion_matrix(X_test_b, y_test_b);
            cout << "Accuracy on test set a: " << classifier.accuracy(X_test_b, y_test_b) << endl;
            cout << "Precision on test set a: " << classifier.precision(X_test_b, y_test_b) << endl;
            cout << "Recall on test set a: " << classifier.recall(X_test_b, y_test_b) << endl;
            cout << "F1 score on test set a: " << classifier.f1_score(X_test_b, y_test_b) << endl;


            // metrics on the datasets of task 4
            cout << "\nEvaluating the model on the datasets of task 4" << endl;

            // metrics on the test set BTC
            cout << "\nEvaluating the model on the test set BTC " << endl;
            Dataset X_test_BTC("data/representation.e.conll.csv");
            Dataset y_test_BTC("data/true_labels.e.conll.csv", false, true);

            classifier.show_confusion_matrix(X_test_BTC, y_test_BTC);
            cout << "Accuracy on test set BTC: " << classifier.accuracy(X_test_BTC, y_test_BTC) << endl;
            cout << "Precision on test set BTC: " << classifier.precision(X_test_BTC, y_test_BTC) << endl;
            cout << "Recall on test set BTC: " << classifier.recall(X_test_BTC, y_test_BTC) << endl;
            cout << "F1 score on test set BTC: " << classifier.f1_score(X_test_BTC, y_test_BTC) << endl;

            // metrics on the test set WIKIGOLD
            cout << "\nEvaluating the model on the test set WIKIGOLD " << endl;
            Dataset X_test_WIKIGOLD("data/representation.wikigold.conll.txt.csv");
            Dataset y_test_WIKIGOLD("data/true_labels.wikigold.conll.txt.csv", false, true);

            classifier.show_confusion_matrix(X_test_WIKIGOLD, y_test_WIKIGOLD);
            cout << "Accuracy on test set WIKIGOLD: " << classifier.accuracy(X_test_WIKIGOLD, y_test_WIKIGOLD) << endl;
            cout << "Precision on test set WIKIGOLD: " << classifier.precision(X_test_WIKIGOLD, y_test_WIKIGOLD) << endl;
            cout << "Recall on test set WIKIGOLD: " << classifier.recall(X_test_WIKIGOLD, y_test_WIKIGOLD) << endl;
            cout << "F1 score on test set WIKIGOLD: " << classifier.f1_score(X_test_WIKIGOLD, y_test_WIKIGOLD) << endl;
        }

    } else {
        cout << "Invalid input !" << endl;
    }
    return 0;
}