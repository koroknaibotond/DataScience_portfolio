#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

// Matrix operations

double dot_product(std::vector<double> a, std::vector<double> b)
{
    double product = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        product += a[i] * b[i];
    }
    return product;
}

double sigmoid(double z)
{
    // Sigmoid function implementation
    // Parameter: we chose z as double since we know that the dot product of two vectors plus a bias is a single real number.
    return 1.0 / (1.0 + std::exp(-z));
}

double log_likelihood(std::vector<std::vector<double>> X,
                      const std::vector<int> y,
                      const std::vector<double> w,
                      double b)
{
    double ll = 0.0; // Since we do not have direct summation in C++, we need to create the sum using a for loop.
    for (size_t i = 0; i < X.size(); ++i)
    {
        // We calculate the linear predictor for each sample
        double z = dot_product(X[i], w) + b;
        // Predict the probability using the sigmoid function
        double p = sigmoid(z);

        // Log-likelihood contribution for sample i
        ll += y[i] * std::log(p) + (1 - y[i]) * std::log(1 - p);
    }
    return ll;
}

// We create a function to fit the logistic regression model using gradient descent

void fit(std::vector<std::vector<double>> X,
         std::vector<int> y,
         double learning_rate,
         int n_iterations,
         std::vector<double> w,
         double b)
{
    // Get number of samples and features
    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    // We initialize the weights and bias to zero at the start of training.
    w.resize(n_features, 0.0);
    b = 0.0;

    for (int iter = 0; iter < n_iterations; ++iter)
    {
        // Initialize gradients for weights and bias
        std::vector<double> dw(n_features, 0.0);
        double db = 0.0;

        // As we loop over the samples.
        for (size_t i = 0; i < n_samples; ++i)
        {
            // We compute the linear predictor
            double z = dot_product(X[i], w) + b;
            // Apply the sigmoid to get predicted probability
            double p = sigmoid(z);
            // And calculate the error between prediction and actual label
            double error = p - y[i];

            // We accumulate gradients for weights
            for (size_t j = 0; j < n_features; ++j)
            {
                dw[j] += error * X[i][j];
            }
            // And for bias
            db += error;
        }

        //
        // Update weights and bias using according to the learning rate.
        for (size_t j = 0; j < n_features; ++j)
        {
            w[j] -= learning_rate * dw[j] / n_samples;
        }

        b -= learning_rate * db / n_samples;

        // Print log-likelihood every 1000 iterations to monitor training
        if (iter % 1000 == 0)
        {
            double ll = log_likelihood(X, y, w, b);
            std::cout << "Iteration " << iter << ", Log-Likelihood: " << ll << std::endl;
        }
    }

    // Write weights and bias to file so tat we can load them in Python for visualization.
    std::ofstream out("weights_bias.txt");
    for (size_t j = 0; j < w.size(); ++j)
        out << w[j] << " ";
    out << b << std::endl;
    out.close();
}

int main()
{
    // Generate synthetic data
    int m = 100;
    int n = 2;
    std::vector<std::vector<double>> X(m, std::vector<double>(n));
    std::vector<int> y(m);

    std::mt19937 gen(42); // We generate random numbers a synthetic data usindg normal distribution
    std::normal_distribution<> d(0, 1);

    for (int i = 0; i < m; ++i)
    {
        X[i][0] = d(gen);
        X[i][1] = d(gen); // With a simple linear decision boundary as we did in Python.
        y[i] = (X[i][0] + X[i][1] > 0) ? 1 : 0;
    }

    // Save X and y to file
    std::ofstream data_out("synthetic_data.txt");
    for (int i = 0; i < m; ++i)
    {
        data_out << X[i][0] << " " << X[i][1] << " " << y[i] << "\n";
    }
    data_out.close();

    // Train the model
    std::vector<double> w;
    double b;
    fit(X, y, 0.1, 10000, w, b);

    return 0;
}
