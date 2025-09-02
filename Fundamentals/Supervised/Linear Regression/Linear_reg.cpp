#include <vector>
#include <random>
#include <iostream>
#include <fstream>

// we create struct to hold the dataset together

struct DataSet
{
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::vector<double> w_true;
    double b_true;
};

struct LinearParam
{
    std::vector<double> w;
    double b;
};

DataSet create_data(int n_samples, int n_features)
{
    DataSet data;

    // we generate random numbers once again
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist_x(0.0, 10.0); // for the x values
    std::normal_distribution<> dist_noise(0.0, 4.0);    // adding some noise
    std::uniform_int_distribution<> dist_wb(1, 10);     // and randomly generated weights and bias

    // we resize the feature vector to match the shape of the features
    data.w_true.resize(n_features);
    for (int i = 0; i < n_features; ++i)
    {
        data.w_true[i] = dist_wb(gen);
    }
    data.b_true = dist_wb(gen);

    // Resize X to hold n_samples rows with the lenght of n_features
    data.X.resize(n_samples, std::vector<double>(n_features));

    // Generate the data
    data.y.resize(n_samples);
    for (int i = 0; i < n_samples; ++i)
    {
        double target = data.b_true;
        for (int j = 0; j < n_features; ++j) // we fill up the vector (or matrix in case of higher dimension) with the randomly generated noisy data
        {
            data.X[i][j] = dist_x(gen);
            target += data.X[i][j] * data.w_true[j];
        }
        target += dist_noise(gen);
        data.y[i] = target;
    }
    return data;
}
// To create the linear regression we need to use some matrix operations
//  I have created a simple function which calculates the dot product of two matrices
double dot_product(std::vector<double> a, std::vector<double> b)
{
    double product = 0;
    for (size_t i = 0; i < a.size(); i++)
    {
        product += a[i] * b[i];
    }
    return product;
}

// We also need to create a function which calculates the inverse matrix, but first we need to have an identity matrix for that.

std::vector<std::vector<double>> identity(int n)
{
    std::vector<std::vector<double>> I(n, std::vector<double>(n, 0.0)); // start with all zeros

    for (int i = 0; i < n; i++)
    {
        I[i][i] = 1.0; // set diagonal to 1
    }

    return I;
}

// A function for matrix multiplication
std::vector<std::vector<double>> matmul(
    std::vector<std::vector<double>> a,
    std::vector<std::vector<double>> b)
{
    int n = a.size();    // number of rows in a
    int m = a[0].size(); // number of columns in a
    int p = b[0].size(); // number of columns in b

    // Check if inner dimensions match
    if (b.size() != m)
        throw std::runtime_error("Matrix dimensions do not match for multiplication");

    std::vector<std::vector<double>> c(n, std::vector<double>(p, 0.0)); // we create a new matrix and fill it up with zeroes

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return c;
}

// Finally we need to transpose the matrices
std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> a)
{
    int rows = a.size();
    int cols = a[0].size();
    std::vector<std::vector<double>> T(cols, std::vector<double>(rows)); // we create the new matrice by switcching up the number or rows and columns
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            T[j][i] = a[i][j];
        }
    }
    return T;
}

// As we need to center the data, we must create a function which calcualtes the means of our synthetic data

std::vector<double> column_means(std::vector<std::vector<double>> a)
{
    int n_rows = a.size();
    int n_cols = a[0].size();
    std::vector<double> result;
    for (int j = 0; j < n_cols; ++j)
    {
        double sum = 0.0; // temporary variable for each column
        for (int i = 0; i < n_rows; ++i)
        {
            sum += a[i][j];
        }
        result.push_back(sum / n_rows); // divide by number of rows and append the result vector with the value of the mean.
    }

    return result;
}

std::vector<std::vector<double>> inversion(std::vector<std::vector<double>> a)
{
    int n = a.size();
    auto I = identity(n);
    for (int i = 0; i < n; i++)
    {
        // As first step we try to make the diagonal elements zero
        double diag = a[i][i];
        if (diag == 0.0)
            throw std::runtime_error("Matrix is singular!"); // in this case it cannot be inverted, thus we throw an error.
        for (int j = 0; j < n; j++)
        {
            a[i][j] /= diag; // since this is an augmented matrix, we work with both matrices the same time.
            I[i][j] /= diag;
        }

        //  The next step is to make other elements in column 0.
        for (int k = 0; k < n; k++)
        {
            if (k == i) // we start looping over the rows, but skipping the pivot row.
                continue;
            double factor = a[k][i]; // we  select the element we would like to eliminate.
            for (int j = 0; j < n; j++)
            {
                a[k][j] -= factor * a[i][j]; // we eliminate the selected element
                I[k][j] -= factor * I[i][j]; // and we do the same in the identity matrix
            }
        }
    } // once we transform our original matrix into an identity matrix, the modified identity matrix will give us the inverse matrix.

    return I;
}

// Linear regression using the normal equation
LinearParam LinearRegression(DataSet data)
{
    // 1. Center X
    std::vector<std::vector<double>> X_tilde = data.X;      // create a nex matrix
    std::vector<double> X_col_means = column_means(data.X); // calcualte the means
    for (size_t i = 0; i < X_tilde.size(); ++i)
        for (size_t j = 0; j < X_tilde[0].size(); ++j) // and substract it from the original
            X_tilde[i][j] -= X_col_means[j];

    // 2. Center y using column_means
    std::vector<std::vector<double>> y_matrix(data.y.size(), std::vector<double>(1));
    for (size_t i = 0; i < data.y.size(); ++i)
        y_matrix[i][0] = data.y[i];

    std::vector<double> y_mean_vec = column_means(y_matrix); // repeat the same on y values

    std::vector<double> y_tilde = data.y;
    for (size_t i = 0; i < y_tilde.size(); ++i)
        y_tilde[i] -= y_mean_vec[0];

    // Convert 1D vector y_tilde into a column 2D vector
    std::vector<std::vector<double>> y_col(y_tilde.size(), std::vector<double>(1));
    for (size_t i = 0; i < y_tilde.size(); ++i)
    {
        y_col[i][0] = y_tilde[i];
    }

    auto w_mat = matmul(inversion(matmul(transpose(X_tilde), X_tilde)), matmul(transpose(X_tilde), y_col));
    // since we use one feture in th moment, I have flattened the w, so the is no dimension mismatch in the dot product
    std::vector<double> w(w_mat.size());
    for (size_t i = 0; i < w_mat.size(); ++i)
        w[i] = w_mat[i][0];

    // compute bias
    double mean_y = 0.0;
    for (auto val : data.y)
        mean_y += val;
    mean_y /= data.y.size();

    auto mean_X = column_means(data.X);
    double dot = dot_product(mean_X, w);

    double b = mean_y - dot;

    // return parameters
    LinearParam params;
    params.w = w;
    params.b = b;
    return params;
}

// test case
int main()
{
    int n_samples = 100;
    int n_features = 1;

    // 1. We generate a synthetic dataset
    DataSet data = create_data(n_samples, n_features);

    // Save it for plotting
    std::ofstream file("dataset.csv");
    for (int i = 0; i < n_samples; ++i)
    {
        for (int j = 0; j < n_features; ++j)
        {
            file << data.X[i][j];
            if (j < n_features - 1)
                file << ",";
        }
        file << "," << data.y[i] << "\n";
    }
    file.close();
    std::cout << "Dataset saved to dataset.csv\n";

    // Perform linear regression
    LinearParam params = LinearRegression(data);

    // Display predicted weights and bias
    std::cout << "Predicted weights: ";
    for (auto w : params.w)
        std::cout << w << " ";
    std::cout << "\nPredicted bias: " << params.b << "\n";

    // Display true weights and bias
    std::cout << "True weights: ";
    for (auto w : data.w_true)
        std::cout << w << " ";
    std::cout << "\nTrue bias: " << data.b_true << "\n";

    return 0;
}
