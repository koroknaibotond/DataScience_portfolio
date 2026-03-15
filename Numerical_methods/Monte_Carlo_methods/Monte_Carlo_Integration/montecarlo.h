#include <iostream>
#include <random>
#include <cmath>

class MonteCarlo
{
private:
    std::random_device rd;
    std::mt19937 gen; // We create a random number generator using the Mersenne Twister engine
    std::uniform_real_distribution<double> distributionx;
    std::uniform_real_distribution<double> distributiony; // To create coordinates, we generate random numbers evenly distributed between the two limits.
    std::uniform_real_distribution<double> distributionz;
    double result;       // To store the result of the Monte Carlo integration
    double controlvalue; // To store the control (deterministic integration) value

public:
    template <typename Func, typename Dom>
    MonteCarlo(Func integrand, Dom domain, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int samples = 0.5e6)
        : gen(rd()), distributionx(xmin, xmax), distributiony(ymin, ymax), distributionz(zmin, zmax)
    {
        // In this constructor, we provide the integrand, the integration domain, and the boundaries of the cuboid that contains the domain,
        // and also initialize the random number generator and distributions with the proper limits.
        try
        {
            result = integrate(integrand, domain, samples); // Perform the Monte Carlo integration
            controlvalue = control(integrand, domain);      // Perform a rectangular method-based integration as control
            double diff = pow(controlvalue - result, 2);    // Calculate the squared difference
            if (diff > 1e-04)                               // If the difference exceeds the allowed error margin
            {
                throw std::runtime_error("Error! The squared difference between the computed result and the control value is greater than 1e-04"); // Throw an error
            }
        }
        catch (const std::runtime_error &e)
        {
            std::cerr << e.what() << std::endl; // Print the error
        }
    }

    double getResult() const
    {
        return result;
    }
    // These are helper functions to access the private members for the Monte Carlo result and the control value.
    double getControl() const
    {
        return controlvalue;
    }

private:
    // Function implementing the Monte Carlo method
    template <typename Func, typename Dom>
    double integrate(Func integrand, Dom domain, int samples) // We specify the integrand, the domain, and the number of repetitions (default is 500,000)
    {
        double sum = 0.0;
        for (int i = 0; i < samples; ++i)
        {
            double x = distributionx(gen); // Generate random coordinates
            double y = distributiony(gen);
            double z = distributionz(gen);

            if (domain(x, y, z)) // Check if the point is inside the domain
            {
                double value = integrand(x, y, z); // If it is, evaluate the integrand at that point
                sum += value;                      // Accumulate the values of the integrand for points inside the domain
            }
        }

        double volume = (distributionx.max() - distributionx.min()) *
                        (distributiony.max() - distributiony.min()) *
                        (distributionz.max() - distributionz.min());
        double integral = sum / static_cast<double>(samples) * volume; // Calculate the integral value
        return integral;
    }

    template <typename Func, typename Dom>
    double control(Func integrand, Dom domain) // Control integration using a rectangular method
    {
        double a = distributionx.min();
        double b = distributionx.max(); // Define the cuboid over which to iterate
        double c = distributiony.min();
        double d = distributiony.max();
        double e = distributionz.min();
        double f = distributionz.max();

        int n = 100;            // Number of divisions per axis
        double h = (b - a) / n; // Step size for x
        double k = (d - c) / n; // Step size for y
        double m = (f - e) / n; // Step size for z

        double sum = 0.0;
        for (int i = 0; i <= n; ++i) // Loop through each axis with the appropriate step size
        {
            double x = a + i * h;
            for (int j = 0; j <= n; ++j)
            {
                double y = c + j * k;
                for (int l = 0; l <= n; ++l)
                {
                    double z = e + l * m;
                    if (domain(x, y, z)) // Ensure that we are integrating over the actual domain, not the whole cuboid
                    {
                        double value = integrand(x, y, z);
                        sum += value;
                    }
                }
            }
        }

        double volume = (b - a) * (d - c) * (f - e);
        double integral = sum * (h * k * m); // Compute the control integral
        return integral;
    }
};
