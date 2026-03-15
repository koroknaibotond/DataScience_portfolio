#include <iostream>
#include <random>
#include <cmath>

// Monte Carlo integrátor 3D-s függvényekhez
template <typename Func, typename Dom>
double MonteCarlo(Func integrand, Dom domain, double xmin, double xmax, double ymin, double ymax, double zmin, double zmax, int samples = 1e06)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distributionx(xmin, xmax);
    std::uniform_real_distribution<double> distributiony(ymin, ymax);
    std::uniform_real_distribution<double> distributionz(zmin, zmax);

    int indomain = 0;
    for (int i = 0; i < samples; ++i)
    {
        double x = distributionx(gen);
        double y = distributiony(gen);
        double z = distributionz(gen);

        if (domain(x, y, z))
        {
            double value = integrand(x, y, z);
            indomain++;
        }
    }

    double volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
    double integral = static_cast<double>(indomain) / static_cast<double>(samples) * volume;
    return integral;
}
