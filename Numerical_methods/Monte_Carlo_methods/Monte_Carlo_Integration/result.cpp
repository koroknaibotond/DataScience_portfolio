#include <iostream>
#include "montecarlo.hpp"

int main()
{
    MonteCarlo params(
        [](double x, double y, double z)
        { return std::exp(-x * x - y * y - z * z); },
        [](double x, double y, double z) -> bool
        { return x * x + y * y + z * z < 16.0; },
        -4.0, 4.0, -4.0, 4.0, -4.0, 4.0, 2e6); // We set the parameters for the Monte Carlo Integration

    double result = params.getResult(); // With the help of this function we call the result of the function from the private part of the class. ezzel a segédfüggvénnyel előhívjuk a class, private részéből az inegrálás eredményét.
    std::cout << "\n";

    std::cout << "Az integrálás eredménye: " << result << std::endl; // We print out the result
    std::cout << "\n";

    return 0;
}
