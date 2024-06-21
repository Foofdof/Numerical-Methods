#pragma once
#include <vector>
#include <algorithm>

class Polinomial
{
private:
	std::vector<double> pol;
public:
	Polinomial(std::vector<double>& pol) {
		this->pol = pol;
	}
	Polinomial(const std::vector<double>& pol) {
		this->pol = pol;
	}

	std::vector<double>& getPol() {
		return pol;
	}

	Polinomial operator* (Polinomial& A);
	Polinomial operator* (double& A);
	Polinomial operator/ (double& A);
	Polinomial operator+ (Polinomial& A);
	Polinomial operator- (Polinomial& A);
	double getValue(double x);
};

