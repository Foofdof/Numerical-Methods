#pragma once
#include <vector>
#include "Polinomial.h"

class InterPol
{
private:
	double x;
	std::vector<double> x_val;
	std::vector<double> y_val;
	int f;
public:
	InterPol(double x, std::vector<double>& x_val, std::vector<double>& y_val, int f = 0 ) {
		this->x = x;
		this->x_val = x_val;
		this->y_val = y_val;
		this->f = f;
	}

	double getCk(int k);
	double findW(int k);
	double solve(int f, double x);

};

