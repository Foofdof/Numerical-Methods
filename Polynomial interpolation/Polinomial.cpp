#include "Polinomial.h"


Polinomial Polinomial::operator*(Polinomial& A)
{
	int N = A.pol.size() + pol.size();
	std::vector<double> tmp(N, 0);
	for (int i = 0; i < pol.size(); i++) {
		for (int j = 0; j < A.pol.size(); j++) {
			tmp[i + j] += pol[i] * A.pol[j];
		}
	}
	return Polinomial(tmp);;
}

Polinomial Polinomial::operator*(double& A)
{
	std::vector<double> tmp(pol.size(), 0);
	for (int i = 0; i < pol.size(); i++) {
		tmp[i] = pol[i] * A;
	}
	return Polinomial(tmp);;
}

Polinomial Polinomial::operator/(double& A)
{
	std::vector<double> tmp(pol.size(), 0);
	for (int i = 0; i < pol.size(); i++) {
		tmp[i] = pol[i] / A;
	}
	return Polinomial(tmp);;
}

Polinomial Polinomial::operator+(Polinomial& A)
{
	int N = std::max(A.pol.size(), pol.size());
	std::vector<double> tmp(N, 0);

	for (int i = 0; i < std::min(A.pol.size(), pol.size()); i++) {
		tmp[i] = A.pol[i] + pol[i];
	}

	for (int i = std::min(A.pol.size(), pol.size()); i < N; i++) {
		tmp[i] = A.pol[i] + pol[i];
	}

	return Polinomial(tmp);
}

Polinomial Polinomial::operator-(Polinomial& A)
{
	int N = std::max(A.pol.size(), pol.size());
	std::vector<double> tmp(N, 0);

	for (int i = 0; i < std::min(A.pol.size(), pol.size()); i++) {
		tmp[i] = - A.pol[i] + pol[i];
	}

	for (int i = std::min(A.pol.size(), pol.size()); i < N; i++) {
		tmp[i] = A.pol[i] + pol[i];
	}

	return Polinomial(tmp);
}

double Polinomial::getValue(double x) {
	double answr = 0;
	for (int i = 0; i < pol.size(); i++) {
		answr += pol[i]*pow(x, i);
	}
	return answr;
}
