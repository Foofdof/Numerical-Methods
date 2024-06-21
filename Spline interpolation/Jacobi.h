#pragma once
#include "MatrixClass.h"

class Jacobi
{
private:
	MatrixClass A;
	MatrixClass X;
	MatrixClass F;
	double eps;
public:
	Jacobi(MatrixClass& A1, MatrixClass& F1, MatrixClass& X1, double eps);
	void change();
	double solve(double w);
	void solveJacobi();
	void solveZiedel();
	double CurEps();
	MatrixClass& getX() {
		return X;
	}
};

