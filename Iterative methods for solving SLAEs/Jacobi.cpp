#include "Jacobi.h"
#include <algorithm>

Jacobi::Jacobi(MatrixClass& A1, MatrixClass& F1, MatrixClass& X1, double eps) {
	this->A = A1;
	this->F = F1;
	this->X = X1;
	this->eps = eps;
}

void Jacobi::change() {
	//A.show();
	int I = 0; int J = 0;
	bool p = false;
	for (int i = 0; i < A.Matrix.size(); i++) {
		double Max = abs(A.Matrix[i][i]);
		for (int j = i; j < A.Matrix.size(); j++) {

			if (Max < abs(A.Matrix[j][i])) {
				Max = abs(A.Matrix[j][i]);
				I = j; J = i;
				p = true;
			}
		}
		if (p) {
			std::vector<double> temp = A.Matrix[I];
			A.Matrix[I] = A.Matrix[i];
			A.Matrix[i] = temp;

			temp = F.Matrix[I];
			F.Matrix[I] = F.Matrix[i];
			F.Matrix[i] = temp;

			temp = X.Matrix[I];
			X.Matrix[I] = X.Matrix[i];
			X.Matrix[i] = temp;
			p = false;
		}
	}
	//A.show();
}

double Jacobi::CurEps() {
	MatrixClass Re;
	Re = A*X;
	Re = F - Re;
	
	return Re.norm()/F.norm();
}

double Jacobi::solve(double w) {
	//this->F = this->A.Transp() * this->F;
	//this->A = this->A.Transp() * this->A;
	//this->change();
	double counter = 0;
	MatrixClass X_old(X);
	std::vector<double> delta(X.Matrix.size(), 0);
	do {
		counter++;
		X_old.Matrix = X.Matrix;
		for (int i = 0; i < X.Matrix.size(); i++) {
			double summ = F.Matrix[i][0];
			for (int j = 0; j < A.Matrix.size(); j++) {
				if (i != j) {
					summ -= A.Matrix[i][j]*X.Matrix[j][0];
				}
			}
			summ /= A.Matrix[i][i];
			delta[i] = abs(summ - X.Matrix[i][0]);
			X.Matrix[i][0] = summ;
			
		}
		for (int i = 0; i < X.Matrix.size(); i++) {
			X.Matrix[i][0] = w * X.Matrix[i][0] + (1 - w) * X_old.Matrix[i][0];
		}
		if (X.norm() >= pow(10,25) ) {
			throw std::exception("inf");
			break;
		}
		//X.show();
		//std::cout << "{" << counter << ", " << CurEps() << "},"<<std::endl;
		//*std::min(delta.begin(), delta.end())
	} while (CurEps() > this->eps);
	std::cout << "Intr (Relks): " << counter << std::endl;
	//X.show();
	return counter;

}

void Jacobi::solveJacobi() {
	//this->F = this->A.Transp() * this->F;
	//this->A = this->A.Transp() * this->A;
	//this->change();
	double counter = 0;
	MatrixClass X_old(X);
	std::vector<double> delta(X.Matrix.size(), 0);
	do {
		counter++;
		X_old.Matrix = X.Matrix;
		for (int i = 0; i < X.Matrix.size(); i++) {
			double summ = F.Matrix[i][0];;
			for (int j = 0; j < A.Matrix.size(); j++) {
				if (i != j) {
					summ -= A.Matrix[i][j] * X_old.Matrix[j][0];
				}
			}
			summ /= A.Matrix[i][i];
			delta[i] = abs(summ - X_old.Matrix[i][0]);
			X.Matrix[i][0] = summ;

		}
		if (X.norm() >= pow(10, 25)) {
			throw std::exception("inf");
			break;
		}
		//std::cout << "{" << counter << ", " << CurEps() << "}," << std::endl;
		
	} while (CurEps() >= this->eps);
	std::cout << "Intr (Jacobi): " << counter << std::endl;
	//X.show();
};

void Jacobi::solveZiedel() {
	//this->F = this->A.Transp() * this->F;
	//this->A = this->A.Transp() * this->A;
	//this->change();
	double counter = 0;
	MatrixClass X_old(X);
	std::vector<double> delta(X.Matrix.size(), 0);
	do {
		counter++;
		X_old.Matrix = X.Matrix;
		for (int i = 0; i < X.Matrix.size(); i++) {
			double summ = F.Matrix[i][0];;
			for (int j = 0; j < A.Matrix.size(); j++) {
				if (i != j) {
					summ -= A.Matrix[i][j] * X.Matrix[j][0];
				}
			}
			summ /= A.Matrix[i][i];
			delta[i] = abs(summ - X.Matrix[i][0]);
			X.Matrix[i][0] = summ;

		}
		if (X.norm() >= pow(10, 25) ) {
			throw std::exception("inf");
			break;
		}
		//std::cout << "{" << counter << ", " << CurEps() << "}," << std::endl;

	} while (CurEps() >= this->eps);
	std::cout << "Intr (Ziedel): " << counter << std::endl;
	//X.show();
};