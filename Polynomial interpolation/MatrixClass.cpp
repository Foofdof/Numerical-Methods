#include "MatrixClass.h"
#include <algorithm>
#include <iomanip>

MatrixClass::MatrixClass() {
	N = 0;
	M = 0;
}


MatrixClass::MatrixClass(int N, int M) {
	this->N = N;
	this->M = M;
	for(int i = 0; i < N; i++) {
		std::vector<double > str;
		for (int j = 0; j < M; j++) {
			std::cout << "Enter " << i+1 << " " << j+1;
			double t;
			std::cin >> t;
			str.push_back(t);
		}
		Matrix.push_back(str);
		std::cout << std::endl;
	}
	std::cout << std::endl;
}


MatrixClass::MatrixClass(std::vector<std::vector<double>>& Matrix) {
		this->N = Matrix.size();
		this->M = Matrix[0].size();
		this->Matrix = Matrix;
	}


MatrixClass::MatrixClass(MatrixClass &A) {
		this->N = A.N;
		this->M = A.M;
		this->Matrix = A.Matrix;
	}

double MatrixClass::norm() {
	double norm = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			norm += Matrix[i][j] * Matrix[i][j];
		}
	}
	norm = sqrt(norm);
	return norm;
}


void MatrixClass::show() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			if (abs(Matrix[i][j]) <= 0.0000000000000001) { std::cout << std::setw(10) << 0 << "  "; }
			else { std::cout << std::setw(10) << Matrix[i][j] << "  "; }
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}