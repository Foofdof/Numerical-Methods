#pragma once
#include <vector>
#include <iostream>
class MatrixClass
{
private:
	int N;
	int M;
public:
	std::vector<std::vector<double>> Matrix;
public:
	MatrixClass();
	MatrixClass(int N, int M);
	MatrixClass(std::vector<std::vector<double>>& Matrix);
	MatrixClass(MatrixClass &A);
	MatrixClass(const MatrixClass& A) {
		this->N = A.N;
		this->M = A.M;
		this->Matrix = A.Matrix;
	};

	void show();
	MatrixClass operator * (MatrixClass& A) {
		try {
			if ((M != A.N)) {
				throw std::exception();
			}
			else {
				std::vector<std::vector<double>> C;
				for (int i = 0; i < A.N; i++) {
					std::vector<double > str;
					for (int j = 0; j < A.M; j++) {
						double sum = 0;
						for (int k = 0; k < M; k++) {
							sum += Matrix[i][k] * A.Matrix[k][j];
						}
						str.push_back(sum);
					}
					C.push_back(str);
				}
				MatrixClass tempt(C);
				return tempt;
			}
		}
		catch (const std::exception& err) {
			std::cout << "Error" <<std::endl;
		}
		
	}
	MatrixClass operator + (MatrixClass& A) {
		try {
			if ((A.N != N) && (A.M != M)) {
				throw std::exception();
			}
			else {
				std::vector<std::vector<double>> C;
				for (int i = 0; i < N; i++) {
					std::vector<double > str;
					for (int j = 0; j < M; j++) {
						str.push_back(Matrix[i][j] + A.Matrix[i][j]);
					}
					C.push_back(str);
				}
				MatrixClass tempt(C);
				return tempt;
			}
		}
		catch (const std::exception& err) {
			std::cout << "Error" << std::endl;
		}

	}
	MatrixClass operator - (MatrixClass& A) {
		try {
			if ((A.N != N) && (A.M != M)) {
				throw std::exception();
			}
			else {
				std::vector<std::vector<double>> C;
				for (int i = 0; i < N; i++) {
					std::vector<double > str;
					for (int j = 0; j < M; j++) {
						str.push_back(Matrix[i][j] - A.Matrix[i][j]);
					}
					C.push_back(str);
				}
				MatrixClass tempt(C);
				return tempt;
			}
		}
		catch (const std::exception& err) {
			std::cout << "Error" << std::endl;
		}

	}
	
	double norm();
	/*friend const MatrixClass& operator*(const MatrixClass& A, const MatrixClass& B) {
		try {
			if ((A.M != B.N)) {
				throw std::exception();
			}
			else {
				std::vector<std::vector<double>> C;
				for (int i = 0; i < A.N; i++) {
					std::vector<double > str;
					for (int j = 0; j < B.M; j++) {
						double sum = 0;
						for (int k = 0; k < A.M; k++) {
							sum += A.Matrix[i][k] * B.Matrix[k][j];
						}
						str.push_back(sum);
					}
					C.push_back(str);
				}
				;
				return MatrixClass (C);
			}
		}
		catch (const std::exception& err) {
			std::cout << "Error" << std::endl;
		}
	}*/

	int getN() {
		return this->N;
	}
	int getM() {
		return this->M;
	}

	void setN(int N) {
		this->N = N;
	}
	void setM(int M) {
		this->M = M;
	}

	MatrixClass Transp() {
		if (N = M) {
			std::vector<std::vector<double>> C = Matrix;
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) {
					C[i][j] = Matrix[j][i];
				}
			}
			MatrixClass tempt(C);
			return tempt;
		}
	}
};

