#pragma once
#include "MatrixClass.h"

class GaussClass
{
private:
	MatrixClass A;
	MatrixClass X;
	MatrixClass F;
public:
	GaussClass(MatrixClass& A1, MatrixClass& F1);
	void solve();
	void change(int I, int J);
	MatrixClass& getX() {
		
		return X;
	}
	void show() {
		std::cout << "X~:" << std::endl;
		X.show();
	}
};

