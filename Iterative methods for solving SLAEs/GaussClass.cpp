#include "GaussClass.h"
#include <thread>
#include <chrono>
#include <iomanip>




GaussClass::GaussClass(MatrixClass& A1, MatrixClass& F1) {
    A.setN(A1.getN());
    A.setM(A1.getN());
    X.setN(A1.getN());
    X.setM(1);
    A.Matrix = A1.Matrix;
    F.Matrix = F1.Matrix;
    F.setN(A1.getN()); F.setM(1);
}

void GaussClass::solve() {
    double summ;

    //Прямой ход
    
    for (int i = 0; i < A.getN(); i++) {
        //change(i, 0);
        F.Matrix[i][0] /= A.Matrix[i][i];
        for (int j = A.getN() - 1; j >= 0; j--) {       //деление строки на первый эл-т
            A.Matrix[i][j] = A.Matrix[i][j] / A.Matrix[i][i];
        }
        for (int j = A.getN() - 1; j >= i; j--) {
            for (int k = i + 1; k < A.getN(); k++) {
                A.Matrix[k][j] -= A.Matrix[i][j] * A.Matrix[k][i];
                if (j == (A.getN()-1)) {
                    F.Matrix[k][0] -= F.Matrix[i][0] * A.Matrix[k][i];
                }
            }
        }

        //for (int j = A.getN()-1; j >=0 ; j--) {
        //    A.Matrix[i][j] = A.Matrix[i][j] / A.Matrix[i][i];
        //    for (int k = i + 1; k < A.getN(); k++) {
        //        //A.show();
        //        A.Matrix[k][j] = A.Matrix[k][j] - A.Matrix[k][i] * A.Matrix[i][j];
        //        if (k == A.getN() - 1) {
        //            F.Matrix[k][0] = F.Matrix[k][0] - F.Matrix[i][0]* A.Matrix[i][j];
        //        }
        //    }
        //}
    }
    if ( abs(A.Matrix[A.getN() - 1][A.getN()-1]) <= 0.0000000001) {
        throw std::exception("Determinant = 0");
    }
    //std::cout << "Diagonal matrix:" << std::endl;
    //std::cout << "A:" << std::endl;
    //A.show();
    //std::cout << "F:" << std::endl;
    //F.show();
    //Обратный ход
    for (int i = A.getN() - 1; i >= 0; i--) {
        summ = 0;
        for (int j = i + 1; j < A.getN(); j++) {
            summ += A.Matrix[i][j] * X.Matrix[X.Matrix.size() - (j - i)][0];
        }
        double x = F.Matrix[i][0] - summ;
        if (abs(x) <= 0.00000000000001) { x = 0; };
        X.Matrix.push_back(std::vector<double >{x});
    }
    std::reverse(X.Matrix.begin(), X.Matrix.end());
    
}

void GaussClass::change(int I, int J) {
    int im = I, jm = J;
    for (int i = I; i < A.getN(); i++) {
        for (int j = I; j < A.getN(); j++) {
            if (abs(A.Matrix[i][j]) > abs(A.Matrix[im][jm])) {
                im = i; jm = j;
            }
        }
        std::vector<double> temp = A.Matrix[I];
        A.Matrix[I] = A.Matrix[im];
        A.Matrix[im] = temp;

        temp = F.Matrix[I];
        F.Matrix[I] = F.Matrix[im];
        F.Matrix[im] = temp;
    }
}