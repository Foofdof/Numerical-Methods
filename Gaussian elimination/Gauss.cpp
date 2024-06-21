#include <iostream>
#include <vector>
#include "MatrixClass.h"
#include "GaussClass.h"
#include "Jacobi.h"
#include "InterPol.h"
#include <cmath>
#include <iomanip>
#include "Timer.h"


double f(double x) {
    return sin( x*x + 1.5 ) - cos( x + 2) - 1;
}

void findseg(double a, double b, std::vector<double>& seg) {
    double dx = 0.01;
    double x = a;
    while (f(x) * f(a) > 0) {
        a = x;
        x += dx;
        if (x >= b) {
            throw std::exception("Out of range");
            return;
        }
    }
    seg[0] = a;
    seg[1] = x;
}

void diff(std::vector<std::vector<double>>& Mat, std::vector<std::vector<double>>& Mat2) {
    for (int i = Mat.size() - 1; i >= 1; i--) {
        if (Mat2[0][0] != 0) { Mat2[0][0] = 0; }
        Mat2[i][0] = Mat[i - 1][0] * (Mat.size() - i);
    }
}

double findRoot(std::vector<double>& seg, double eps) {
    std::vector<double> x_val;
    std::vector<double> y_val;
    double dx = 0.005;
    double x = seg[0];
    while (x <= seg[1]) {
        x_val.push_back(x);
        y_val.push_back(f(x));
        x += dx;
    }

    std::vector<std::vector<double>> Mat;
    std::vector<std::vector<double>> F1;

    for (int i = 0; i < x_val.size(); i++) {
        std::vector<double> temp;
        for (int j = 0; j < x_val.size(); j++) {
            temp.push_back(pow(x_val[i], j));
        }
        Mat.push_back(temp);
        F1.push_back(std::vector<double> {y_val[i]});
    }

    MatrixClass M(Mat);
    MatrixClass F(F1);
    GaussClass sol2(M, F);
    sol2.solve(); 

    Mat = sol2.getX().Matrix;
    std::reverse(Mat.begin(), Mat.end());
    std::vector<std::vector<double>> Mat2 = Mat;

    diff(Mat, Mat2); F.Matrix = Mat2;
    double root = seg[0];
    double diff = 0;
    do {

        diff = 0;
        for (int i = Mat2.size() - 1; i >= 0; i--) {
            diff += Mat2[i][0] * pow(root, Mat2.size() - i - 1);
        }
        x = root;
        root = root - f(root) / diff;

    }while (abs(root - x) > eps);
    
    return root;
}

void SLAETEST(int N) {
    std::vector<std::vector<double>> MM(N, std::vector<double>(N, 0));
    std::vector<std::vector<double>> FF(N, std::vector<double>(1, 0));
    for (int i = 0; i < N; i++) {
        FF[i][0] = -5 + rand() % (5 + 5 + 1);
        for (int j = 0; j < N; j++) {
            MM[i][j] = - 5 + rand() % (5 + 5 + 1);
        }
    }

    MatrixClass M(MM);
    MatrixClass F(FF);

    GaussClass sol(M, F);
    sol.solve();
}

int main()
{
    std::cout << "--------------------------\n";
    srand(time(NULL));
    std::vector<std::vector<double>> MM = {
        {11, 6, 9, -12},
        { 13, -5, 8, 5 },
        { 3, 10, -1, -3 },
        { 13, -8, 9, 9 }
    };
    std::vector<std::vector<double>> FF{
        { -296 },
        { -61 },
        { -121 },
        { -1 }
    };

    std::vector<std::vector<double>> X_M{
        { -10 },
        { -6 },
        { -2 },
        { 11 }
    };
    MatrixClass M(MM);
    MatrixClass F(FF);
    MatrixClass XX(X_M);
    
    GaussClass sol( M, F);
    std::cout << "Gauss:" << std::endl;
    std::cout << "A:" << std::endl;
    M.show();
    std::cout << "F:" << std::endl;
    F.show();
    sol.solve();
    sol.show();

    MatrixClass X1(sol.getX().Matrix);
    X1 = X1 - XX;
    std::cout << "Norm x-x~: " << X1.norm() << std::endl;
    std::cout << std::endl;

    M = M * sol.getX();
    MatrixClass Re;
    Re = F - M;
    std::cout << "Norm r: " << Re.norm() << std::endl;
    std::cout << std::endl << "R: " << std::endl;
    Re.show();
    
    for (int i = 3; i < 100; i++) {
        Timer t;
        SLAETEST(i);
        std::cout << "{" << i << ", " << t.End().count() << "}," << std::endl;
    }
    
    /*
    MM = {
        {5, 2, -1},
        {-4, 7, 3},
        {2, -2, 4},
    };
    
    FF = {
        { 12 },
        { 24 },
        { 9 },
    };

    
    std::vector<std::vector<double>> XX = {
        { 0 },
        { 0 },
        { 0 }
    };
    std::cout << "Zeidel:" << std::endl;
    MatrixClass m2 (MM);
    m2.show();
    MatrixClass f2 (FF);
    MatrixClass X (XX);
    Jacobi sol2(m2, f2, X, 0.001);
    sol2.solve();

    m2 = m2 * sol2.getX();
    Re = f2 - m2;
    std::cout << "R:" << std::endl;
    Re.show();

    //Интерполяция
    std::cout << std::endl << "Interpol. Lagrange: " << std::endl;
    std::vector<double> x_val = { 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5 };
    std::vector<double> y_val = { -0.2293,  -0.3258, -0.3086, -0.1836, -0.0056, -0.1928, -0.3127 };
    
    x_val = { 0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43 };
    y_val = { -0.9789,  -0.8264, -0.6806, -0.5408, -0.4062, -0.2764, -0.1507 };
    std::vector<std::vector<double>> Mat;
    std::vector<std::vector<double>> F1;

    for (int i = 0; i < x_val.size(); i++) {
        std::vector<double> temp;
        for (int j = 0; j < x_val.size(); j++) {
            temp.push_back( pow(x_val[i], j) );
        }
        Mat.push_back(temp);
        F1.push_back(std::vector<double> {y_val[i]});
    }
    
    MatrixClass M1(Mat); M1.show();
    MatrixClass F11(F1); F11.show();

    GaussClass sol4(M1, F11);
    sol4.solve();

    double x = 1.5;
    InterPol intrp(x, x_val, y_val, 0);
    std::cout << intrp.solve(0, x);
    std::cout << std::endl;
    //Решение уравнений
    std::cout << std::endl << "Solving of Equations: " << std::endl;
    double a = -2; double b = 2; double eps = 0.00001;
    std::vector<double> seg = { 0, 0 };
    do {
        try {
            findseg(a, b, seg);
            std::cout << "Segment: " << seg[0] << ", " << seg[1] << "; Root: " << findRoot(seg, eps) << std::endl;
        }
        catch(std::exception e) {
            break;
        }
        a = seg[1];
    } while (true);

    */
}
