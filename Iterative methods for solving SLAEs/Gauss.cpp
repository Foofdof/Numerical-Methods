#include <iostream>
#include <vector>
#include "MatrixClass.h"
#include "GaussClass.h"
#include "Jacobi.h"
#include "InterPol.h"
#include <cmath>
#include <iomanip>
#include "Timer.h"
#include "Polinomial.h"


double f(double x) {
    return x * sin(x) / (1 + cos(x) * cos(x));
    //return cos(x)*cos(x);
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

int fact(int i)
{
    if (i == 0) {
        return 1;
    }
    else {
        return i * fact(i - 1);
    }
}

void Hilbert(int N, int p) {
    std::vector<std::vector<double>> MM(N, std::vector<double>(N, 0));
    std::vector<std::vector<double>> MINV(N, std::vector<double>(N, 0));
    std::vector<std::vector<double>> FF(N, std::vector<double>(1, 0));
    std::vector<std::vector<double>> XX(N, std::vector<double>(1, 0));
    for (int i = 0; i < N; i++) {
        XX[i][0] = 1;
        for (int j = 0; j < N; j++) {
            MM[i][j] = 1.0 / (p + i+1 + j+1 - 1);
            double temp = 1;
            MINV[i][j] = MM[i][j] * pow(-1, i+j+2);
            for (int k = 0; k < N - 1; k++) {
                temp *= (p + i+1 + k) * (p + j+1 + k);
            }
            MINV[i][j] = MINV[i][j] * temp / (fact(i+1 - 1) * fact(N - i - 1 ) * fact(j + 1 - 1) * fact(N - j - 1));
        }
    }

    MatrixClass H(MM);
    MatrixClass X(XX);
    MatrixClass F(FF);
    MatrixClass X_0(FF);
    MatrixClass HINV(MINV);
    //std::cout << "NormH * NormH^-1: " << H.norm() * HINV.norm() << std::endl;
    F = H * X;
    GaussClass sol(H, F);
    sol.solve();

    Jacobi sol4(H, F, X_0, 0.1);
    //sol4.solveJacobi();

    Jacobi sol2(H, F, X_0, 0.001);
    sol2.solveZiedel();

    Jacobi sol3(H, F, X_0, 0.001);
    sol3.solve(1.40);

    try {
        std::cout << "N: " << N << std::endl;
        std::cout << "eps Gauss: " << (X - sol.getX()).norm() / X.norm() << std::endl;
        std::cout << "eps Zeidel: " << (X - sol2.getX()).norm() / X.norm() << std::endl;
        std::cout << "eps Relks: " << (X - sol3.getX()).norm() / X.norm() << std::endl;
    }
    catch (std::exception e) {
        std::cout << "error" << std::endl;
    }
    /*H = H * sol.getX();
    MatrixClass Re;
    Re = F - H;
    std::cout << "NormR/NormF: " << Re.norm()/F.norm() << std::endl;*/
    std::cout << std::endl;

}

void ITERTEST(int N) {
    srand(time(NULL));
    std::cout << "Rand Matrix: N = " << N << std::endl;
    std::vector<std::vector<double>> MM(N, std::vector<double>(N, 0));
    std::vector<std::vector<double>> FF(N, std::vector<double>(1, 0));
    std::vector<std::vector<double>> XX(N, std::vector<double>(1, 0));
    for (int i = 0; i < N; i++) {
        XX[i][0] = 1;
        double diag = N + rand() % (N);
        for (int j = i; j < N; j++) {
            if (i == j) {
                MM[i][j] = diag;
            }
            else {
                MM[i][j] = (diag/N - rand() % (N) )/pow(N,0.8);
                MM[j][i] = MM[i][j];
            }
        }
    }
    MatrixClass M(MM);
    MatrixClass F(FF);
    MatrixClass X(XX);
    F = M * X;
    //M.show();
    //F.show();
    try {
        Jacobi sol2(M, F, F, 0.001);
        std::cout << "RELKS___________________________" << std::endl;
        sol2.solve(1.1);

        Jacobi sol3(M, F, F, 0.001);
        std::cout << "Zeidel___________________________" << std::endl;
        sol3.solveZiedel();

        Jacobi sol4(M, F, F, 0.001);
        std::cout << "Jacobi___________________________" << std::endl;
        sol4.solveJacobi();
    }
    catch (std::exception e) {
        std::cout << "error" << std::endl;
    }
    

}

double finp(double x) {
    double e = 2.718281828459045;
    return pow(e, -2 * x * x) * sin(x);
}

double intrp1(int N, int N2) {
    double x_min = -2;
    double x_max = 3;

    std::vector<double> x_val;
    std::vector<double> y_val;
    
    double h = (x_max - x_min) / N;
    for (double x = x_min; x <= x_max; x += h) {
        x_val.push_back(x);
        y_val.push_back(finp(x));
    }

    std::vector<std::vector<double>> Mat(N + 1, std::vector<double>(N + 1, 0));
    std::vector<std::vector<double>> b(N + 1, std::vector<double>(1, 0));


    for (int i = 0; i < x_val.size(); i++) {
        for (int j = 0; j < x_val.size(); j++) {
            Mat[i][j] = pow(x_val[i], j);
        }
        b[i][0] = y_val[i];
    }

    MatrixClass M(Mat);
    MatrixClass F(b);
    GaussClass sol(M, F);
    sol.solve();
    sol.getX().show();

    std::vector<double> tmp(N + 1, 0);
    for (int i = 0; i <= N; i++) {
        tmp[i] = sol.getX().Matrix[i][0];
    }

    Polinomial pol(tmp);
    h = (x_max - x_min) / (N2-1);
    std::vector<double> err;
    for (double x = x_min; x <= x_max; x += h) {
        err.push_back(abs(pol.getValue(x) - finp(x)));
    }
    double delta = *std::max_element(err.begin(), err.end());
    return delta;
}

double intrp2(int N, int N2) {
    double x_min = -2;
    double x_max = 3;

    std::vector<double> x_val;
    std::vector<double> y_val;

    for (int i = 0; i <= N; i++) {
        x_val.push_back( (x_min + x_max)/2 + (x_max - x_min)*cos( (2*i+1)*3.1415/(2.0*N+2.0) )/2.0);
        y_val.push_back(finp(x_val[i]));
    }

    std::vector<std::vector<double>> Mat(N + 1, std::vector<double>(N + 1, 0));
    std::vector<std::vector<double>> b(N + 1, std::vector<double>(1, 0));


    for (int i = 0; i < x_val.size(); i++) {
        for (int j = 0; j < x_val.size(); j++) {
            Mat[i][j] = pow(x_val[i], j);
        }
        b[i][0] = y_val[i];
    }

    MatrixClass M(Mat);
    MatrixClass F(b);
    GaussClass sol(M, F);
    sol.solve();
    sol.getX().show();

    std::vector<double> tmp(N + 1, 0);
    for (int i = 0; i <= N; i++) {
        tmp[i] = sol.getX().Matrix[i][0];
    }

    Polinomial pol(tmp);
    double h = (x_max - x_min) / (N2 - 1);
    std::vector<double> err;
    for (double x = x_min; x <= x_max; x += h) {
        err.push_back(abs(pol.getValue(x) - finp(x)));
    }
    double delta = *std::max_element(err.begin(), err.end());
    return delta;
}

void DLF() {
    int N = 2;
    std::vector<double> x_val(N + 1, 0);
    std::vector<double> y_val(N + 1, 0);
    double x = 1;
    for (int i = 0; i < 3; i++) {
        x_val[i] = x;
        y_val[i] = atan(x);
        x++;
    }

    std::vector<std::vector<double>> Mat(N + 1, std::vector<double>(N + 1, 0));
    std::vector<std::vector<double>> b(N + 1, std::vector<double>(1, 0));

    for (int i = 0; i < 3; i++) {
        Mat[i][0] = x_val[i];
        Mat[i][1] = 1;
        Mat[i][2] = -x_val[i] * y_val[i];
        b[i][0] = y_val[i];
    }

    MatrixClass M(Mat); M.show();
    MatrixClass F(b); F.show();

    GaussClass sol(M, F);
    sol.solve();

    sol.getX().show();

    for (int i = 0; i < x_val.size(); i++) {
        for (int j = 0; j < x_val.size(); j++) {
            Mat[i][j] = pow(x_val[i], j);
        }
        b[i][0] = y_val[i];
    }

    MatrixClass M2(Mat); M2.show();
    MatrixClass F2(b); F2.show();

    GaussClass sol2(M2, F2);
    sol2.solve();

    sol2.getX().show();
}

void splain(int n, int ppp) {
    double x_min = -2;
    double x_max = 3;

    std::vector<double> x_val;
    std::vector<double> y_val;
    int N = n - 1;
    double h = (x_max - x_min) / N;
    double x = x_min;
    for (int i = 0; i <= N; i++) {
        x_val.push_back(x);
        y_val.push_back(finp(x));
        x += h;
        //std::cout << x << std::endl;
    }

    std::vector<std::vector<double>> Mat(n, std::vector<double>(n, 0));
    std::vector<std::vector<double>> B(n, std::vector<double>(1, 0));

    Mat[0][0] = -1.0 / h;   Mat[0][1] = 1.0 / h + 1.0 / h;  Mat[0][2] = -1.0 / h; B[0][0] = 0;
    Mat[N][N - 2] = 1.0 / h;  Mat[N][N - 1] = -2.0 / h;     Mat[N][N] = 1.0 / h; B[N][0] = 0;
    for (int i = 2; i <= N; i++) {
        B[i - 1][0] = 3 * (y_val[i] - 2 * y_val[i - 1] + y_val[i - 2]) / h;
        Mat[i - 1][i - 1 - 1] = h;   Mat[i - 1][i - 1] = 4 * h;  Mat[i - 1][i + 1 - 1] = h;
    }

    MatrixClass M(Mat); M.show();
    MatrixClass F(B); F.show();
    GaussClass sol(M, F);
    sol.solve();
    std::cout << "C: " << std::endl;
    sol.getX().show();
    std::vector<std::vector<double>> a(N, std::vector<double>(1, 0)), b(N, std::vector<double>(1, 0)), d(N, std::vector<double>(1, 0));
    for (int i = 0; i < N; i++) {
        a[i][0] = y_val[i];
    }

    for (int i = 1; i <= N; i++) {
        d[i - 1][0] = (sol.getX().Matrix[i][0] - sol.getX().Matrix[i - 1][0]) / h / 3;

        b[i - 1][0] = (y_val[i] - y_val[i - 1]) / h - (sol.getX().Matrix[i][0] + 2 * sol.getX().Matrix[i - 1][0]) * h / 3;
    }
    MatrixClass A(a), B2(b), D(d);
    std::cout << "A: " << std::endl; A.show();
    std::cout << "B: " << std::endl; B2.show();
    std::cout << "D: " << std::endl; D.show();
    std::cout << "Spline"<<ppp<<"[x_] : = Piecewise[{";
    for (int i = 0; i < a.size(); i++) {
        std::cout << "{";
        std::cout << a[i][0] << "+" << b[i][0] << "*(" << "x - " << x_val[i] << ")+" << sol.getX().Matrix[i][0] << "*(" << "x - " << x_val[i] << ")^2+" << d[i][0] << "*(" << "x - " << x_val[i] << ")^3, ";
        std::cout << x_val[i] << "<= x <=" << x_val[i + 1] << "}," << std::endl;
    }
    std::cout << "}];"<<std::endl;
}

void splainqw(int n, int ppp) {
    double x_min = -2;
    double x_max = 3;

    std::vector<double> x_val;
    std::vector<double> y_val;
    int N = n - 1;
    double h = (x_max - x_min) / N;
    double x = x_min;
    for (int i = 0; i <= N; i++) {
        x_val.push_back(x);
        y_val.push_back(finp(x));
        x += h;
        //std::cout << x << std::endl;
    }

    std::vector<std::vector<double>> Mat(n, std::vector<double>(n, 0));
    std::vector<std::vector<double>> B(n, std::vector<double>(1, 0));

    //Mat[0][0] = -1.0 / (2 * h);   Mat[0][1] = 2/(2 * h);  Mat[0][2] = -1.0 / (2 * h); B[0][0] = 0;
    Mat[N][N - 2] = 1.0 / (2*h);  Mat[N][N - 1] = -2.0 / (2*h);     Mat[N][N] = 1/(2 * h); B[N][0] = 0;

    for (int i = 1; i <= N; i++) {
        //std::cout << i << "\n";
        B[i-1][0] = 2 * (y_val[i]-y_val[i-1]) / h;
        Mat[i-1][i - 1] = 1;   Mat[i-1][i + 1 -1] = 1;
    }

    MatrixClass M(Mat); M.show();
    MatrixClass F(B); F.show();
    GaussClass sol(M, F);
    sol.solve();
    std::cout << "B: " << std::endl;
    sol.getX().show();
    std::vector<std::vector<double>> a(N, std::vector<double>(1, 0)), c(N, std::vector<double>(1, 0));
    for (int i = 0; i < N; i++) {
        a[i][0] = y_val[i];
    }

    for (int i = 1; i <= N; i++) {
        //c[i - 1][0] = (sol.getX().Matrix[i][0] - sol.getX().Matrix[i - 1][0]) / h / 2;
        c[i - 1][0] = (y_val[i] - y_val[i - 1]) / h / h  - sol.getX().Matrix[i-1][0]/h;
    }
    MatrixClass A(a), C(c);
    //std::cout << "A: " << std::endl; A.show();
    //std::cout << "B: " << std::endl; B2.show();
    //std::cout << "D: " << std::endl; D.show();
    std::cout << "SplineQW" << "[x_] := Piecewise[{";
    for (int i = 0; i < a.size(); i++) {
        std::cout << "{";
        std::cout << a[i][0] << "+" << sol.getX().Matrix[i][0] << "*(" << "x - " << x_val[i] << ")+" << c[i][0] << "*(" << "x - " << x_val[i] << ")^2,";
        std::cout << x_val[i] << "<= x <=" << x_val[i + 1] << "}," << std::endl;
    }
    std::cout << "}];" << std::endl;
}

int main()
{
    std::cout << "--------------------------\n";
    srand(time(NULL));
    splain(11, 3);
    
}
