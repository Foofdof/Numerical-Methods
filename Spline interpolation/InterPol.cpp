#include "InterPol.h"

double InterPol::getCk(int k) {
    double Ck = 0;
    for (int i = 0; i <= k; i++) {
        double t = 1;
        for (int j = 0; j <= k; j++) {
            if (i != j) {
                t *= x_val[i] - x_val[j];
            }
        }
        Ck += y_val[i] / t;
    }
    return Ck;
}

double InterPol::findW(int k){
    double w = 1;
    int N = x_val.size();
    for (int i = 0; i < N; i++) {
        if (i != k) {
            w *= (x - x_val[i]) / (x_val[k] - x_val[i]);
        }
        else {
            continue;
        }
    }
    return w;
}

double InterPol::solve(int f, double x) {
    if (f != this->f) {
        this->f = f;
    }
    if (x != this->x) {
        this->x = x;
    }
    //Lagrn.
    if (f == 0) {
        double L = 0;
        for (int k = 0; k < x_val.size(); k++) {
            L += y_val[k] * this->findW(k);
        }
        return L;
    }//Newt.
    else {
        double P = 0;
        for (int i = 0; i < x_val.size(); i++) {
            double phi = 1;
            for (int j = 0; j < i; j++) {
                phi *= x - x_val[j];
            }
            P += phi * this->getCk(i);
        }
        return P;
    }
    

}