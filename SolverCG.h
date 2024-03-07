#pragma once

/**
 * @class SolverCG 
 * @brief Includes method calling conjugate gradient algorithm to solve a linear matrix system
 * along with others for preconditioning and discretising methods, along with domain parameters
*/
class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy);
    ~SolverCG();

    void Solve(double* b, double* x);

private:
    double dx;
    double dy;
    int Nx;
    int Ny;
    double* r;
    double* p;
    double* z;
    double* t;

    void ApplyOperator(double* p, double* t);
    void Precondition(double* p, double* t);
    void ImposeBC(double* p);

};

