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

    void Solve(double* b, double* x,MPI_Comm comm, int left, int right, int up, int down, int rank);
    

private:
    double dx;
    double dy;
    int Nx;
    int Ny;
    double* r;
    double* p;
    double* z;
    double* t;
    int* dim_prcs;
    int rank;
    int left;
    int right; 
    int up;
    int down;
    double U    = 1.0;
    double* xsolve = nullptr;
    double* psolve = nullptr;
    double* tsolve = nullptr;
    double* tsolve1 = nullptr;
    MPI_Comm comm;
    double* in_cg_x = nullptr;
    double* out_cg_x = nullptr;
    double* in_cg_y = nullptr;
    double* out_cg_y = nullptr;

    void ApplyOperator(double* p, double* t, MPI_Comm comm, int left, int right, int up, int down);
    void Precondition(double* p, double* t, int left, int right, int up, int down);
    void ImposeBC(double* p, int left, int right, int up, int down);

};

