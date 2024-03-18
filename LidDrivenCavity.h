#pragma once

#include <string>
using namespace std;

class SolverCG;

/**
 * @class LidDrivenCavity
 * @brief Includes methods specific to assigning domain size, parameters and methods 
 * for advancing solution in time and writing solution to files and terminal
*/
class LidDrivenCavity
{
public:
    LidDrivenCavity();
    ~LidDrivenCavity();

    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);

    void Initialise();
    void Integrate(MPI_Comm comm, int left, int right, int up, int down, int rank);
    void WriteSolution(std::string file);
    void PrintConfiguration();
    int* dim_prcs;
    int rank;
    int size;
    int left;
    int up; 
    int right;
    int down;
    MPI_Comm comm;
    int size_grid;
    int coords[2];
    

private:
    double* v   = nullptr;
    double* vnew = nullptr;
    double* vsolve = nullptr;
    double* ssolve = nullptr;
    double* s   = nullptr;
    double* tmp = nullptr;
    

    double dt   = 0.01;
    double T    = 1.0;
    double dx;
    double dy;
    int    Nx   = 9;
    int    Ny   = 9;
    int    Npts = 81;
    double Lx   = 1.0;
    double Ly   = 1.0;
    double Re   = 10;
    double U    = 1.0;
    double nu   = 0.1;
    double* in_grd_x = nullptr;
    double* out_grd_x = nullptr;
    double* in_grd_y = nullptr;
    double* out_grd_y = nullptr;

    

    
    SolverCG* cg = nullptr;

    void CleanUp();
    void UpdateDxDy();
    void Advance(MPI_Comm comm, int left, int right, int up, int down, int rank);
};

