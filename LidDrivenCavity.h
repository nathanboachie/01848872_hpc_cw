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
    void WriteSolution(std::string file, int rank, int* dim_prcs, int coords[2], MPI_Comm comm);
    void PrintConfiguration(int size);
    
    //Getters for testing
    double Get_Lx() const {return Lx;}
    double Get_Ly() const {return Ly;}
    int Get_Nx() const {return Nx;}
    int Get_Ny() const {return Ny;}
    double Get_Dt() const {return dt;}
    double Get_T() const {return T;}
    double Get_Re() const {return Re;}
    
private:
    double* v   = nullptr;
    double* vnew = nullptr;
    double* vsolve = nullptr;
    double* ssolve = nullptr;
    double* s   = nullptr;
    double* tmp = nullptr;
    double* in_grd_x = nullptr;
    double* out_grd_x = nullptr;
    double* in_grd_y = nullptr;
    double* out_grd_y = nullptr;
    double* in_grd_x2 = nullptr;
    double* out_grd_x2 = nullptr;
    double* in_grd_y2 = nullptr;
    double* out_grd_y2 = nullptr;
    double* in_grd_x3 = nullptr;
    double* out_grd_x3 = nullptr;
    double* in_grd_y3 = nullptr;
    double* out_grd_y3 = nullptr;
    int* dim_prcs = nullptr;
    

    double dt;
    double T ;
    double dx;
    double dy;
    int    Nx ;
    int    Ny ;
    int    Npts;
    double Lx ;
    double Ly ;
    double Re ;
    double U = 1.0 ;
    double nu ;
    int rank;
    int size;
    int left;
    int up; 
    int right;
    int down;
    MPI_Comm comm;
    int coords[2];
    

    
    SolverCG* cg = nullptr;

    void CleanUp();
    void UpdateDxDy();
    void Advance(MPI_Comm comm, int left, int right, int up, int down, int rank);
    void BinaryWrite(const std::string& filename, int* dim_prcs, double* local_array, int coords[2]);
    void readbinary(const std::string& filename, double* global_array, size_t array_size);
};
