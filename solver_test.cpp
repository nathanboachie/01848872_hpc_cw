
#define BOOST_TEST_MODULE Solvertest
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <mpi.h>
using namespace std;
#include <omp.h>
#include <cblas.h>
#include <fstream>

#include "LidDrivenCavity.h"
#include "SolverCG.h"

//Macros
#define IDX(I,J) ((J)*Nx + (I))
struct MPIFixture {
    public:
        explicit MPIFixture() {
            argc = boost::unit_test::framework::master_test_suite().argc;
            argv = boost::unit_test::framework::master_test_suite().argv;
            cout << "Initialising MPI" << endl;
            MPI_Init(&argc, &argv);
        }

        ~MPIFixture() {
            cout << "Finalising MPI" << endl;
            MPI_Finalize();
        }

        int argc;
        char **argv;
};
BOOST_GLOBAL_FIXTURE(MPIFixture);

//Setter + Getter testing
BOOST_AUTO_TEST_CASE(Lid_Driven_Cavity)
{
        //Bringing Values into scope
    int rank, size ;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    int periodic[2] = {0,0};
    double Lx = 1;
    double Ly = 1;
    int Nx = 9;
    int Ny = 9;
    double dt = 0.002;
    double T = 0.1;
    double Re = 10;
    //Cartesian Grid implementation
    int* dim_prcs = new int[2]();
    MPI_Dims_create(size,2,dim_prcs);
    MPI_Comm cartesian_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_prcs, periodic, 1, &cartesian_comm);
    int coords[2];
    //Coordinate implementation
    MPI_Comm_rank(cartesian_comm,&rank);
    //MPI_Cart_rank(cartesian_comm, coords, &rank);
    int left;
    int right;
    int down;
    int up;
    MPI_Cart_shift(cartesian_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cartesian_comm, 1, 1, &down, &up);
    MPI_Cart_coords(cartesian_comm,rank,2,coords);

    //Load Balancing
    int loc_nx;
    int loc_ny;
    int k_x;
    int k_y;
    int r_x = Nx % dim_prcs[0];
    k_x     = (Nx - r_x) / dim_prcs[0];
    if (coords[0] < (Nx % dim_prcs[0])) {           
        k_x++;
    }
    loc_nx = k_x;
    int r_y = Ny % dim_prcs[1];      
    k_y    = (Ny - r_y) / dim_prcs[1];
    if (coords[1] < (Ny % dim_prcs[1])) {           
        k_y++;
    }  
    loc_ny = k_y;
    double loc_lx = Lx/dim_prcs[0];
    double loc_ly = Ly/dim_prcs[1];

    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(loc_lx,loc_ly);
    solver->SetGridSize(loc_nx,loc_ny);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);

    double Lx_test = solver->Get_Lx();
    double Ly_test = solver->Get_Ly();
    int Nx_test = solver->Get_Nx();
    int Ny_test = solver->Get_Ny();
    double dt_test = solver->Get_Dt();
    double T_test = solver->Get_T();
    double Re_test = solver->Get_Re();

    //Testing Setters and Getters in LidDrivenCacity
    BOOST_TEST(Lx_test == Lx);
    BOOST_TEST(Ly_test == Ly);
    BOOST_TEST(Nx_test == Nx);
    BOOST_TEST(Ny_test == Ny);
    BOOST_TEST(dt_test == dt);
    BOOST_TEST(T_test == T);
    BOOST_TEST(Re_test == Re);
    cout << "test passed" << endl;
}

//Testing difference in errors between analytical and numerical solutions
BOOST_AUTO_TEST_CASE(SolverCG)
{
        //Bringing Values into scope
    int rank, size ;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    int periodic[2] = {0,0};
    double Lx = 1;
    double Ly = 1;
    int Nx = 9;
    int Ny = 9;
    double dt = 0.002;
    double T = 0.1;
    double Re = 10;
    //Cartesian Grid implementation
    int* dim_prcs = new int[2]();
    MPI_Dims_create(size,2,dim_prcs);
    MPI_Comm cartesian_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_prcs, periodic, 1, &cartesian_comm);
    int coords[2];
    //Coordinate implementation
    MPI_Comm_rank(cartesian_comm,&rank);
    //MPI_Cart_rank(cartesian_comm, coords, &rank);
    int left;
    int right;
    int down;
    int up;
    MPI_Cart_shift(cartesian_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cartesian_comm, 1, 1, &down, &up);
    MPI_Cart_coords(cartesian_comm,rank,2,coords);

    //Load Balancing
    int loc_nx;
    int loc_ny;
    int k_x;
    int k_y;
    int r_x = Nx % dim_prcs[0];
    k_x     = (Nx - r_x) / dim_prcs[0];
    if (coords[0] < (Nx % dim_prcs[0])) {           
        k_x++;
    }
    loc_nx = k_x;
    int r_y = Ny % dim_prcs[1];      
    k_y    = (Ny - r_y) / dim_prcs[1];
    if (coords[1] < (Ny % dim_prcs[1])) {           
        k_y++;
    }  
    loc_ny = k_y;
    double loc_lx = Lx/dim_prcs[0];
    double loc_ly = Ly/dim_prcs[1];

    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(loc_lx,loc_ly);
    solver->SetGridSize(loc_nx,loc_ny);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);
    solver->Initialise();
    double dx = loc_nx/Lx;
    double dy = loc_ny/Ly;

    double* v = new double[(loc_nx)*(loc_ny)];
    double* s = new double[(loc_nx)*(loc_ny)];
    double* s_aly = new double[(loc_nx)*(loc_ny)];
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < loc_nx; ++i) {
        for (int j = 0; j < loc_ny; ++j) {
            v[IDX(i,j)] = -M_PI * M_PI * (k * k + l * l)
                                       * sin(M_PI * k * i * dx)
                                       * sin(M_PI * l * j * dy);
        }
    }
    cg->Solve(v,s,cartesian_comm,left,right,up,down,rank);
    for(int i = 0; i < loc_nx; ++i)
    {
        for(int j = 0; j < loc_ny ; ++j)
        {
            s_aly[IDX(i,j)] = (-1.0/(M_PI*M_PI*2.0))*(sin(M_PI*dx))*(sin(M_PI)*dy);
        }
    }
    cblas_daxpy(loc_nx*loc_ny,-1.0,s,1.0,s_aly,1.0);
    double err = cblas_dnrm2(loc_nx*loc_ny,s_aly,1.0);
    MPI_Allreduce(&err, &err, 1, MPI_DOUBLE, MPI_SUM, cartesian_comm);
    double tol = 1e-6;
    BOOST_TEST(err < tol)    



}