// ldc_cg_test.cpp

#include "LidDrivenCavity.h"
#include "SolverCG.h"
#define BOOST_TEST_MODULE ldc_cg_test
#include <boost/test/included/unit_test.hpp>
#define IDX(I,J) ((J)*Nx + (I))

BOOST_AUTO_TEST_SUITE(unit_test)

BOOST_AUTO_TEST_CASE(solver_cg)
{
    //Creation of test case parameters to define domain
    int Nx = 10;
    int Ny = 10;
    int pts = Nx*Ny;
    int Lx = 1.0;
    int Ly = 1.0;
    double dx = Lx / (Nx-1);
    double dy = Ly / (Ny-1);

    //Create class object
    SolverCG solver_cg_obj(Nx,Ny,dx,dy);
    //Memory allocation
    double* v = new double[pts];
    double* s = new double[pts];
    double* s_aly = new double[pts];

    //Analytic Solutions
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < Nx; ++i) 
    {
        for (int j = 0; j < Ny; ++j) 
        {
            v[IDX(i,j)] = -M_PI*M_PI*(k*k+l*l)*sin(M_PI*k*i*dx)*sin(M_PI*l*j*dy);
            s_aly[IDX(i,j)] = -1.0/(2.0*M_PI*M_PI)*sin(k*M_PI*i*dx)*sin(l*M_PI*M_PI*j*dy);
        }
    }
    //Poisson solver 
    solver_cg_obj.Solve(v,s);
    

}

BOOST_AUTO_TEST_SUITE_END();
