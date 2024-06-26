#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>
#include <omp.h>

#include "SolverCG.h"

/**
 * @brief Macro maps 2d coordinates to 1d index in array (Matrix --> flattened array)
*/
#define IDX(I,J) ((J)*Nx + (I))
#define IDX_p(I,J) ((J)*(Nx+2) + (I))
/**
 * @brief Constructor of SolverCG class
 * @param pNx Number of grid points in x direction
 * @param pNy Number of grid points in y direction
 * @param pdx Grid spacing in x direction 
 * @param pdy Grid spacing in y direction
*/
SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    int n = Nx*Ny;
    r = new double[n];
    p = new double[n];
    z = new double[n];
    t = new double[n]; //temp
    xsolve = new double[(Nx+2)*(Ny+2)]();
    tsolve = new double[(Nx+2)*(Ny+2)]();
    psolve = new double[(Nx+2)*(Ny+2)]();
    tsolve1 = new double[(Nx+2)*(Ny+2)]();
    in_gc_x = new double[Ny];
    in_gc_y = new double[Nx];
    out_gc_x = new double[Ny];
    out_gc_y = new double[Nx];
}

/**
 * @brief Destructor of SolverCG class
*/
SolverCG::~SolverCG()
{
    delete[] r;
    delete[] p;
    delete[] z;
    delete[] t;
    delete[] xsolve;
    delete[] psolve;
    delete[] tsolve;
    delete[] tsolve1;
    delete[] in_gc_x;
    delete[] out_gc_x;
    delete[] in_gc_y;
    delete[] out_gc_y;
}

/**
 * @brief Applies Conjugate gradient method to solve Poisson's equation, using cblas library
 * @param b is the output vector in Ax = b
 * @param x is the vector to be solved 
 * @param comm Cartesian communicator
 * @param left Source rank x direction
 * @param right Destination rank x direction
 * @param up Destination rank y direction
 * @param down source rank x direction
*/
void SolverCG::Solve(double* b, double* x, MPI_Comm comm, int left, int right, int up, int down, int rank) {
    unsigned int n = Nx*Ny;
    int k;
    double alpha;
    double beta;
    double eps;
    double tol = 0.001;

    eps = cblas_dnrm2(n, b, 1);//Matrix norm
    MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_SUM, comm);
    // If error below tolerance create a 0 vector for x0
    if (eps < tol * tol) {
        std::fill(x, x + n, 0.0);
        if (rank == 0) 
        {
            std::cout << "Norm is " << eps << std::endl;
        }
        return;
    }   
    //
    //Padding for apply operator
    for(int j = 1; j <= Ny ; ++j)
    {
        for(int i = 1 ; i <= Nx ; ++i)
        {
            xsolve[IDX_p(i,j)] = x[IDX(i-1,j-1)];
        }
    }
    for(int j = 1; j <= Ny ; ++j)
    {
        for(int i = 1 ; i <= Nx ; ++i)
        {
            tsolve[IDX_p(i,j)] = t[IDX(i-1,j-1)];
        }
    }

    
    ApplyOperator(xsolve, tsolve, comm, left, right, up, down);
    //Removing padding
    for(int j = 1; j <= Ny ; ++j)
    {
        for(int i = 1; i <= Nx ; ++i)
        {
            x[IDX(i-1,j-1)] = xsolve[IDX_p(i,j)];
        }
    }

    for(int j = 1; j <= Ny ; ++j)
    {
        for(int i = 1; i <= Nx ; ++i)
        {
            t[IDX(i-1,j-1)] = tsolve[IDX_p(i,j)];
        }
    }
    

    cblas_dcopy(n, b, 1, r, 1);  // r_0 = b (i.e. b)
    ImposeBC(r, left, right, up , down);
   
    cblas_daxpy(n, -1.0, t, 1, r, 1);
    Precondition(r, z, left, right, up, down);
    
    
    // Assuming you want to print the contents of the 'out' vector with a left shift
    cblas_dcopy(n, z, 1, p, 1);// p_0 = r_0 

    k = 0;
    do {
        k++;
        // Perform action of Nabla^2 * p
        for(int j = 1; j <= Ny ; ++j)
        {
            for(int i = 1 ; i <= Nx ; ++i)
            {
                psolve[IDX_p(i,j)] = p[IDX(i-1,j-1)];
            }
        }
        
        
        for(int j = 1; j <= Ny ; ++j)
        {   
            for(int i = 1 ; i <= Nx ; ++i)
            {
                tsolve[IDX_p(i,j)] = t[IDX(i-1,j-1)];
            }
         }
        
        ApplyOperator(psolve, tsolve, comm, left, right, up, down);
        
        for(int j = 1; j <= Ny ; ++j)
        {
            for(int i = 1; i <= Ny ; ++i)
            {
                p[IDX(i-1,j-1)] = psolve[IDX_p(i,j)];
            }
        }

        for(int j = 1; j <= Ny ; ++j)
        {
            for(int i = 1; i <= Nx ; ++i)
            {
                t[IDX(i-1,j-1)] = tsolve[IDX_p(i,j)];
            }
        }
       
        
        
        //ALl reduce to make sure its the global alpha and beta being edited 
        alpha = cblas_ddot(n, t, 1, p, 1);// alpha = p_k^T A p_k
        MPI_Allreduce(&alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha = cblas_ddot(n, r, 1, z, 1) / alpha; // compute alpha_k 
        MPI_Allreduce(&alpha, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        beta  = cblas_ddot(n, r, 1, z, 1); // z_k^T r_k
        MPI_Allreduce(&beta, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cblas_daxpy(n,  alpha, p, 1, x, 1); // x_{k+1} = x_k + alpha_k p_k
        cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k
        eps = cblas_dnrm2(n, r, 1);
        MPI_Allreduce(&eps, &eps, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (eps < tol*tol) {
            break;
        }
        Precondition(r, z, left, right, up, down);
        beta = cblas_ddot(n, r, 1, z, 1) / beta;
        MPI_Allreduce(&beta, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       
        cblas_dcopy(n, z, 1, t, 1);
        cblas_daxpy(n, beta, p, 1, t, 1);
        cblas_dcopy(n, t, 1, p, 1);
    
    } while (k < 5000); // Set a maximum number of iterations

    if (k == 5000) 
    {
        if(rank == 0)
        {
            cout << "FAILED TO CONVERGE" << endl;
        }
        MPI_Finalize();
        exit(-1);
    }
    if(rank == 0)
    {
        cout << "Converged in " << k << " iterations. eps = " << eps << endl;
    }
}

/**
 * @brief ApplyOperator applies second order finite difference discretisation to input and output vectors
 * @param in Input vector 
 * @param out Output vector
 * @param comm Cartesian communicator
 * @param left Source rank x direction
 * @param right Destination rank x direction
 * @param up Destination rank y direction
 * @param down source rank x direction
*/
void SolverCG::ApplyOperator(double* in, double* out, MPI_Comm comm, int left, int right, int up, int down) {
    // Assume ordered with y-direction fastest (column-by-column)
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    int start_x_cg;
    int start_y_cg;
    int end_x_cg;
    int end_y_cg;  

    int jm1 = (down == MPI_PROC_NULL) ? 1: 0;
    int jp1 = (down == MPI_PROC_NULL) ? 3: 2;
    start_y_cg = (down == MPI_PROC_NULL) ? 2 : 1;
    end_y_cg = (up == MPI_PROC_NULL) ? Ny: Ny+1;
    end_x_cg = (right == MPI_PROC_NULL) ? Nx: Nx+1;
    start_x_cg = (left == MPI_PROC_NULL) ? 2 : 1;

    //Sending information from left guard cell to right
    for(int j = 1; j <=Ny; ++j)
    {
        in_gc_x[j-1] = in[IDX_p(1,j)];
        out_gc_x[j-1] = in[IDX_p(Nx+1,j)];
    }
    MPI_Sendrecv(in_gc_x,Ny,MPI_DOUBLE,left,0,out_gc_x,Ny,MPI_DOUBLE,right,0,comm,MPI_STATUS_IGNORE);
    for(int j = 1; j<=Ny;++j)
    {
        in[IDX_p(Nx+1,j)] = out_gc_x[j-1];
    }

    //Right gc to left 
    for(int j = 1; j <= Ny; ++j)
    {
        in_gc_x[j-1] = in[IDX_p(Nx,j)];
        out_gc_x[j-1] = in[IDX_p(0,j)];
    }
    MPI_Sendrecv(in_gc_x,Ny,MPI_DOUBLE,right,0,out_gc_x,Ny,MPI_DOUBLE,left,0,comm,MPI_STATUS_IGNORE);
    for(int j = 1; j<=Ny;++j)
    {
        in[IDX_p(0,j)] = out_gc_x[j-1];
    }

    //down to up
    for(int i = 1; i<=Nx;++i)
    {
        in_gc_y[i-1] = in[IDX_p(i,1)];
        out_gc_y[i-1] = in[IDX_p(i,Ny+1)];
    }
    MPI_Sendrecv(in_gc_y,Nx,MPI_DOUBLE,down,0,out_gc_y,Nx,MPI_DOUBLE,up,0,comm,MPI_STATUS_IGNORE);
    for(int i = 1; i<=Nx ; ++i)
    {
        in[IDX_p(i,Ny+1)] = out_gc_y[i-1];
    }

    //Up to Down
    for(int i =1; i<=Nx;++i)
    {
        in_gc_y[i-1] = in[IDX_p(i,Ny)];
        out_gc_y[i-1] = in[IDX_p(i,0)];
    }
    MPI_Sendrecv(in_gc_y,Nx,MPI_DOUBLE,up,0,out_gc_y,Nx,MPI_DOUBLE,down,0,comm,MPI_STATUS_IGNORE);
    for(int i = 1; i<=Nx;++i)
    {
        in[IDX_p(i,0)] = out_gc_y[i-1];
    }

    
    #pragma omp parallel for shared(in, out, dx2i, dy2i, jm1, jp1) schedule(static)
    for(int j = start_y_cg ; j < end_y_cg ; ++j)
    {
        for(int i = start_x_cg; i < end_x_cg; ++i)
        {
            out[IDX_p(i,j)] = ( -     in[IDX_p(i-1, j)]
                                + 2.0*in[IDX_p(i,   j)]
                                -     in[IDX_p(i+1, j)])*dx2i
                            + ( -     in[IDX_p(i, jm1)]
                                + 2.0*in[IDX_p(i,   j)]
                                -     in[IDX_p(i, jp1)])*dy2i;
        }
        jm1++;
        jp1++;
    }
   
    
}
    




/**
 * @brief Preconditioning to speed up convergence, better solution representation
 * @param in Input vector (matrix)
 * @param out Output vector (vector)
 * @param comm Cartesian communicator
 * @param left Source rank x direction
 * @param right Destination rank x direction
 * @param up Destination rank y direction
 * @param down source rank x direction
*/
void SolverCG::Precondition(double* in, double* out, int left, int right, int up, int down) {

    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
    double factor = 2.0*(dx2i + dy2i);
    

    int start_y_pr = (down == MPI_PROC_NULL) ? 1 : 0;
    int end_y_pr = (up == MPI_PROC_NULL) ? (Ny-1) : (Ny);
    int end_x_pr = (right == MPI_PROC_NULL) ? (Nx-1) : (Nx);
    int start_x_pr = (left == MPI_PROC_NULL) ? 1 : 0;
   
    
    #pragma omp parallel for collapse(2) shared(in, out, factor) schedule(static)
    for (int j = start_y_pr; j < end_y_pr; ++j) {
        for (int i = start_x_pr; i < end_x_pr; ++i) {
            out[IDX(i, j)] = in[IDX(i, j)] / factor;
        }
    }

    
    
    if (down == MPI_PROC_NULL) 
    {   
        #pragma omp parallel for
        for (int i = 0; i < Nx; ++i) 
        {
            // bottom boundary condition
            out[IDX(i, 0)] = in[IDX(i,0)];
        }
    }

    // Assigning top boundary condition
    if (up == MPI_PROC_NULL) 
    {   
        #pragma omp parallel for
        for (int i = 0; i < Nx ; ++i) {
            // top boundary condition
            out[IDX(i, Ny-1)] = in[IDX(i,Ny-1)];
        }
    }

    // Assigning right boundary condition
    if (right == MPI_PROC_NULL) 
    {
        #pragma omp parallel for
        for (int j = 0 ; j < Ny; ++j) {
            // right boundary condition
            out[IDX(Nx-1, j)] = in[IDX(Nx-1,j)];
        }
    }

    // Assigning left boundary condition
    if (left == MPI_PROC_NULL) 
    {
        #pragma omp parallel for
        for (int j = 0; j < Ny;  ++j) {
            // left boundary condition
            out[IDX(0, j)] = in[IDX(0,j)];
        }
    }
    
    }

/**
 * @brief Applying boundary conditions on vectors in specific directions
 * @param inout Vector in which bcs applied to, most likely velocity 
 * @param left Source rank x direction
 * @param right Destination rank x direction
 * @param up Destination rank y direction
 * @param down source rank x direction
*/
void SolverCG::ImposeBC(double* inout, int left, int right, int up, int down) {
        // Boundaries

    // Assigning bottom boundary condition 
    if (down == MPI_PROC_NULL) {
        #pragma omp parallel for
        for (int i = 0; i < Nx; ++i) {
            // bottom boundary condition
            inout[IDX(i, 0)] = 0.0; //2.0 * dy2i * inout[IDX(i, 0)] - inout[IDX(i, 1)];
        }
    }

    // Assigning top boundary condition
    if (up == MPI_PROC_NULL) {
        #pragma omp parallel for
        for (int i = 0; i < Nx; ++i) {
            // top boundary condition
            inout[IDX(i, Ny-1)] = 0.0; //2.0 * dy2i * inout[IDX(i, Ny-1)] - inout[IDX(i, Ny-2)] - 2.0 * dyi * U;
        }
    }

    // Assigning right boundary condition
    if (right == MPI_PROC_NULL) {
        #pragma omp parallel for
        for (int j = 0; j < Ny; ++j) {
            // right boundary condition
            inout[IDX(Nx-1, j)] = 0.0; //2.0 * dx2i * inout[IDX(Nx-1, j)] - inout[IDX(Nx-2, j)];
        }
    }

    // Assigning left boundary condition
    if (left == MPI_PROC_NULL) {
        #pragma omp parallel for
        for (int j = 0; j < Ny; ++j) {
            // left boundary condition
            inout[IDX(0, j)] = 0.0; //2.0 * dx2i * inout[IDX(0, j)] - inout[IDX(1, j)];
        }
    }
}

/**
 * @brief Function to replace cblas library, Inner product but multi threaded for openmp parallelisation
 * @param a Matrix 1
 * @param b Matrix 2
 * @param n Flattened length of matrices
 * @return Dot product of two vectors
*/
double SolverCG::OMP_InnerProd(double* a, double* b, int n) {
        double product = 0;
        int i;
        #pragma omp parallel for \
            default(shared) private(i) \
            schedule(static) \
            reduction(+:product)
          for(i = 0; i < n ; ++i)
          {
            product += a[i]*b[i];
          }

        return product;
}
/**
 * @brief Function to replace cblas library, Norm but multi threaded for openmp parallelisation
 * @param a Matrix 1
 * @param n Flattened length of matrices
 * @return Norm of vector
*/
double SolverCG::OMP_Norm(double* a, int n){
    double norm = 0;
    int i;
    #pragma omp parallel for \
        default(shared) private(i) \
        schedule(static) \
        reduction(+:norm)
      for(i = 0; i <n ; ++i)
      {
        norm+=sqrt(a[i]*a[i]);
      }
      return norm;
}
/**
 * @brief Copy vector into another of same size
 * @param a Source vector
 * @param b Destination vector
 * @param n Length of vector
*/
void SolverCG::OMP_Copy(double*a, double*b, int n){
    int i;
    #pragma omp parallel for \
        private(i)
    for(i = 0; i < n; ++i)
    {
        b[i] = a[i];
    }
}

/**
 * @brief Matrix addition, y = alpha*x +y
 * @param x Source vector
 * @param y Destination vector
 * @param alpha Factor 
 * @param n Length of vector
*/
void SolverCG::OMP_Daxpy(double*x , double*y, double alpha,int n){
    int i;
    #pragma omp parallel for \
        private(i)
    for(i = 0; i < n; ++i )
    {
        y[i]+=alpha*x[i];
    }
}