#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <mpi.h>
using namespace std;

#include <cblas.h>

/**
 * @brief Macro maps 2d coordinates to 1d index in array (Matrix --> flattened array)
*/
#define IDX(I,J) ((J)*Nx + (I))
#define IDX_p(I,J) ((J)*(Nx+2) + (I))
#define IDX_Global(I,J) ((J)*(Nx+2)*size + (I))

#include "LidDrivenCavity.h"
#include "SolverCG.h"

/**
 * @brief Constructor of LidDrivenCavity class
*/
LidDrivenCavity::LidDrivenCavity()
{
}

/**
 * @brief Destructor of LidDrivenCavity class
*/
LidDrivenCavity::~LidDrivenCavity()
{
    CleanUp();
}

/**
 * @brief setter for domain size 
 * @param xlen The length of the grid in the direction
 * @param ylen The length of the grid in the y direction
 
*/
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
    UpdateDxDy();
}

/**
 * @brief Setter for grid size 
 * @param nx The number of grid points in the x direction
 * @param ny The number of grid points in the y direction
*/
void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
    UpdateDxDy();
}

/** 
 * @brief Setter for number of time steps 
 * @param deltat Number of time steps
*/
void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

/**
 * @brief Setter for final time 
 * @param finalt Final time
*/
void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

/**
 * @brief Setter for Reynolds number and creates 1/reynolds number for numerical purposes
 * @param re Reynolds number
*/
void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0/re;
}

/**
 * @brief Initalising simulation and call numerical solver 
*/
void LidDrivenCavity::Initialise()
{
    CleanUp();

    v   = new double[(Nx+2)*(Ny+2)]();
    vnew = new double[(Nx+2)*(Ny+2)]();
    s   = new double[(Ny+2)*(Ny+2)]();
    tmp = new double[(Nx+2)*(Ny+2)]();
    vsolve = new double[Nx*Ny]();
    ssolve = new double[Nx*Ny]();
    cg  = new SolverCG(Nx, Ny, dx, dy);
    in_grd_x = new double[Ny]();
    out_grd_x = new double[Ny]();
    in_grd_y = new double[Nx]();
    out_grd_y = new double[Nx]();
}

/**
 * @brief Output number of steps and time of simulation, and step in time for vorticity, print on rank zero
*/
void LidDrivenCavity::Integrate(MPI_Comm comm, int left, int right, int up, int down, int rank)
{
    int NSteps = ceil(T/dt);
    for (int t = 0; t < NSteps; ++t)
    { 
            if(rank == 0)
            std::cout << "Step: " << setw(8) << t
                    << "  Time: " << setw(8) << t*dt
                    << std::endl;
            
    Advance(comm, left, right, up, down, rank);
    }
}

/**
 * @brief Writing simulation to output files for data analysis
*/
void LidDrivenCavity::WriteSolution(std::string file)
{   
    double* u0 = new double[(Nx+2)*(Ny+2)]();
    double* u1 = new double[(Nx+2)*(Ny+2)]();
    double* u0_global = new double[(Nx+2)*size*(Ny+2)*size]();
    double* u1_global = new double[(Nx+2)*size*(Ny+2)*size]();
    double* v_global = new double[(Nx+2)*size*(Ny+2)*size]();
    double* s_global = new double[(Nx+2)*size*(Ny+2)*size]();
    for (int i = 1; i < Nx+1; ++i) {
        for (int j = 1; j < Ny+1; ++j) {
            u0[IDX_p(i,j)] =  (s[IDX_p(i,j+1)] - s[IDX_p(i,j)]) / dy;
            u1[IDX_p(i,j)] = -(s[IDX_p(i+1,j)] - s[IDX_p(i,j)]) / dx;
        }
    }
    for (int i = 1; i < Nx+1; ++i) {
        u0[IDX_p(i,Ny)] = U;
    }


    MPI_Allgather(u0,(Nx+2)*(Ny+2), MPI_DOUBLE, u0_global,(Nx+2)*(Ny+2), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(u1,(Nx+2)*(Ny+2), MPI_DOUBLE, u1_global, (Nx+2)*(Ny+2), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(v,(Nx+2)*(Ny+2), MPI_DOUBLE, v_global, (Nx+2)*(Ny+2), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(s,(Nx+2)*(Ny+2), MPI_DOUBLE, s_global, (Nx+2)*(Ny+2), MPI_DOUBLE, MPI_COMM_WORLD);
    if(rank == 0)
    {
    std::ofstream f(file.c_str());
    std::cout << "Writing file " << file << std::endl;
    int k = 0;
    {
        for (int i = 1; i < (Nx)*size+1; ++i)
        {
            for (int j = 1; j < (Ny)*size+1; ++j)
            {
                k = IDX_Global(i, j);
                f << i * dx << " " << j * dy << " " << v_global[k] <<  " " << s_global[k] 
                << " " << u0_global[k] << " " << u1_global[k] << std::endl;
            }
            f << std::endl;
        }
        f.close();
    }
    }

    delete[] u0;
    delete[] u1;
    delete[] u0_global;
    delete[] u1_global;
    delete[] v_global;
    delete[] s_global;
}

/**
 * @brief Outputting configuration onto screen 
*/
void LidDrivenCavity::PrintConfiguration()
{
    {
        cout << "Grid size: " << Nx*size << " x " << Ny*size << endl;
        cout << "Spacing:   " << dx << " x " << dy << endl;
        cout << "Length:    " << Lx << " x " << Ly << endl;
        cout << "Grid pts:  " << Npts*size*size << endl;
        cout << "Timestep:  " << dt << endl;
        cout << "Steps:     " << ceil(T/dt) << endl;
        cout << "Reynolds number: " << Re << endl;
        cout << "Linear solver: preconditioned conjugate gradient" << endl;
        cout << endl;
        if (nu * dt / dx / dy > 0.25) 
        {
            cout << "ERROR: Time-step restriction not satisfied!" << endl;
            cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
            MPI_Finalize();
            exit(-1);
        }
    }
}

/**
 * @brief Memory management, deleting dynamically allocated arrays
*/
void LidDrivenCavity::CleanUp()
{
    if (v) {
        delete[] v;
        delete[] vnew;
        delete[] vsolve;
        delete[] ssolve;
        delete[] s;
        delete[] tmp;
        delete cg;
        delete[] in_grd_x;
        delete[] out_grd_x;
        delete[] in_grd_y;
        delete[] out_grd_y;
    }
}

/**
 * @brief Creating x and y grid spacing and number of grid points
*/
void LidDrivenCavity::UpdateDxDy()
{
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1);
    Npts = Nx * Ny;
}

/**
 * @brief Find vortcity at boundary and interior grid points then integrate and advance, call solver to integrate poission equation
*/
void LidDrivenCavity::Advance(MPI_Comm comm, int left, int right, int up, int down, int rank)
{
    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;

    

    // Compute interior vorticity
    for (int i = 1; i < Nx+1; ++i) {
        for (int j = 1; j < Ny+1; ++j) {
            v[IDX_p(i,j)] = dx2i*(
                    2.0 * s[IDX_p(i,j)] - s[IDX_p(i+1,j)] - s[IDX_p(i-1,j)])
                        + 1.0/dy/dy*(
                    2.0 * s[IDX_p(i,j)] - s[IDX_p(i,j+1)] - s[IDX_p(i,j-1)]);
        }
    }

    //Cartesian shift to apply boundary conditions 
    //Assigning bottom boundary condition 
    int start_x; 
    int start_y;
    int end_x;
    int end_y;

    start_y = (down == MPI_PROC_NULL) ? 2 : 1;
    end_y = (up == MPI_PROC_NULL) ? Ny: Ny + 1 ;
    end_x = (right == MPI_PROC_NULL) ? Nx  : Nx+1;
    start_x = (left == MPI_PROC_NULL) ? 2 : 1;
    
    if(down == MPI_PROC_NULL)
    {
        for (int i = 1; i < Nx+1; ++i) 
        {
        // bottom
            v[IDX_p(i,1)]    = 2.0 * dy2i * (s[IDX_p(i,1)]    - s[IDX_p(i,2)]);
        }
    }
    //Assiging top boundary condition
    if(up == MPI_PROC_NULL)
    {
        for (int i = 1; i < Nx+1; ++i) 
        {
        // top
            v[IDX_p(i,Ny)] = 2.0 * dy2i * (s[IDX_p(i,Ny)] - s[IDX_p(i,Ny-1)]) - 2.0 * dyi*U;
        }
    }
    
    //Assigning right boundary condition
    if(right == MPI_PROC_NULL)
    {
        for (int j = 1; j < Ny+1; ++j) 
        {
            // right
            v[IDX_p(Nx,j)] = 2.0 * dx2i * (s[IDX_p(Nx,j)] - s[IDX_p(Nx-1,j)]);
        }     
    }

    //Assigning left boundary condition
    if(left == MPI_PROC_NULL)
    {
        for (int j = 1; j < Ny+1; ++j) 
        {
            // left
            v[IDX_p(1,j)]    = 2.0 * dx2i * (s[IDX_p(1,j)]    - s[IDX_p(2,j)]);
        }
    }

    
    //Implementing guard cell around subdomain (repeated for each rank)
    //Sending information from left guard cell to right
    for(int j = 1; j <= Ny; ++j)
    {
        in_grd_x[j-1] = v[IDX_p(1,j)];
        out_grd_x[j-1] = v[IDX_p(Nx+1,j)];
    }
    MPI_Sendrecv(in_grd_x,Ny,MPI_DOUBLE,left,0,out_grd_x,Ny,MPI_DOUBLE,right,0,comm,MPI_STATUS_IGNORE);
    for(int j = 1; j < Ny +1 ; ++j)
    {
        v[IDX_p(Nx+1,j)] = out_grd_x[j-1];
    }


    //Right guard cell to left
    for(int j = 1; j <= Ny;++j)
    {
        in_grd_x[j-1] = v[IDX_p(Nx,j)];
        out_grd_x[j-1] = v[IDX_p(0,j)];
    }

    
    MPI_Sendrecv(in_grd_x,Ny,MPI_DOUBLE,right,0,out_grd_x,Ny,MPI_DOUBLE,left,0,comm,MPI_STATUS_IGNORE);
    for(int j = 1; j <=Ny;++j)
    {
        v[IDX_p(0,j)] = out_grd_x[j-1];
    }


    // In y direction bottom to top
    for(int i = 1; i <= Nx ; ++i)
    {
        in_grd_y[i-1] = v[IDX_p(i,1)];
        out_grd_y[i-1] = v[IDX_p(i,Ny+1)];
    }
    MPI_Sendrecv(in_grd_y,Nx,MPI_DOUBLE,down,0,out_grd_y,Nx,MPI_DOUBLE,up,0,comm,MPI_STATUS_IGNORE);
    for(int i = 1; i <= Nx;++i)
    {
        v[IDX_p(i,Ny+1)] = out_grd_y[i-1];
    }

    //Top to bottom
    for(int i = 1; i <= Nx ; ++i)
    {
        in_grd_y[i-1] = v[IDX_p(i,Ny)];
        out_grd_y[i-1] = v[IDX_p(i,0)];
    }
    MPI_Sendrecv(in_grd_y,Nx,MPI_DOUBLE,up,0,out_grd_y,Nx,MPI_DOUBLE,down,0,comm,MPI_STATUS_IGNORE);
    for(int i = 1; i<= Nx; ++i)
    {
        v[IDX_p(i,0)] = out_grd_y[i-1];
    }

    
    
    // Time advance vorticity
    for (int i = start_x; i < end_x; ++i) {
        for (int j = start_y; j < end_y; ++j) {
            v[IDX_p(i,j)] = v[IDX_p(i,j)] + dt*(
                ( (s[IDX_p(i+1,j)] - s[IDX_p(i-1,j)]) * 0.5 * dxi
                 *(v[IDX_p(i,j+1)] - v[IDX_p(i,j-1)]) * 0.5 * dyi)
              - ( (s[IDX_p(i,j+1)] - s[IDX_p(i,j-1)]) * 0.5 * dyi
                 *(v[IDX_p(i+1,j)] - v[IDX_p(i-1,j)]) * 0.5 * dxi)
              + nu * (v[IDX_p(i+1,j)] - 2.0 * v[IDX_p(i,j)] + v[IDX_p(i-1,j)])*dx2i
              + nu * (v[IDX_p(i,j+1)] - 2.0 * v[IDX_p(i,j)] + v[IDX_p(i,j-1)])*dy2i);
        }
    }

    // Assuming you want to print the contents of the 'out' vector
    

    //Repeating guard cell sendrecvs;
        //Implementing guard cell around subdomain (repeated for each rank)
    //Sending information from left guard cell to right
 
    for(int j = 1; j <= Ny; ++j)
    {
        in_grd_x[j-1] = v[IDX_p(1,j)];
        out_grd_x[j-1] = v[IDX_p(Nx+1,j)];
    }
    MPI_Sendrecv(in_grd_x,Ny,MPI_DOUBLE,left,0,out_grd_x,Ny,MPI_DOUBLE,right,0,comm,MPI_STATUS_IGNORE);
    for(int j = 1; j <= Ny ; ++j)
    {
        v[IDX_p(Nx+1,j)] = out_grd_x[j-1];
    }

    //Right guard cell to left
    for(int j = 1; j <= Ny;++j)
    {
        in_grd_x[j-1] = v[IDX_p(Nx,j)];
        out_grd_x[j-1] = v[IDX_p(0,j)];
    }
    MPI_Sendrecv(in_grd_x,Ny,MPI_DOUBLE,right,0,out_grd_x,Ny,MPI_DOUBLE,left,0,comm,MPI_STATUS_IGNORE);

    //Y directions
    for(int i = 1; i <= Nx ; ++i)
    {
        in_grd_y[i-1] = v[IDX_p(i,1)];
        out_grd_y[i-1] = v[IDX_p(i,Ny+1)];
    }
    MPI_Sendrecv(in_grd_y,Nx,MPI_DOUBLE,down,0,out_grd_y,Nx,MPI_DOUBLE,up,0,comm,MPI_STATUS_IGNORE);
    for(int i = 1; i <= Nx;++i)
    {
        v[IDX_p(i,Ny+1)] = out_grd_y[i-1];
    }


    for(int i = 1; i <= Nx ; ++i)
    {
        in_grd_y[i-1] = v[IDX_p(i,Ny)];
        out_grd_y[i-1] = v[IDX_p(i,0)];
    }
    MPI_Sendrecv(in_grd_y,Nx,MPI_DOUBLE,up,0,out_grd_y,Nx,MPI_DOUBLE,down,0,comm,MPI_STATUS_IGNORE);
    for(int i = 1; i<= Nx; ++i)
    {
        v[IDX_p(i,0)] = out_grd_y[i-1];
    }


    //Calling of solve method 
    
    //Removing guard cells for cg algorithim
    for(int i = 1; i <= Nx ; ++i)
    {
        for(int j = 1; j <= Ny ; ++j)
        {
            vsolve[IDX(i-1,j-1)] = v[IDX_p(i,j)];
        }
    }

    for(int i = 1; i <= Nx ; ++i)
    {
        for(int j = 1; j <= Ny ; ++j)
        {
            ssolve[IDX(i-1,j-1)] = s[IDX_p(i,j)];
        }
    }
    //CG Algorithm
    cg->Solve(vsolve, ssolve,comm, left, right, up, down, rank);

    // Re-adding padding back into loops
    for(int i = 1; i <= Nx ; ++i)
    {
        for(int j = 1 ; j <= Ny ; ++j)
        {
            v[IDX_p(i,j)] = vsolve[IDX(i-1,j-1)];
        }
    }

    for(int i = 1; i <= Nx ; ++i)
    {
        for(int j = 1 ; j <= Ny ; ++j)
        {
            s[IDX_p(i,j)] = ssolve[IDX(i-1,j-1)];
        }
    }

    //Sendrecv for streamfunction
    for(int j = 1; j <= Ny; ++j)
    {
        in_grd_x[j-1] = s[IDX_p(1,j)];
        out_grd_x[j-1] = s[IDX_p(Nx+1,j)];
    }
    MPI_Sendrecv(in_grd_x,Ny,MPI_DOUBLE,left,0,out_grd_x,Ny,MPI_DOUBLE,right,0,comm,MPI_STATUS_IGNORE);
    for(int j = 1; j <= Ny ; ++j)
    {
        s[IDX_p(Nx+1,j)] = out_grd_x[j-1];
    }

    //Right guard cell to left
    for(int j = 1; j <= Ny;++j)
    {
        in_grd_x[j-1] = s[IDX_p(Nx,j)];
        out_grd_x[j-1] = s[IDX_p(0,j)];
    }
    MPI_Sendrecv(in_grd_x,Ny,MPI_DOUBLE,right,0,out_grd_x,Ny,MPI_DOUBLE,left,0,comm,MPI_STATUS_IGNORE);

    // Y direction
    for(int i = 1; i <= Nx ; ++i)
    {
        in_grd_y[i-1] = s[IDX_p(i,1)];
        out_grd_y[i-1] = s[IDX_p(i,Ny+1)];
    }
    MPI_Sendrecv(in_grd_y,Nx,MPI_DOUBLE,down,0,out_grd_y,Nx,MPI_DOUBLE,up,0,comm,MPI_STATUS_IGNORE);
    for(int i = 1; i <= Nx;++i)
    {
        s[IDX_p(i,Ny+1)] = out_grd_y[i-1];
    }
    for(int i = 1; i <=Nx ; ++i)
    {
        in_grd_y[i-1] = s[IDX_p(i,Ny)];
        out_grd_y[i-1] = s[IDX_p(i,0)];
    }
    MPI_Sendrecv(in_grd_y,Nx,MPI_DOUBLE,up,0,out_grd_y,Nx,MPI_DOUBLE,down,0,comm,MPI_STATUS_IGNORE);
    for(int i = 1; i<= Nx; ++i)
    {
        s[IDX_p(i,0)] = out_grd_y[i-1];
    }


}
