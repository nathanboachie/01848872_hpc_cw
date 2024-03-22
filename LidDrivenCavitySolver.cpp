#include <iostream>
#include "mpi.h"
#include <cmath>
#include <omp.h>
using namespace std;

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "LidDrivenCavity.h"

/**
 * @brief Takes options from solver executable command line, runs solver and writes solutions to text files
*/
int main(int argc, char **argv, int argc_mpi, char* argv_mpi[])
{
    double Lx, Ly, dt, T, Re;
    int Nx, Ny;
    double* param_double = new double[5];
    int* param_int = new int[2];
    //Initialising MPI, comm, rank and size 
    int periodic[2] = {0,0}; // Non-periodic
    MPI_Init(&argc_mpi,&argv_mpi);
    int rank, size ;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    //
    if(rank == 0)
    {
        
        if(ceil(static_cast<double>(sqrt(size))) != floor(static_cast<double>(sqrt(size))))
        {   
                cout << "Please use a perfect square for your number of processors" << endl;
                MPI_Finalize();
                exit(-1);
        }

        po::options_description opts(
            "Solver for the 2D lid-driven cavity incompressible flow problem");
        opts.add_options()
            ("Lx",  po::value<double>()->default_value(1.0),
                    "Length of the domain in the x-direction.")
            ("Ly",  po::value<double>()->default_value(1.0),
                    "Length of the domain in the y-direction.")
            ("Nx",  po::value<int>()->default_value(9),
                    "Number of grid points in x-direction.")
            ("Ny",  po::value<int>()->default_value(9),
                    "Number of grid points in y-direction.")
            ("dt",  po::value<double>()->default_value(0.01),
                    "Time step size.")
            ("T",   po::value<double>()->default_value(1.0),
                    "Final time.")
            ("Re",  po::value<double>()->default_value(10),
                    "Reynolds number.")
            ("verbose",    "Be more verbose.")
            ("help",       "Print help message.");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, opts), vm);
        po::notify(vm);

        if (vm.count("help")) {
            cout << opts << endl;
            return 0;
        }
        //Assigning into blocks for later
        Lx = vm["Lx"].as<double>();
        param_double[0] = Lx;
        Ly = vm["Ly"].as<double>();
        param_double[1] = Ly;
        Nx = vm["Nx"].as<int>();
        param_int[0] = Nx;
        Ny = vm["Ny"].as<int>();
        param_int[1] = Ny;
        dt = vm["dt"].as<double>();
        param_double[2] = dt;
        T = vm["T"].as<double>();
        param_double[3] = T;
        Re = vm["Re"].as<double>();
        param_double[4] = Re;
    }
    
    //Broadcasting values across all ranks;
    MPI_Bcast(param_double,5,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Bcast(param_int,2,MPI_INT,0,MPI_COMM_WORLD);
    
    //Bringing Values into scope
    Lx = param_double[0];
    Ly = param_double[1];
    Nx = param_int[0];
    Ny = param_int[1];
    dt = param_double[2];
    T = param_double[3];
    Re = param_double[4];

    
    
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

   
    //Setting variables
    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetDomainSize(loc_lx,loc_ly);
    solver->SetGridSize(loc_nx,loc_ny);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);
    
    //Print config to rank zero
    if(rank == 0)
    {
        solver->PrintConfiguration(size);
    }
    MPI_Barrier(cartesian_comm);

    //Initialise variables
    solver->Initialise();

    //Integrate and write solutions
    solver->WriteSolution("ic.txt",rank,dim_prcs,coords,cartesian_comm);
    solver->Integrate(cartesian_comm, left, right, up, down, rank);
    solver->WriteSolution("final.txt",rank,dim_prcs,coords,cartesian_comm);
    
    //Memory deallocation//
    MPI_Comm_free(&cartesian_comm);  
    MPI_Finalize();
    delete[] param_double;
    delete[] param_int;
    delete[] dim_prcs;
    return 0;
}
