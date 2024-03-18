#include <iostream>
#include "mpi.h"
#include <cmath>
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
    //int loc_nx = Nx/dim_prcs[0];
    //int loc_ny = Ny/dim_prcs[1];
    MPI_Comm cartesian_comm;
    int periodic[2] = {0,0}; // Non-periodic
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_prcs, periodic, 0, &cartesian_comm);
   
    int coords[2];
    //Coordinate implementation
    //MPI_Comm_rank(cartesian_comm,&rank);
    int left;
    int right;
    int down;
    int up;
    MPI_Cart_coords(cartesian_comm,rank,2,coords);
    MPI_Cart_shift(cartesian_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cartesian_comm, 1, 1, &down, &up);
    MPI_Cart_rank(cartesian_comm, coords, &rank);

    //Load Balancing
    
    int rem_x     = Nx % size;            // remainder
    int min_loc_nx    = (Nx - rem_x) / size;      // minimum size of chunk
    int    start_x = 0;                   // start index of chunk
    int    end_x  = 0;                   // end index of chunk
    if (rank < (Nx % size)) {            // for ranks < r, chunk is size k + 1
        min_loc_nx++;
        start_x = min_loc_nx * rank;
        end_x   = min_loc_nx* (rank + 1);
    }
    else {                              // for ranks > r, chunk size is k
        start_x = (min_loc_nx+1) * rem_x + min_loc_nx * (rank - rem_x);
        end_x  = (min_loc_nx+1) * rem_x + min_loc_nx * (rank - rem_x+ 1);
    }
    int loc_nx = end_x - start_x;

    int rem_y     = Ny % size;            // remainder
    int min_loc_ny    = (Ny - rem_y) / size;      // minimum size of chunk
    int    start_y = 0;                   // start index of chunk
    int    end_y  = 0;                   // end index of chunk
    if (rank < (Ny % size)) {            // for ranks < r, chunk is size k + 1
        min_loc_ny++;
        start_y = min_loc_ny * rank;
        end_y   = min_loc_ny* (rank + 1);
    }
    else {                              // for ranks > r, chunk size is k
        start_y = (min_loc_ny+1) * rem_y + min_loc_ny * (rank - rem_y);
        end_y  = (min_loc_nx+1) * rem_y + min_loc_ny * (rank - rem_y+ 1);
    }
    int loc_ny = end_y - start_y;


    

    LidDrivenCavity* solver = new LidDrivenCavity();
    solver->SetGridSize(loc_nx,loc_ny);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);
    
    /*
    if(rank == 0)
    {
        solver->PrintConfiguration();
    }
    */
    solver->Initialise();

    //solver->WriteSolution("ic.txt");
    solver->Integrate(cartesian_comm, left, right, up, down, rank);

    //solver->WriteSolution("final.txt");
    MPI_Comm_free(&cartesian_comm);    
    MPI_Finalize();
    delete[] param_double;
    delete[] param_int;
    delete[] dim_prcs;
    return 0;
}
