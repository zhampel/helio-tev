from __future__ import absolute_import

try:
    import sys # System tools (path, modules, maxint)
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib.ticker import ScalarFormatter
    import numpy as np 
    from matplotlib.backends.backend_pdf import PdfPages
    import gzip

    import pyopencl as cl # OpenCL - GPU computing interface
    from pyopencl.tools import get_gl_sharing_context_properties

    import time

    import os

    import argparse

except ImportError as e:
    print e
    raise ImportError

mf = cl.mem_flags
np.set_printoptions(threshold=np.nan)

# Number of test particles
num_particles = 1000

# Species dictionary
particleDict = {
                "proton": [ 10439684078.6 , 938.272013 , 938272013.0 , 1.67262161014e-27 , 1.602176462e-19 ],
                "helium": [ 41472686429.3 , 3727.37917045 , 3727379170.45 , 6.64465620129e-27 , 3.204352924e-19 ],
                "carbon": [ 1.24371227955e+11 , 11177.928521 , 11177928521.0 , 1.9926465398e-26 , 9.613058772e-19 ],
                "oxygen": [ 1.65775597689e+11 , 14899.1676931 , 14899167693.1 , 2.65601760592e-26 , 1.2817411696e-18 ],
                "neon": [ 2.07207027869e+11 , 18622.8389368 , 18622838936.8 , 3.31982222813e-26 , 1.602176462e-18 ],
                "magnesium": [ 2.48587424065e+11 , 22341.9234747 , 22341923474.7 , 3.98280919586e-26 , 1.9226117544e-18 ],
                "silicon": [ 2.89960392271e+11 , 26060.3404182 , 26060340418.2 , 4.64567715409e-26 , 2.2430470468e-18 ],
                "iron": [ 5.79724738538e+11 , 52103.0611003 , 52103061100.3 , 9.28821330525e-26 , 4.1656588012e-18 ],
                "electron": [ 5685629.65855 , 0.51099891 , 510998.91 , 9.10938201058e-31 , -1.602176462e-19 ],
                "positron": [ 5685629.65855 , 0.51099891 , 510998.91 , 9.10938201058e-31 , 1.602176462e-19 ]
                }

# Particle Properties
Emin = 1e11# eV 
Emax = 1e14# eV 
emax = np.log10(Emax)
erange = np.log10(Emax)-np.log10(Emin)

# Set scale of positions and velocities
outer_radius = 2.501
inner_radius = 2.5
norm_vel = 1

# Cos ( Lowest Zenith )
cosThetaMin = 0.707106781186548 # 45 deg

# Set the time step and pause functionality
time_step = 0.0005

# Solar's radius in m
Sun_radius = 6.957*10**8

# Set scale of positions and velocities
outer_radius = 20.501
inner_radius = 20.5
norm_vel = 1

# Set the time step and pause functionality
time_step = 0.05
#time_step = 0.0005
time_pause_var = time_step
# Set initial step to 0 to start paused
time_step = 0

platform = cl.get_platforms()[0]
context = cl.create_some_context()
queue = cl.CommandQueue(context)

f = open("cl_funcs_heliosphere_run.cl",'r')
fstr = "".join(f.readlines())
program = cl.Program(context, fstr).build()


def initial_buffers(species,num_particles):

    # Get AERIE particle object
    partProps = particleDict[species.lower()]

    pmass = partProps[0]
    massMeV = partProps[1]
    masseV = partProps[2]
    masskg = partProps[3]
    chargeC = partProps[4]

    print "\n\n"
    print "Particle Properties: species\tmass [MeV]\tmass [eV]\tmass [kg]\tcharge [C]"
    print "\t\t     %s\t%.02e\t%.02e\t%.02e\t%.02e"%(args.species, massMeV, masseV, masskg, chargeC)
    print "\n\n"

    
    # Load heliosphere table
    fdata = gzip.open('../LZ019500_sorted_uniq.dat.gz','rb')
    nx, ny, nz = 10, 10, 10
    #nx, ny, nz = 312, 288, 288
    num_points = nx*ny*nz
    np_datax = np.zeros((num_points), dtype=np.float32)
    np_datay = np.zeros((num_points), dtype=np.float32)
    np_dataz = np.zeros((num_points), dtype=np.float32)
    #np_datax = np.zeros((nx,ny,nz), dtype=np.float32)
    #np_datay = np.zeros((nx,ny,nz), dtype=np.float32)
    #np_dataz = np.zeros((nx,ny,nz), dtype=np.float32)



    for line in fdata:
        #  x   y   z   bx   by   bz
        row = line.strip().split()
        
        if (len(row)==3):
            nx, ny, nz = int(row[0]), int(row[1]), int(row[2])
        else:
    
            ix, iy, iz = int(row[0]), int(row[1]), int(row[2])
            bx, by, bz = float(row[3]), float(row[4]), float(row[5])
  
            if (ix < 10 and iy < 10 and iz < 10):
                # Units are uG so * 10^-6 to G then * 10^-4 to T
                idx = ix + ny*(iy + nz*iz)
                np_datax[idx] = bx#*10**-4#10
                np_datay[idx] = by#*10**-4#10
                np_dataz[idx] = bz#*10**-4#10
                print ix, iy, iz, idx, np_datax[idx], np_datay[idx], np_dataz[idx]
            if (ix > 10 and iy > 10 and iz > 10):
                break

    ## Initialize particle properties
    #np_life = np.ndarray((num_particles,1), dtype=np.bool)
    #np_position = np.ndarray((num_particles, 4), dtype=np.float32)
    #np_color = np.ndarray((num_particles, 4), dtype=np.float32)
    #np_velocity = np.ndarray((num_particles, 4), dtype=np.float32)
    #np_zmel = np.ndarray((num_particles, 4), dtype=np.float32)

    ### Test values
    #Energy_array = 10**np.random.uniform(np.log10(Emin),np.log10(Emax),num_particles)
    ##Energy_array = np.logspace(np.log10(Emin),np.log10(Emax),num_particles)
    #Gamma_array = Energy_array/masseV+1.
    #np_zmel[:,0] = -chargeC
    #np_zmel[:,1] = masskg
    #np_zmel[:,2] = Energy_array
    #np_zmel[:,3] = Gamma_array

    #np_life[:] = True

    ## Start particles just beyond Sun radius (1.1)
    #np_position[:,0:3] = 1.1,1.1,1.1
    #npr = np.sqrt(3*1.1**2)
    #np_position[:,3] = npr

    #for i in range(num_particles):
    #    
    #    # Generate random local spherical vector
    #    rZen = np.arccos(np.random.uniform(cosThetaMin,cosThetaMax,1))[0]
    #    rAzi = 2*np.pi*np.random.random()
    #    np_velocity[i,0] = np.sin(rZen)*np.cos(rAzi)
    #    np_velocity[i,1] = np.sin(rZen)*np.sin(rAzi)
    #    np_velocity[i,2] = np.cos(rZen)

    return (np_datax, np_datay, np_dataz, num_points)
    #return (np_datax, np_datay, np_dataz, np_life, np_position, np_velocity, np_zmel)


def run_kernel(np_datax,np_datay,np_dataz,num_points):
    
    gsize = num_particles if num_points < num_particles else num_points

    cl_datax = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_datax)
    cl_datay = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_datay)
    cl_dataz = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dataz)
    #cl_datax = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_datax)
    #cl_datay = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_datay)
    #cl_dataz = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dataz)


    global_size = (gsize,)
    #global_size = (num_particles,)
    #global_size = np_position.shape
    local_size = None

    kernelargs = (cl_datax, cl_datay, cl_dataz, np.int32(num_points), np.int32(num_particles))

    tstart = time.time()
    print("Starting kernel")
    program.particle_fountain(queue, global_size, local_size, *(kernelargs))

    queue.finish()

    print("Leaving kernel")
    tend = time.time()
    print("Execution time: %g s" %(tend-tstart))

    #dx = np.empty_like(np_datax)
    #dy = np.empty_like(np_datay)
    #dz = np.empty_like(np_dataz)

    #cl.enqueue_read_buffer(queue, life_buf, flife).wait()
    #cl.enqueue_read_buffer(queue, moon_int_buf, fmoon_int).wait()
    #cl.enqueue_read_buffer(queue, zmel_buf, fzmel).wait()
    #cl.enqueue_read_buffer(queue, pos_buf, fpos).wait()
    #cl.enqueue_read_buffer(queue, vel_buf, fvel).wait()
    #return fpos, fvel, fzmel, flife, fmoon_int
    

def main():
    global args
    p = argparse.ArgumentParser(description="Script to Run an Particle Propagator on the GPU")
    p.add_argument("-o", "--outFile", dest="outFile", default="test.root", required=False, help="Output ROOT file name")
    p.add_argument("-n", "--nbatches", dest="nbatches", default=0, type=int, required=False, help="Number of batches to loop over kernel")
    p.add_argument("-s", "--species", dest="species", default="proton", required=False, help="Species type")
    args = p.parse_args()


    np_datax, np_datay, np_dataz, num_points = initial_buffers(args.species,num_particles)
    run_kernel(np_datax, np_datay, np_dataz, num_points)

    import sys
    sys.exit()


if __name__ == "__main__":
    main()
