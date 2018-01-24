from __future__ import absolute_import
# Visualization of particles with gravity
# Source: http://enja.org/2010/08/27/adventures-in-opencl-part-2-particles-with-opengl/

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
    from OpenGL.GL import * # OpenGL - GPU rendering interface
    from OpenGL.GLU import * # OpenGL tools (mipmaps, NURBS, perspective projection, shapes)
    from OpenGL.GLUT import * # OpenGL tool to make a visualization window
    from OpenGL.arrays import vbo 
    
    import argparse

except ImportError as e:
    print e
    raise ImportError

mf = cl.mem_flags
np.set_printoptions(threshold=np.nan)

# OpenGL window dimensions and perspective
width = 1000
height = 800
zoom = 40.
#zoom = 60.

# Species dictionary
# Species name, mass, massMeV, masseV, masskg, chargeC
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


nx, ny, nz = 100, 100, 100
num_points = nx*ny*nz


# Can request species at command line
p = argparse.ArgumentParser(description="Script to Run an Particle Propagator on the GPU")
p.add_argument("-s", "--species", dest="species", default="proton", required=False, help="Species type")
p.add_argument("-e", "--Emin", dest="Emin", default=1e7, type=float, required=False, help="Minimum Energy (eV)")
p.add_argument("-E", "--Emax", dest="Emax", default=1e14, type=float, required=False, help="Maximum Energy (eV)")
p.add_argument("-n", "--num_particles", dest="num_particles", default=1000, type=int, required=False, help="Number of Particles to Simulate")
args = p.parse_args()

# Get AERIE particle object
partProps = particleDict[args.species.lower()]

# Number of test particles
num_particles = args.num_particles
#num_particles = num_points#args.num_particles

# Particle Properties
Emin = args.Emin 
Emax = args.Emax
emax = np.log10(Emax)
erange = np.log10(Emax)-np.log10(Emin)

pmass = partProps[0]
massMeV = partProps[1]
masseV = partProps[2]
masskg = partProps[3]
chargeC = partProps[4]

print "\n\n"
print "Particle Properties: species\tmass [MeV]\tmass [eV]\tmass [kg]\tcharge [C]"
print "\t\t     %s\t%.02e\t%.02e\t%.02e\t%.02e"%(args.species, massMeV, masseV, masskg, chargeC)
print "\n\n"

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

# Mouse functionality
mouse_down = False
mouse_old = {'x': 0., 'y': 0.}
rotate = {'x': -55., 'y': 0., 'z': 45.}
translate = {'x': 0., 'y': 0., 'z': 0.}
initial_translate = {'x': 0., 'y': 0., 'z': -outer_radius}


# Cos ( Zenith )
thetaMin = 0 * np.pi/180
cosThetaMax = np.cos(thetaMin)
thetaMax = 180 * np.pi/180
#thetaMax = 30 * np.pi/180
#thetaMax = 90 * np.pi/180
cosThetaMin = np.cos(thetaMax)



def glut_window():
    global initRun
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(width, height)
    glutInitWindowPosition(0, 0)
    window = glutCreateWindow("Particle Simulation")

    #glClearColor(0.0,0.0,0.0,0.0) # Black Background
    glClearColor(1.0,1.0,1.0,0.0) # White Background
    glutDisplayFunc(on_display)  # Called by GLUT every frame
    glutKeyboardFunc(on_key)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutTimerFunc(10, on_timer, 10)  # Call draw every 30 ms

    return(window)

def initial_buffers(num_particles):
    
    # Load heliosphere table
    fdata = gzip.open('../LZ019500_sorted_uniq.dat.gz','rb')
    #nx, ny, nz = 10, 10, 10
    #nx, ny, nz = 312, 288, 288
    np_datax = np.zeros((num_points), dtype=np.float32)
    np_datay = np.zeros((num_points), dtype=np.float32)
    np_dataz = np.zeros((num_points), dtype=np.float32)


    for line in fdata:
        #  x   y   z   bx   by   bz
        row = line.strip().split()
        
        if (len(row)==3):
            nxx, nyy, nzz = int(row[0]), int(row[1]), int(row[2])
        else:
    
            ix, iy, iz = int(row[0]), int(row[1]), int(row[2])
            bx, by, bz = float(row[3]), float(row[4]), float(row[5])
  
            #if (ix < 10 and iy < 10 and iz < 10):
            if (ix < nx and iy < ny and iz < nz):
                idx = ix + nx*(iy + ny*iz)
                #idx = ix + ny*(iy + nz*iz)
                # Units are uG so * 10^-6 to G then * 10^-4 to T
                np_datax[idx] = bx*10**-4#10
                np_datay[idx] = by*10**-4#10
                np_dataz[idx] = bz*10**-4#10
                #np_datax[ix,iy,iz] = bx*10**-4#10
                #np_datay[ix,iy,iz] = by*10**-4#10
                #np_dataz[ix,iy,iz] = bz*10**-4#10
            if (ix > nx and iy > ny and iz > nz):
                break


    # Initialize particle properties
    np_life = np.ndarray((num_particles,1), dtype=np.bool)
    np_position = np.ndarray((num_particles, 4), dtype=np.float32)
    np_color = np.ndarray((num_particles, 4), dtype=np.float32)
    np_velocity = np.ndarray((num_particles, 4), dtype=np.float32)
    np_zmel = np.ndarray((num_particles, 4), dtype=np.float32)

    ## Test values
    Energy_array = 10**np.random.uniform(np.log10(Emin),np.log10(Emax),num_particles)
    #Energy_array = np.logspace(np.log10(Emin),np.log10(Emax),num_particles)
    Gamma_array = Energy_array/masseV+1.
    np_zmel[:,0] = -chargeC
    np_zmel[:,1] = masskg
    np_zmel[:,2] = Energy_array
    np_zmel[:,3] = Gamma_array

    np_life[:] = True

    # Start particles just beyond Sun radius (1.1)
    np_position[:,0:3] = 1.1,1.1,1.1
    npr = np.sqrt(3*1.1**2)
    np_position[:,3] = npr

    for i in range(num_particles):
        
        # Generate random local spherical vector
        rZen = np.arccos(np.random.uniform(cosThetaMin,cosThetaMax,1))[0]
        rAzi = 2*np.pi*np.random.random()
        np_velocity[i,0] = np.sin(rZen)*np.cos(rAzi)
        np_velocity[i,1] = np.sin(rZen)*np.sin(rAzi)
        np_velocity[i,2] = np.cos(rZen)

    # Arrays for OpenGL bindings
    gl_position = vbo.VBO(data=np_position, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    gl_position.bind()
    gl_color = vbo.VBO(data=np_color, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    gl_color.bind()

    return (np_datax, np_datay, np_dataz, np_life, np_position, np_velocity, np_zmel, gl_position, gl_color)

def on_timer(t):
    glutTimerFunc(t, on_timer, t)
    glutPostRedisplay()

def on_key(*args):
    global time_step
    if args[0] == ' ' or args[0] == 'p':
        time_step = time_pause_var-time_step
    if args[0] == '\033' or args[0] == 'q':
        sys.exit()

def mouse(button, state, x, z):
    global action
    if (button==GLUT_LEFT_BUTTON):
        action = "ROTATE"
    elif (button==GLUT_RIGHT_BUTTON):
        action = "ZOOM"
    elif (button==GLUT_MIDDLE_BUTTON):
        action = "TRANS"

    mouse_old['x'] = x
    mouse_old['z'] = z

def on_mouse_rotate(x, z):
    rotate['x'] += (z - mouse_old['z']) * .1#2
    rotate['z'] += (x - mouse_old['x']) * .1#2

def on_mouse_trans(x, z):
    translate['x'] += x - mouse_old['x']
    translate['z'] += z - mouse_old['z']

def on_mouse_zoom(x, z):
    global zoom
    zoom -= z - mouse_old['z']
    if (zoom > 150.):
        zoom = 150.
    elif zoom < 1:
        zoom = 1.1

def motion(x, z):
    if action=="ROTATE":
        on_mouse_rotate(x, z)
    elif action=="ZOOM":
        on_mouse_zoom(x, z)
    elif action=="TRANS":
        on_mouse_trans(x, z)
    else:
        print("Unknown action\n")
    mouse_old['x'] = x
    mouse_old['z'] = z
    glutPostRedisplay()


def axis(length):
    """ Draws an axis (basicly a line with a cone on top) """
    glPushMatrix()
    glBegin(GL_LINES)
    glVertex3d(0,0,0)
    glVertex3d(0,0,length)
    glEnd()
    glTranslated(0,0,length)
    glutWireCone(0.04,0.2, 12, 9)
    glPopMatrix()

def Haxis(length):
    """ Draws an axis (basicly a line with a cone on top) """
    glPushMatrix()
    glBegin(GL_LINES)
    glVertex3d(0,0,1.)
    glVertex3d(0,0,length)
    glEnd()
    glTranslated(0,0,length)
    glutWireCone(0.04,0.2, 12, 9)
    glPopMatrix()
    
def threeAxis(length):
    """ Draws an X, Y and Z-axis """ 
    glPushMatrix()
    # Z-axis
    glColor3f(1.0,0.0,0.0)
    axis(length)
    # X-axis
    glRotated(90,0,1.0,0)
    glColor3f(0.0,1.0,0.0)
    axis(length)
    # Y-axis
    glRotated(-90,1.0,0,0)
    glColor3f(0.0,0.0,1.0)
    axis(length)

    ## HAWC Normal
    #glRotated(187,1.0,0,0)
    #glRotated(-19,0,1.0,0)
    #glColor3f(0.0,0.0,0.0)
    ##glColor3f(1.0,1.0,1.0)
    #Haxis(2*length)

    glPopMatrix()

def on_display():
    """Render the particles"""        
    # Update or particle positions by calling the OpenCL kernel
    cl.enqueue_acquire_gl_objects(queue, [cl_gl_position, cl_gl_color])

    kernelargs = (cl_datax, cl_datay, cl_dataz, cl_life, cl_gl_position, cl_velocity, cl_zmel, cl_gl_color, cl_start_position, cl_start_velocity, np.float32(emax), np.float32(erange), np.float32(time_step))

    gsize = num_particles #if num_points < num_particles else num_points

    #tab_size = 312 * 288 * 288 * 6
    program.particle_fountain(queue, (gsize,), None, *(kernelargs))
    cl.enqueue_release_gl_objects(queue, [cl_gl_position, cl_gl_color])
    queue.finish()
    glFlush()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(zoom, width / float(height), .1, 10000.)

    # Handle mouse transformations
    glTranslatef(initial_translate['x'], initial_translate['y'], initial_translate['z'])
    glRotatef(rotate['x'], 1, 0, 0)
    glRotatef(rotate['z'], 0, 0, 1)
    glTranslatef(translate['x'], translate['y'], translate['z'])
    
    # Render the particles
    glEnable(GL_POINT_SMOOTH)
    glPointSize(2)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Set up the VBOs
    gl_color.bind()
    glColorPointer(4, GL_FLOAT, 0, gl_color)
    gl_position.bind()
    glVertexPointer(4, GL_FLOAT, 0, gl_position)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)

    # Draw the VBOs
    glDrawArrays(GL_POINTS, 0, num_particles)

    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)

    glDisable(GL_BLEND)
    
    # Draw Axes
    threeAxis(1.5)
    #glClear(GL_COLOR_BUFFER_BIT)
    
    # Draw Transparent Sun
    glEnable(GL_BLEND)
    #glBlendFunc (GL_SRC_ALPHA, GL_ONE) 
    #glBlendFunc(GL_ONE, GL_ONE_MINUS_DST_ALPHA)
    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_DST_ALPHA)
    glColor4f(0.0,0.0,1.0, 0.15)
    glutSolidSphere(1.0,32,32)
    glDisable(GL_BLEND)

    #glPopMatrix()
   
    glutSwapBuffers()


def printHelp():
    print """\n\n
          ------------------------------------------------------------------------------\n
          Left Mouse Button:        - rotate viewing position\n
          Middle Mouse Button:      - translate the scene\n
          Right Mouse Button:       - zoom in and out of scene\n
          
          Keys
            p:                      - start or pause the program\n
            q,Esc:                  - exit the program\n
          ------------------------------------------------------------------------------\n
          \n"""

#-----
# MAIN
#-----
if __name__=="__main__":
    printHelp()
    window = glut_window()
    
    (np_datax, np_datay, np_dataz, np_life, np_position, np_velocity, np_zmel, gl_position, gl_color) = initial_buffers(num_particles)
    
    platform = cl.get_platforms()[0]
    context = cl.Context(properties=[(cl.context_properties.PLATFORM, platform)] + get_gl_sharing_context_properties())  
    queue = cl.CommandQueue(context)
    
    cl_life = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=np_life)
    cl_velocity = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=np_velocity)
    cl_zmel = cl.Buffer(context, mf.COPY_HOST_PTR, hostbuf=np_zmel)
    
    cl_datax = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_datax)
    cl_datay = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_datay)
    cl_dataz = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_dataz)

    cl_start_position = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_position)
    cl_start_velocity = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_velocity)
    
    if hasattr(gl_position,'buffers'):
        cl_gl_position = cl.GLBuffer(context, mf.READ_WRITE, int(gl_position.buffers[0]))
        cl_gl_color = cl.GLBuffer(context, mf.READ_WRITE, int(gl_color.buffers[0]))
    elif hasattr(gl_position,'buffer'):
        cl_gl_position = cl.GLBuffer(context, mf.READ_WRITE, int(gl_position.buffer))
        cl_gl_color = cl.GLBuffer(context, mf.READ_WRITE, int(gl_color.buffer))
    else:
        print "Can not find a proper buffer object in pyopencl install. Exiting..."
        sys.exit()
    
    f = open("cl_funcs_heliosphere.cl",'r')
    fstr = "".join(f.readlines())
    program = cl.Program(context, fstr).build()
    
    glutMainLoop()
