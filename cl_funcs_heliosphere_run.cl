static int my_isnan(float d)
{
  return (d != d);              /* IEEE: only NaN is not equal to itself */
}

//#define NaN log(-1.0)
//#define FT2KM (1.0/0.0003048)
#define PI 3.141592653589793238462
#define RAD2DEG (180.0/PI)
#define DEG2RAD (PI/180.0)
#define THETA_MIN 0.01
//#define THETA_MIN 0.001


#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)>(b))?(b):(a))

// Constants and useful values / vectors
__constant float speed_of_light = 299792458.; //m/s
__constant float c2 = 299792458.*299792458.;
__constant float4 zero_vec = {0.0f,0.0f,0.0f,0.0f};
__constant float S_r = 695700; // Solar Radius (km)
__constant float S_r_m = 695700000; // Solar Radius (m)
__constant float BMAG = 0.000033; //Tesla
__constant float au_to_m = 149600000000; // A.U. to m
__constant float m_to_au = 0.00000000000668459; // m to A.U.

// Particle struct
struct particle_struct
{
    float4 pos;  // position vector
    float4 vel;  // velocity vector
    float4 ZMEL; // Charge, mass, energy, lambda (C,kg,eV,[])
    bool alive;   // particle life status
    float time;
};


static float4 vec_sum(const float4 a,
                   const float4 b)
{ 
    return a+b;
}

static float vec_three_dot(const float4 a,
                   const float4 b)
{ 
    float dot_prod = 0.;
    dot_prod += a.x*b.x;
    dot_prod += a.y*b.y;
    dot_prod += a.z*b.z;
    return dot_prod;
}

static float vec_three_Mag(const float4 a)
{ 
    float norm = vec_three_dot(a,a);
    return sqrt(norm);
}

static float4 vec_scale(const float scale,
                   const float4 vec)
{
    float4 svec;
    svec.x = vec.x*scale;
    svec.y = vec.y*scale;
    svec.z = vec.z*scale;
    return svec;
}

static float4 vec_normalize(const float4 a)
{
    float4 nvec;
    float norm = vec_three_Mag(a);
    if (norm > 0.)
    {
        nvec = vec_scale(1./norm,a);
    }

    return nvec;
}

static float4 vec_three_cross(const float4 a, const float4 b)
{
    float4 cross;
    cross.w = 0.0f;
    cross.x = a.y * b.z - b.y * a.z;
    cross.y = b.x * a.z - a.x * b.z;
    cross.z = a.x * b.y - b.x * a.y;
    return cross;
}

static void swap(float *a, float *b)
{
    float temp = *a;
    *a = *b;
    *b = temp;
}



static float4 Colors(float lambda)
{
    float R, G, B, SSS;

    //nm to RGB
    if (lambda >= 380 && lambda < 440)
    {
        R = -(lambda - 440.) / (440. - 350.);
        G = 0.0;
        B = 1.0;
    }
    else if (lambda >= 440 && lambda < 490)
    {
        R = 0.0;
        G = (lambda - 440.) / (490. - 440.);
        B = 1.0;
    }
    else if (lambda >= 490 && lambda < 510)
    {
        R = 0.0;
        G = 1.0;
        B = -(lambda - 510.) / (510. - 490.);
    }
    else if (lambda >= 510 && lambda < 580)
    {
        R = (lambda - 510.) / (580. - 510.);
        G = 1.0;
        B = 0.0;
    }
    else if (lambda >= 580 && lambda < 645)
    {
        R = 1.0;
        G = -(lambda - 645.) / (645. - 580.);
        B = 0.0;
    }
    else if (lambda >= 645 && lambda <= 780)
    {
        R = 1.0;
        G = 0.0;
        B = 0.0;
    }
    else
    {
        R = 0.0;
        G = 0.0;
        B = 0.0;
    }
    //Intensity Correction
    if (lambda >= 380 && lambda < 420)
        SSS = 0.3 + 0.7*(lambda - 350) / (420 - 350);
    else if (lambda >= 420 && lambda <= 700)
        SSS = 1.0;
    else if (lambda > 700 && lambda <= 780)
        SSS = 0.3 + 0.7*(780 - lambda) / (780 - 700);
    else
        SSS = 0.0;
    SSS *= 255;
    R *= SSS;
    G *= SSS;
    B *= SSS;
    float4 color = {R,G,B,0.0};
    return color;
}


// Uniform test field
static float4 GetUniformField(void)
{
    // Similar strength to Earth's field
    float4 B;
    B.x = 0.;
    B.y = 0.;
    B.z = BMAG;
    B.w = BMAG;
    return B;
}


//// Dipole Approx from AERIE Code:
//// aerie/trunk/src/astro-service/src/GeoDipoleService.cc
//static float4 GetField(float4 pos)
//{
//    float r = vec_three_Mag(pos)/S_r_m; //EquatorialRadius (meters)
//    float B0 = 31.2 * 1e-6; // micro Tesla
//    float tilt = 0.;//11.5*PI/180;//DEG2RAD; // deg to rad
//
//    float xp = pos.x;
//    float yp = pos.y*cos(tilt) - pos.z*sin(tilt);
//    float zp = pos.y*sin(tilt) + pos.z*cos(tilt);
//    float rp = sqrt(xp*xp + yp*yp + zp*zp);
//
//    float theta = acos(zp/rp);
//    float phi = atan2(yp,xp);
//
//    float r3 = r*r*r;
//    float Br = -2*B0*(1./r3)*cos(theta);
//    float Btheta = -B0*(1./r3)*sin(theta);
//    
//    float4 B;
//    B.x = sin(theta)*cos(phi)*Br + cos(theta)*cos(phi)*Btheta;
//    B.y = sin(theta)*sin(phi)*Br + cos(theta)*sin(phi)*Btheta;
//    B.z = cos(theta)*Br - sin(theta)*Btheta;
//
//    B.w = vec_three_Mag(B);
//    return B;
//}

static float4 GetField(float4 pos, __global float* datax, __global float* datay, __global float* dataz)
{

    float4 B;

    // Get bin values (pos is sun-centered)
    int xbin = (int)((pos.x*m_to_au-5200)/20.);
    int ybin = (int)((pos.y*m_to_au-2880)/20.);
    int zbin = (int)((pos.z*m_to_au-2880)/20.);

    // Get magnetic field components
    B.x = datax[xbin,ybin,zbin];
    B.y = datay[xbin,ybin,zbin];
    B.z = dataz[xbin,ybin,zbin];

    return B;
}



static void PropStepBoris(float dt, struct particle_struct *particle, __global float* datax, __global float* datay, __global float* dataz)
{
// https://en.wikipedia.org/wiki/Particle-in-cell
// http://e-collection.library.ethz.ch/eserv/eth:5175/eth-5175-01.pdf
// http://www.osti.gov/scitech/servlets/purl/1090047/

    float4 p = particle->pos;
    float4 v = particle->vel;

    float Z = particle->ZMEL.x;
    float mass = particle->ZMEL.y;
    float inv_gamman = 1./particle->ZMEL.w;

    float4 B = GetField(p,datax,datay,dataz);
    //float4 B = GetUniformField();

    float q = 0.5*Z*dt*inv_gamman/mass;
    //float q = 0.5*Z*dt/mass;
    float4 h = vec_scale(q,B);
    float hMag2 = vec_three_dot(h,h);

    float4 s = vec_scale(2./(1.+hMag2),h);

    float4 u = v; // v+Q*E where E is electric field

    float4 u_prime = vec_three_cross(u,h);
    u_prime = vec_sum(u,u_prime);
    u_prime = vec_three_cross(u_prime,s);
    u_prime = vec_sum(u,u_prime);


    // Update velocity
    v = u_prime; // u_prime + Q*E

    // Update position
    float4 dx = vec_scale(dt,v);
    particle->vel = v;
    particle->pos = vec_sum(p,dx);
//    particle->ZMEL.w = 1./inv_gamman;
    
    // Adding time step
    particle->time += dt;
}

static void PropStepAdaptBoris(float dt, struct particle_struct *particle, __global float* datax, __global float* datay, __global float* dataz)
{
// https://en.wikipedia.org/wiki/Particle-in-cell
// http://e-collection.library.ethz.ch/eserv/eth:5175/eth-5175-01.pdf
// http://www.osti.gov/scitech/servlets/purl/1090047/

    float4 p = particle->pos;
    float4 v = particle->vel;

    float Z = particle->ZMEL.x;
    float Z_mag = fabs(Z);
    float mass = particle->ZMEL.y;
    float inv_gamman = 1./particle->ZMEL.w;

    float4 B = GetField(p,datax,datay,dataz);
    //float4 B = GetUniformField();
    float Bmag = B.w;

    float dt_theta = 2.*mass*tan(0.5*THETA_MIN)/(Z_mag*Bmag*inv_gamman);
    //float dt_theta = 2.*mass*tan(0.5*THETA_MIN)/(Z*Bmag*inv_gamman);
    //float dt_theta = 2.*mass*tan(0.5*THETA_MIN)/(Z*Bmag);
    dt = MIN(dt_theta,dt); 

    float q = 0.5*Z*dt*inv_gamman/mass;
    //float q = 0.5*Z*dt/mass;
    float4 h = vec_scale(q,B);
    float hMag2 = vec_three_dot(h,h);

    float4 s = vec_scale(2./(1.+hMag2),h);

    float4 u = v; // v+Q*E where E is electric field

    float4 u_prime = vec_three_cross(u,h);
    u_prime = vec_sum(u,u_prime);
    u_prime = vec_three_cross(u_prime,s);
    u_prime = vec_sum(u,u_prime);

    // Update velocity
    v = u_prime; // u_prime + Q*E

    // Update position
    float4 dx = vec_scale(dt,v);
    particle->vel = v;
    particle->pos = vec_sum(p,dx);
    particle->ZMEL.w = 1./inv_gamman;
    
    // Adding time step
    particle->time += dt;
}

// Propagation Step function
static void Propagate(float time_step, __global float*datax, __global float* datay, __global float* dataz, struct particle_struct *particle, float4 startp, float4 startv)
{

    //// If particle hits Sun or get too far away, respawn
    //float speed = vec_three_Mag(particle->vel);
    //float posr = vec_three_Mag(particle->pos);
    
    //PropStepRK4(time_step,particle);
    //PropStepLin(time_step,particle);
    PropStepBoris(time_step,particle,datax,datay,dataz);
    //PropStepAdaptBoris(time_step,particle);

    //bool palive = (particle->alive);
    //if (palive)
    //{
    //    if (posr<=S_r_m)
    //    {
    //        particle->pos = vec_scale(S_r_m,startp);
    //        particle->vel = vec_scale(speed,startv);
    //    }
    //    else
    //    {
    //        //PropStepRK4(time_step,particle);
    //        //PropStepLin(time_step,particle);
    //        PropStepBoris(time_step,particle);
    //        //PropStepAdaptBoris(time_step,particle);
    //    }
    //}
}


// Main kernel function
__kernel void particle_fountain(
                                __global float* datax,
                                __global float* datay,
                                __global float* dataz,
                                int num_points,
                                int num_particles
                                )
//                                __global bool* life,
//                                __global float4* position, 
//                                __global float4* velocity,
//                                __global float4* zmel,
//                                __global float4* color,
//                                __global float4* start_position,
//                                __global float4* start_velocity,
//                                float maxE,
//                                float range,
//                                float time_step)
{
    // Get this particles address on GPU
    unsigned int gid = get_global_id(0);
    float bvalx = datax[gid];
    float bvaly = datay[gid];
    float bvalz = dataz[gid];

    printf("%i %i %i %.05f %.05f %.05f\n",gid,num_points,num_particles,bvalx,bvaly,bvalz);
    

    /*
    // Grab position and direction vectors
    float4 p = vec_scale(S_r_m,position[gid]);
    float4 v = velocity[gid];

    // Scale dir vector to speed
    float gamma = zmel[gid].w;
    float speed = sqrt(1.-1./(gamma*gamma))*speed_of_light;
    v = vec_normalize(v);
    v = vec_scale(speed,v);

    // Put gid particle's properties in struct
    struct particle_struct particle;
    particle.pos = p;
    particle.vel = v;
    particle.ZMEL = zmel[gid];
    particle.alive = life[gid];
    particle.time = velocity[gid].w;

    // Propagate in time via stepper function of choice
    Propagate(time_step, datax, datay, dataz, &particle, start_position[gid], start_velocity[gid]);

    // Grab position and velocity, ensuring to scale properly for viewing
    p = particle.pos;
    p = vec_scale(1./S_r_m,p);
    p.w = 1.;
    position[gid] = p;

    v = particle.vel;
    v.w = particle.time;
    velocity[gid] = v;
    
    zmel[gid].w = particle.ZMEL.w;

    life[gid] = particle.alive;
    
    float energy = log10(particle.ZMEL.z);
    float lambda = (780.-380.)*(maxE-energy)/range+380.;
    */

    //float range = 100.;
    //float lambda = (780.-380.)*(maxB - bval)/range+ 380.;

    //if (range == 0.)
    //    lambda = 580.;
    //color[gid] = Colors(lambda);
    //color[gid].w = 1.0f; /* Fade points as life decreases */
}
