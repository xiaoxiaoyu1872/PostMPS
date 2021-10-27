#ifndef DEFINE_H_
#define DEFINE_H_

#include "Helper.h"
#include "MarchingCubesHelper.h"

#define DEBUG 0

#define PI 3.14159265f
#define EPSILON_ 1.0e-7
#define SIGMA 1.56668147f //315.0f / 64.0f * 3.1415926535898f

#define SPARY 6
#define BUBBLE 20

#define SURFACE 15

#define MAXUINT 4294967295

// thrust support bool char for scan in cuda9.0 strange
// #define CUDA9 

// #define PRINT
#define NOBOUND
#define LOWRES
// #define USING_GRID
#ifdef CUDA9
typedef unsigned char uchar;

typedef uchar  NumParticleGrid;	
typedef uint   NumParticleGridScan;	

typedef bool   IsSurface;
typedef uint   IsSurfaceScan;

typedef uchar  IsValid;
typedef uint  IsValidScan;

typedef bool  IsDiffuse;
typedef uint  IsDiffuseScan;

typedef int NumDiffuseParticle;	
typedef uint   NumDiffuseParticleScan;	

typedef uchar  NumVertexCube;
typedef uint   NumVertexCubeScan;

typedef uchar  CubeFlag;

typedef uint   Index;
#else
typedef uint  NumParticleGrid;	
typedef uint   NumParticleGridScan;	

typedef uint   IsSurface;
typedef uint   IsSurfaceScan;

typedef uint  IsValid;
typedef uint  IsValidScan;

typedef uint  IsDiffuse;
typedef uint  IsDiffuseScan;

typedef int NumDiffuseParticle;	
typedef uint   NumDiffuseParticleScan;	

typedef uint  NumVertexCube;
typedef uint   NumVertexCubeScan;

typedef uint  CubeFlag;

typedef uint   Index;
#endif

typedef float  ScalarFieldGrid;

typedef float3 Vertex;
typedef float3 Normal;
typedef uint3  VertexIndex;

typedef float  ColorField;

typedef float3 ThinFeature;

struct SimParams
{
    float particleSpacing;    
    float mass;

};

struct SurfaceParams
{
    int  smoothingRadiusRatio;
    float smoothingRadius;
    float smoothingRadiusInv;
    float smoothingRadiusSq;
    float anisotropicRadius;

    int minNumNeighbors;
    int isolateNumNeighbors;

    float lambdaForSmoothed;

    float isoValue;
};


struct DiffuseParams
{
    int  smoothingRadiusRatio;
    float smoothingRadius;
    float smoothingRadiusInv;
    float smoothingRadiusSq;
    float anisotropicRadius;

    char minNumNeighbors;

    float coefficient;

    float minWaveCrests;
    float maxWaveCrests;

    float minTrappedAir;
    float maxTrappedAir;

    float minKineticEnergy;
    float maxKineticEnergy;

    float buoyancyControl;
    float dragControl;

    int trappedAirMultiplier;
    int waveCrestsMultiplier;

    int lifeTime;

    float timeStep;
};


struct GridParams
{
    float3 minPos;
    float3 maxPos;

    float3 scminPos;
    float3 scmaxPos;

    int spScale;
    int scScale;
    float sptoscScaleInv;
    
    float spGridSize;
    float scGridSize;

    uint3 spresolution;
    uint3 scresolution;

    int spexpandExtent;
    int scexpandExtent;

    //------------------------------------diffuse-----------------------------------
    float3 minPos_diffuse;
    float3 maxPos_diffuse;

    // float spGridSize_diffuse;
    // uint3 spresolution_diffuse;

    int spexpandExtent_diffuse;

    uint spSize;
    uint scSize;
};


struct ConfigParams
{
    int gpu_id;

    int frameStart;
    int frameEnd;
    int frameStep;

    bool isDiffuse;
    bool isSurface;
    std::string Directory_Param;
    std::string FileName_Param;

    int nzeros;
    std::string fluidPath;
    std::string fluidPrefix;
    
    std::string boundPath;
    std::string boundPrefix;

    std::string surfacePrefix;
    std::string surfacePath;

    std::string diffusePrefix;
    std::string diffusePath;
};


struct FluidParticle
{ 
	float3 pos; 
	float3 vel;
    float3 nor;
    float  rhop;
    float flag;
};


struct SmoothedPos
{ 
	float3 pos; 
};

struct MeanPos
{ 
	float3 pos; 
};

struct DiffuseParticle
{ 
	float3 pos; 
    float3 vel;
    int type;
    int TTL;
    bool life;

    float waveCrest;
    float Ita;
    float energy;
};

struct DiffusePotential
{ 
	float waveCrest;
    float Ita;
    float energy;
};


struct BoundParticle
{ 
    float3 pos; 
    float3 vel;
    float rhop;    
};

struct BoundGrid
{
	bool bound;
};

struct SpatialGrid
{
    bool fluid;
    bool surface;
    bool inner;
    bool bound;
    char classify;
};

struct ScalarValue
{
    float3 vel;
    float vel_;
    float rhop;
};

struct MatrixValue 
{
	float a11, a12, a13;
	float a21, a22, a23;
	float a31, a32, a33;
	float maxvalue;
};

struct IndexRange { uint start, end;};

struct record
{
    float pp;
    float sf;
    float tri;

    float surpar;
    float surver;
    float surcell;

    float trinum;

    float mem;
    float mem_mc;
//----------------------------------

    float pf;
    float po;
    float gen;
    float up;

    float de;

    float isdif;
    float numdif;

    float comde;

    float mem_dif;

    float total;
//-------------------------------------
    float feal;
    float mem_feal;

    float temp;
//-------------------------------------
    float our;
    float im12;
    float mu03;
//-------------------------------------
    float muller;
};


struct Triangle
{
	fVector3 vertices[3];
	fVector3 normals[3];
};

#endif