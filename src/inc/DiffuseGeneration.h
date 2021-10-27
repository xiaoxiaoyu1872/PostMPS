#ifndef DIFFUSEGENERATION_H_
#define DIFFUSEGENERATION_H_

#include "Params.h"
#include "GPUmemory.h"
#include "myCuda.cuh"
#include "FileData.h"

class DiffuseGeneration
{
  public:
      DiffuseGeneration();
      ~DiffuseGeneration();
      void Init(GPUmemory *gMemory, Params *params, FileData *fileData);
      void Destory();
      void runsimulation();
      void Fealrunsimulation();
      void savemiddlefile();
      void savefealfile();
  private:
      GPUmemory* gMemory;
      Params* params;
      FileData* fileData;

      uint NumFreeSurfaceParticles;
      uint NumParticles;

      uint GeneratedNumDiffuseParticles;
      uint OldNumDiffuseParticles;

      uint NumIsDiffuseParticles;
      
      std::vector<DiffuseParticle> mDiffuse;

      void constantMemCopy_Sim();
      void constantMemCopy_Diffuse();
      void constantMemCopy_Grid();

      void processingOfFreesurface(record& record_single);
      void estimatingOfPotention(record& record_single);
      void generatingOfDiffuse(record& record_single);
      void updatingOfDiffuse(record& record_single);

      void extractionOfFreeSurfaceParticles();
      void thrustscan_freeparticles();
      void streamcompact_freeparticles();
      void memallocation_freeparticles();
      void transformmatrices_freeparticles();

      void memallocation_potional();
      void calculationOftrappedair();
      void calculationOfwavecrests();

      void calculationOfnumberofdiffuseparticles();
      void thrustscan_diffuseparticles();
      void streamcompact_diffuseparticles();
      void memallocation_diffuseparticles();
      void calculationOfdiffuseposition();
      void determinationOfdiffusetype();
      
      void memallocation_olddiffuseparticles();

      void deleteAndappendParticles(record& record_single);


//--------------------------------------------------------------
      void Fealtrappedair();
      void Fealgradient();
      void Fealwavecrests();
      void memallocation_feal();
//--------------------------------------------------------------

      std::vector<record> records;
      record record_single;
};

#endif 