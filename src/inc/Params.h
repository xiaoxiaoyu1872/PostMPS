#ifndef _PARAMS_H_
#define _PARAMS_H_
#include "Define.h"
#include "INIReader.h"
class Params
{
  public:
      Params();
      ~Params();
      SimParams     mSimParams;
      SurfaceParams mSurfaceParams;
      DiffuseParams mDiffuseParams;
      GridParams   mGridParams;
      
      ConfigParams mConfigParams;

      void setFilname();
      void setParams();
      void printInfo();

      void setGPUId();
      void printGPUInfo();
};
#endif 
