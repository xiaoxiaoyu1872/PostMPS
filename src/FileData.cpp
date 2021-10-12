#include "FileData.h"
#include <string>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkPolyDataReader.h>
#include <vtkCommand.h>
#include <Params.h>
#include <array>

#include "tinyply.h"
using namespace tinyply;

const int  maxNum     = 8;
const char zero       = '0';

std::string num2str_full(int i)
{
   std::stringstream ss;

   ss << setfill(zero);
   ss << setw(maxNum);
   ss << i;
   return ss.str();
}

class ErrorObserver : public vtkCommand
{
public:
  ErrorObserver():
    Error(false),
    Warning(false),
    ErrorMessage(""),
    WarningMessage("") {}

  static ErrorObserver *New()
  {
    return new ErrorObserver;
  }

  bool GetError() const
  {
    return this->Error;
  }

  bool GetWarning() const
  {
    return this->Warning;
  }

  void Clear()
  {
    this->Error = false;
    this->Warning = false;
    this->ErrorMessage = "";
    this->WarningMessage = "";
  }

  virtual void Execute(vtkObject *vtkNotUsed(caller),
                       unsigned long event,
                       void *calldata)
  {
  switch(event)
    {
    case vtkCommand::ErrorEvent:
      ErrorMessage = static_cast<char *>(calldata);
      this->Error = true;
      break;
    case vtkCommand::WarningEvent:
      WarningMessage = static_cast<char *>(calldata);
      this->Warning = true;
      break;
    }
  }

  std::string GetErrorMessage()
  {
    return ErrorMessage;
  }

  std::string GetWarningMessage()
  {
    return WarningMessage;
  }

private:
  bool        Error;
  bool        Warning;
  std::string ErrorMessage;
  std::string WarningMessage;
};

FileData::FileData(Params *_params, GPUmemory *_gMemory)
{
    params = _params;
    gMemory = _gMemory;
}

void FileData::setExclusionZone(std::string const &fileName) 
{
  exclude = true;
  exFile = fileName;
}


void FileData::loadVTKFile()
{
    ConfigParams mConfigParams = params->mConfigParams;
    std::string fileName = (fs::path(mConfigParams.boundPath) / 
    (mConfigParams.boundPrefix + "All" + ".vtk")).generic_string();

    std::cout << "Opening: " << fileName << std::endl;

    vtkSmartPointer<ErrorObserver> errorObserver =
      vtkSmartPointer<ErrorObserver>::New();
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();

    reader->SetFileName(fileName.c_str());
    reader->AddObserver(vtkCommand::ErrorEvent, errorObserver);
    reader->Update();

    if (errorObserver->GetError())
    {
        std::cerr << "ERROR: the file cannot be loaded." << std::endl;
    }

    vtkPolyData *output = reader->GetOutput();

    vtkDataArray *Points = output->GetPoints()->GetData();

    vtkPointData *pointData = output->GetPointData();

    vtkDataArray *type = pointData->GetArray("Type"); 

    vtkDataArray *vel = pointData->GetArray("Vel");

    vtkDataArray *rhop = pointData->GetArray("Rhop");

    std::vector<FluidParticle> mBoundParticle;
    mBoundParticle.resize(output->GetPoints()->GetNumberOfPoints());

    int NumBoundParticles = 0;
    int NumFlag = 0;
    for (long i = 0; i < output->GetPoints()->GetNumberOfPoints(); i++) 
    {
        double *p = Points->GetTuple(i);
        double *v = vel->GetTuple(i);      
        
        FluidParticle pi;
        pi.pos = {float(p[0]), float(p[1]), float(p[2])};
        pi.vel = {float(v[0]), float(v[1]), float(v[2])};
        pi.rhop = float(rhop->GetTuple(i)[0]);

        if (type->GetTuple(i)[0] > 1.5)  //0: Bound 1: Piston 2: floatingBox
        {
            pi.flag = 0;
            mBoundParticle[NumBoundParticles] = pi;
            NumBoundParticles++;
        }else if(type->GetTuple(i)[0] > 0.5){
            pi.flag = 1;
            mBoundParticle[NumBoundParticles] = pi;
            NumBoundParticles++;
            NumFlag++;
        }
    }

    mBoundParticle.resize(NumBoundParticles);

    gMemory->memAlcandCpy_fluid(mBoundParticle);
    std::vector<FluidParticle>().swap(mBoundParticle); 

    // saveFluidDatfile();
}



void FileData::loadFluidFile(int _frameIndex, float3& motion) 
{
    frameIndex = _frameIndex;
    ConfigParams mConfigParams = params->mConfigParams;
    std::string seqnum(mConfigParams.nzeros, '0');
    std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
    std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
    std::string fileName = (fs::path(mConfigParams.fluidPath) / 
    (mConfigParams.fluidPrefix + seqnum + ".vtk")).generic_string();

    std::cout << "\n== [" << " Step " << frameIndex << " of " << mConfigParams.frameEnd 
    << " ] ===================================================================\n";
    std::cout << "Opening: " << fileName << std::endl;

    vtkSmartPointer<ErrorObserver> errorObserver =
      vtkSmartPointer<ErrorObserver>::New();
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();

    reader->SetFileName(fileName.c_str());
    reader->AddObserver(vtkCommand::ErrorEvent, errorObserver);
    reader->Update();

    if (errorObserver->GetError())
    {
        std::cerr << "ERROR: the file cannot be loaded." << std::endl;
    }

    vtkPolyData *output = reader->GetOutput();

    vtkDataArray *Points = output->GetPoints()->GetData();

    vtkPointData *pointData = output->GetPointData();

    vtkDataArray *pvel = pointData->GetArray("Vel");
    vtkDataArray *rhop = pointData->GetArray("Rhop"); 

    std::vector<FluidParticle> mFluidParticle;
    mFluidParticle.resize(output->GetPoints()->GetNumberOfPoints());

    float3 temp = {0, 0, 0};

    int number = 0;
    for (long i = 0; i < output->GetPoints()->GetNumberOfPoints(); i++) 
    {
        double *p = Points->GetTuple(i);
        double *v = pvel->GetTuple(i);

        FluidParticle pi;
        pi.pos = {float(p[0]), float(p[1]), float(p[2])};
        pi.pos = pi.pos- motion;
        pi.vel = {float(v[0]), float(v[1]), float(v[2])};
        pi.rhop = float(rhop->GetTuple(i)[0]);
        
        mFluidParticle[number] = pi;
        number++;
        temp += pi.pos;
    }
    temp /= number;
    center.push_back(temp);

    mFluidParticle.resize(number);

    float3 max = mFluidParticle[0].pos;
    float3 min = mFluidParticle[0].pos;

    for(int i = 1; i < number; i++){
      if(max.x <= mFluidParticle[i].pos.x)
        max.x = mFluidParticle[i].pos.x;
      if(max.y <= mFluidParticle[i].pos.y)
        max.y = mFluidParticle[i].pos.y;
      if(max.z <= mFluidParticle[i].pos.z)
        max.z = mFluidParticle[i].pos.z;

      if(min.x >= mFluidParticle[i].pos.x)
        min.x = mFluidParticle[i].pos.x;
      if(min.y >= mFluidParticle[i].pos.y)
        min.y = mFluidParticle[i].pos.y;
      if(min.z >= mFluidParticle[i].pos.z)
        min.z = mFluidParticle[i].pos.z;
    }

#ifdef NOBOUND
    params->mGridParams.scminPos = min - 2*params->mSurfaceParams.smoothingRadius;
    params->mGridParams.scmaxPos = max + 2*params->mSurfaceParams.smoothingRadius;

    params->mGridParams.minPos = min - 4*params->mSurfaceParams.smoothingRadius;
    params->mGridParams.maxPos = max + 4*params->mSurfaceParams.smoothingRadius;

    params->mGridParams.spresolution = make_uint3(
    ((params->mGridParams.maxPos.x - params->mGridParams.minPos.x) / params->mGridParams.spGridSize),
    ((params->mGridParams.maxPos.y - params->mGridParams.minPos.y) / params->mGridParams.spGridSize),
    ((params->mGridParams.maxPos.z - params->mGridParams.minPos.z) / params->mGridParams.spGridSize));

    params->mGridParams.scresolution = make_uint3(
    ((params->mGridParams.scmaxPos.x - params->mGridParams.scminPos.x) / params->mGridParams.scGridSize) + 0.5,
    ((params->mGridParams.scmaxPos.y - params->mGridParams.scminPos.y) / params->mGridParams.scGridSize) + 0.5,
    ((params->mGridParams.scmaxPos.z - params->mGridParams.scminPos.z) / params->mGridParams.scGridSize) + 0.5);

    params->mGridParams.spSize = 
        params->mGridParams.spresolution.x * params->mGridParams.spresolution.y * params->mGridParams.spresolution.z;
    params->mGridParams.scSize =  
          params->mGridParams.scresolution.x *  params->mGridParams.scresolution.y *  params->mGridParams.scresolution.z;
#endif

#ifdef PRINT
    std::cout << "posMin " << params->mGridParams.minPos << std::endl;
    std::cout << "posMax " << params->mGridParams.maxPos << std::endl;

    std::cout << "spextent " << params->mGridParams.spexpandExtent << std::endl;
    std::cout << "scextent " << params->mGridParams.scexpandExtent << std::endl;

    std::cout << "spextent_diffuse " << params->mGridParams.spexpandExtent_diffuse << std::endl;

    std::cout << "spresolution " << params->mGridParams.spresolution << std::endl;
    std::cout << "scresolution " << params->mGridParams.scresolution << std::endl;
    
    std::cout << "spsize " << params->mGridParams.spSize << std::endl;
    std::cout << "scsize " << params->mGridParams.scSize << std::endl;

    std::cout << "spGridsize " << params->mGridParams.spGridSize << std::endl;
    std::cout << "scGridsize " << params->mGridParams.scGridSize << std::endl;
#endif

    gMemory->memAlcandCpy_fluid(mFluidParticle);
    uint NumParticles = mFluidParticle.size();
    std::vector<FluidParticle>().swap(mFluidParticle);
    std::cout << "NumParticles: " << NumParticles << std::endl;
    // saveFluidDatfile();
}

void FileData::saveCenterfile()
{
  ConfigParams mConfigParams = params->mConfigParams;
  std::string basename = "Centen";
	std::string path = params->mConfigParams.fluidPath + std::string(basename) + ".csv";
  std::ofstream file;
  file.open(path.c_str(), std::ios::out);
  std::cout << "path = " << path << std::endl;

  file<< setw(10)<< "x" << " "
			<< setw(10)<< "y" << " "
			<< setw(10)<< "z" << " " 
      << std::endl;

  for (int i = 0; i < center.size(); i++)
  {
    int index = i;
    auto pos =  center[index];

    file  << setiosflags(ios::fixed)<<setprecision(6) 
          << setw(10)<< pos.x << " "
          << setw(10)<< pos.y << " "
          << setw(10)<< pos.z << " "
          << std::endl;
  }
}

void FileData::loadMotionFile(int _frameIndex, float3& motion) 
{
    frameIndex = _frameIndex;
    ConfigParams mConfigParams = params->mConfigParams;
    std::string seqnum(mConfigParams.nzeros, '0');
    std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
    std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
    std::string fileName = (fs::path(mConfigParams.boundPath) / 
    (mConfigParams.boundPrefix + seqnum + ".vtk")).generic_string();

    std::cout << "Opening MotionFile: " << fileName << std::endl;

    vtkSmartPointer<ErrorObserver> errorObserver =
      vtkSmartPointer<ErrorObserver>::New();
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();

    reader->SetFileName(fileName.c_str());
    reader->AddObserver(vtkCommand::ErrorEvent, errorObserver);
    reader->Update();

    if (errorObserver->GetError())
    {
        std::cerr << "ERROR: the file cannot be loaded." << std::endl;
    }

    vtkPolyData *output = reader->GetOutput();

    vtkDataArray *Points = output->GetPoints()->GetData();

    vtkPointData *pointData = output->GetPointData();

    vtkDataArray *pvel = pointData->GetArray("Vel");

    std::vector<BoundParticle> mBoundParticle;
    mBoundParticle.resize(output->GetPoints()->GetNumberOfPoints());

    float3 average = {0,0,0};
    for (long i = 0; i < output->GetPoints()->GetNumberOfPoints(); i++) 
    {
        double *p = Points->GetTuple(i);

        BoundParticle pi;
        pi.pos = {float(p[0]), float(p[1]), float(p[2])};

        mBoundParticle[i] = pi;

        average+=pi.pos;
    }

    average = average/output->GetPoints()->GetNumberOfPoints();
    motion = average;

    // saveMotionVtkfile(mBoundParticle, frameIndex);
    std::vector<BoundParticle>().swap(mBoundParticle);
}


int readnum(std::string infilename) {
    int numPoints = 0;  
    std::string tmp;

    ifstream infile_line; 
    infile_line.open(infilename);

    if (infile_line.fail())  
    {  
        std::cout<<"Could not open"<<infilename<<std::endl;
    }  

    while(getline(infile_line,tmp))
    {       
        numPoints++;
    }

    return numPoints;
}


void FileData::loadFluidDatFile(int _frameIndex, float3& motion)
{
  frameIndex = _frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.fluidPath) / 
  (mConfigParams.fluidPrefix + seqnum + ".dat")).generic_string();

  std::cout << "Opening: " << fileName << std::endl;
  std::cout << "\n\n== [" << " Step " << frameIndex << " of " << mConfigParams.frameEnd 
  << " ] ===================================================================\n";

  int numPoints = readnum(fileName);
  float x,y,z;
  float vx,vy,vz;
  float rhop;
  std::ifstream in(fileName, std::ios::in);

  std::vector<FluidParticle> mFluidParticle;
  mFluidParticle.resize(numPoints);
  int number = 0;

  for (int i = 0; i < numPoints; i++) {
      in  >> x >> y >> z
          >> vx >> vy >> vz 
          >> rhop;
      FluidParticle pi;

      pi.pos.x = x;
      pi.pos.y = y;
      pi.pos.z = z;

      pi.vel.x = vx;
      pi.vel.y = vy;
      pi.vel.z = vz;

      pi.rhop = rhop;
      mFluidParticle[number] = pi;
      number++;
  }

  mFluidParticle.resize(number);

  float3 max = mFluidParticle[0].pos;
  float3 min = mFluidParticle[0].pos;

  for(int i = 1; i < number; i++){
    if(max.x <= mFluidParticle[i].pos.x)
      max.x = mFluidParticle[i].pos.x;
    if(max.y <= mFluidParticle[i].pos.y)
      max.y = mFluidParticle[i].pos.y;
    if(max.z <= mFluidParticle[i].pos.z)
      max.z = mFluidParticle[i].pos.z;

    if(min.x >= mFluidParticle[i].pos.x)
      min.x = mFluidParticle[i].pos.x;
    if(min.y >= mFluidParticle[i].pos.y)
      min.y = mFluidParticle[i].pos.y;
    if(min.z >= mFluidParticle[i].pos.z)
      min.z = mFluidParticle[i].pos.z;
  }

#ifdef NOBOUND
  params->mGridParams.scminPos = min - 2*params->mSurfaceParams.smoothingRadius;
  params->mGridParams.scmaxPos = max + 2*params->mSurfaceParams.smoothingRadius;

  params->mGridParams.minPos = min - 4*params->mSurfaceParams.smoothingRadius;
  params->mGridParams.maxPos = max + 4*params->mSurfaceParams.smoothingRadius;

  params->mGridParams.spresolution = make_uint3(
  ((params->mGridParams.maxPos.x - params->mGridParams.minPos.x) / params->mGridParams.spGridSize),
  ((params->mGridParams.maxPos.y - params->mGridParams.minPos.y) / params->mGridParams.spGridSize),
  ((params->mGridParams.maxPos.z - params->mGridParams.minPos.z) / params->mGridParams.spGridSize));

  params->mGridParams.scresolution = make_uint3(
  ((params->mGridParams.scmaxPos.x - params->mGridParams.scminPos.x) / params->mGridParams.scGridSize) + 0.5,
  ((params->mGridParams.scmaxPos.y - params->mGridParams.scminPos.y) / params->mGridParams.scGridSize) + 0.5,
  ((params->mGridParams.scmaxPos.z - params->mGridParams.scminPos.z) / params->mGridParams.scGridSize) + 0.5);

  params->mGridParams.spSize = 
    params->mGridParams.spresolution.x * params->mGridParams.spresolution.y * params->mGridParams.spresolution.z;
  params->mGridParams.scSize =  
      params->mGridParams.scresolution.x *  params->mGridParams.scresolution.y *  params->mGridParams.scresolution.z;
#endif

#ifdef PRINT
  std::cout << "posMin " << params->mGridParams.minPos << std::endl;
  std::cout << "posMax " << params->mGridParams.maxPos << std::endl;

  std::cout << "spextent " << params->mGridParams.spexpandExtent << std::endl;
  std::cout << "scextent " << params->mGridParams.scexpandExtent << std::endl;

  std::cout << "spextent_diffuse " << params->mGridParams.spexpandExtent_diffuse << std::endl;

  std::cout << "spresolution " << params->mGridParams.spresolution << std::endl;
  std::cout << "scresolution " << params->mGridParams.scresolution << std::endl;

  std::cout << "spsize " << params->mGridParams.spSize << std::endl;
  std::cout << "scsize " << params->mGridParams.scSize << std::endl;

  std::cout << "spGridsize " << params->mGridParams.spGridSize << std::endl;
  std::cout << "scGridsize " << params->mGridParams.scGridSize << std::endl;
#endif

  gMemory->memAlcandCpy_fluid(mFluidParticle);
  std::vector<FluidParticle>().swap(mFluidParticle);
}


void FileData::loadMotionDatFile(int _frameIndex, float3& motion)
{
  frameIndex = _frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.boundPath) / 
  (mConfigParams.boundPrefix + seqnum + ".dat")).generic_string();

  std::cout << "Opening MotionFile: " << fileName << std::endl;

  int numPoints = readnum(fileName);
  float x,y,z;
  float vx,vy,vz;
  float rhop;
  std::ifstream in(fileName, std::ios::in);

  std::vector<BoundParticle> mBoundParticle;
  mBoundParticle.resize(numPoints);

  float3 average = {0,0,0};
  for (int i=0; i < numPoints; i++) {
      in  >> x >> y >> z
          >> vx >> vy >> vz
          >> rhop;
      BoundParticle pi;

      pi.pos.x = x;
      pi.pos.y = y;
      pi.pos.z = z;

      pi.vel.x = x;
      pi.vel.y = y;
      pi.vel.z = z;

      mBoundParticle[i] = pi;

      average += pi.pos;
  }
  average = average/numPoints;
  motion = average;

  // saveMotionVtkfile(mBoundParticle, frameIndex);
  std::vector<BoundParticle>().swap(mBoundParticle);
}






void FileData::loadFluidFiledat(int frameIndex)
{
  ConfigParams mConfigParams = params->mConfigParams;
  std::string infilename = "TECResult";
  infilename = infilename + num2str_full(frameIndex) + ".out";
  infilename = (fs::path(mConfigParams.fluidPath)/infilename);   

  int numPoints = readnum(infilename);

  float x,y,z;
  float vx,vy,vz;
  float rhop;

  std::ifstream in(infilename, std::ios::in);

  std::vector<FluidParticle> mFluidParticle;
  mFluidParticle.resize(numPoints);

  for (int i=0; i < numPoints; i++) {
      in  >> x >> y >> z
          // >> vx >> vy >> vz 
          >> rhop;
      FluidParticle pi;

      pi.pos.x = x;
      pi.pos.y = y;
      pi.pos.z = z;

      // pi.vel.x = x;
      // pi.vel.y = y;
      // pi.vel.z = z;

      pi.rhop = 1000;

      mFluidParticle[i] = pi;
  }

  // gMemory->memAlcandCpy_fluid(mFluidParticle);

  saveFluidVTKfile(mFluidParticle, frameIndex);

  std::vector<FluidParticle>().swap(mFluidParticle);
}


void FileData::loadBoundFile()
{
    ConfigParams mConfigParams = params->mConfigParams;
    std::string fileName = (fs::path(mConfigParams.boundPath) / 
    (mConfigParams.boundPrefix + "Bound" + ".vtk")).generic_string();

    std::cout << "Opening: " << fileName << std::endl;

    vtkSmartPointer<ErrorObserver> errorObserver =
      vtkSmartPointer<ErrorObserver>::New();
    vtkSmartPointer<vtkPolyDataReader> reader =
        vtkSmartPointer<vtkPolyDataReader>::New();

    reader->SetFileName(fileName.c_str());
    reader->AddObserver(vtkCommand::ErrorEvent, errorObserver);
    reader->Update();

    if (errorObserver->GetError())
    {
        std::cerr << "ERROR: the file cannot be loaded." << std::endl;
    }

    vtkPolyData *output = reader->GetOutput();

    vtkDataArray *Points = output->GetPoints()->GetData();

    vtkPointData *pointData = output->GetPointData();

    vtkDataArray *type = pointData->GetArray("Type"); 

    std::vector<BoundParticle> mBoundParticle;
    mBoundParticle.resize(output->GetPoints()->GetNumberOfPoints());

    int NumBoundParticles = 0;
    for (long i = 0; i < output->GetPoints()->GetNumberOfPoints(); i++) 
    {
        double *p = Points->GetTuple(i);

        BoundParticle pi;
        pi.pos = {float(p[0]), float(p[1]), float(p[2])};

        if (type->GetTuple(i)[0] < 0.5)  //0: Bound 1: Piston 2: floatingBox
        {
            mBoundParticle[NumBoundParticles] = pi;
            NumBoundParticles++;
        }
    }
    
    gMemory->memAlcandCpy_bound(mBoundParticle);

    std::vector<BoundParticle>().swap(mBoundParticle);
}



void FileData::saveSurfaceVTKfile()
{
  size_t nums = gMemory->NumSurfaceMeshVertices;
  positions.resize(nums);
  normals.resize(nums);
  scalarValue.resize(nums);

  checkCudaErrors(cudaMemcpy(static_cast<void*>(positions.data()), gMemory->dVertex,
		sizeof(Vertex) * nums, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(static_cast<void*>(normals.data()), gMemory->dNormal,
		sizeof(Normal) * nums, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(static_cast<void*>(scalarValue.data()), gMemory->dScalarValue,
		sizeof(ScalarValue) * nums, cudaMemcpyDeviceToHost));

	std::cout << "Number of vertices : " << nums << std::endl;
	std::cout << "Number of face: " << nums/3 << std::endl;

  Points = vtkSmartPointer<vtkPoints>::New();
  Points->SetNumberOfPoints(nums);

  Normals = vtkSmartPointer<vtkFloatArray>::New();
  Normals->SetName("Normal");
  Normals->SetNumberOfComponents(3);
  Normals->SetNumberOfTuples(nums);

  Rhop = vtkSmartPointer<vtkFloatArray>::New();
  Rhop->SetName("Rhop");
  Rhop->SetNumberOfComponents(1);
  Rhop->SetNumberOfTuples(nums);

  Vel_ = vtkSmartPointer<vtkFloatArray>::New();
  Vel_->SetName("VelLength");
  Vel_->SetNumberOfComponents(1);
  Vel_->SetNumberOfTuples(nums);

  Vel = vtkSmartPointer<vtkFloatArray>::New();
  Vel->SetName("Vel");
  Vel->SetNumberOfComponents(3);
  Vel->SetNumberOfTuples(nums);

  for (int i = 0; i < nums; i++)
	{
    auto p =  positions[i];
    auto n =  normals[i];
    auto rhop = scalarValue[i].rhop;
    auto vel_ = scalarValue[i].vel_;
    auto vel = scalarValue[i].vel;
		Points->InsertPoint(static_cast<vtkIdType>(i), p.x, p.y, p.z);
    Normals->InsertTuple3(static_cast<vtkIdType>(i), -n.x , -n.y, -n.z);

    Rhop->InsertTuple1(static_cast<vtkIdType>(i), rhop);
    Vel_->InsertTuple1(static_cast<vtkIdType>(i), vel_);
    Vel->InsertTuple3(static_cast<vtkIdType>(i), vel.x, vel.y, vel.z);

	}

  triangles =  vtkSmartPointer< vtkCellArray >::New();

  for (int i = 0; i < nums; i += 3)
	{
    vtkSmartPointer< vtkTriangle > triangle = vtkSmartPointer< vtkTriangle >::New();
    for (int j = 0; j < 3; j++)
    {
       triangle->GetPointIds()->SetId(j, i + j);
    }
    triangles->InsertNextCell(triangle);
	}

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();

  polydata->SetPoints(Points);
  polydata->SetPolys(triangles);
  polydata->GetPointData()->AddArray(Normals);

  polydata->GetPointData()->AddArray(Rhop);
  polydata->GetPointData()->AddArray(Vel_);
  polydata->GetPointData()->AddArray(Vel);

  vtkSmartPointer<vtkPolyDataWriter> writer =
      vtkSmartPointer<vtkPolyDataWriter>::New();

  frameIndex = frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.surfacePath) / 
  (mConfigParams.surfacePrefix + seqnum + ".vtk")).generic_string();

  // writer->SetFileTypeToBinary();
  writer->SetFileTypeToASCII();
  writer->SetInputData(polydata);

  writer->SetFileName(fileName.c_str());
  writer->Write();
}


void FileData::saveFlagVTKfile()
{  

  size_t numsFlu = gMemory->NumParticles;
  size_t numsSur = gMemory->NumSurfaceParticles;

  std::vector<FluidParticle> fluidParticle;
  std::vector<Index> surfaceIndex;

  fluidParticle.resize(numsFlu);
  surfaceIndex.resize(numsSur);

  checkCudaErrors(cudaMemcpy(static_cast<void*>(fluidParticle.data()), gMemory->dFluidParticle,
		sizeof(FluidParticle) * numsFlu, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(static_cast<void*>(surfaceIndex.data()), gMemory->dSurfaceParticlesIndex,
		sizeof(Index) * numsSur, cudaMemcpyDeviceToHost));


  int numflag = 0;
  for (int i = 0; i < numsSur; i++)
	{
    int index = surfaceIndex[i];
    auto pos =  fluidParticle[index].pos;
    if(fluidParticle[index].flag){
      numflag++;
    }
	}  

  vtkSmartPointer<vtkPoints> FlagParticles1;
  FlagParticles1 = vtkSmartPointer<vtkPoints>::New();
  FlagParticles1->SetNumberOfPoints(numflag);


  vtkSmartPointer<vtkFloatArray> flag1 = vtkSmartPointer<vtkFloatArray>::New();
  flag1->SetName("Flag");
  flag1->SetNumberOfComponents(1);
  flag1->SetNumberOfTuples(numflag);

  vtkSmartPointer<vtkFloatArray> size1 = vtkSmartPointer<vtkFloatArray>::New();
  size1->SetName("Size");
  size1->SetNumberOfComponents(1);
  size1->SetNumberOfTuples(numflag);


  vtkSmartPointer<vtkFloatArray> velocity1 = vtkSmartPointer<vtkFloatArray>::New();
  velocity1->SetName("Velocity");
  velocity1->SetNumberOfComponents(3);
  velocity1->SetNumberOfTuples(numflag);


  vtkSmartPointer<vtkPoints> FlagParticles2;
  FlagParticles2 = vtkSmartPointer<vtkPoints>::New();
  FlagParticles2->SetNumberOfPoints(numsSur - numflag);

  vtkSmartPointer<vtkFloatArray> flag2 = vtkSmartPointer<vtkFloatArray>::New();
  flag2->SetName("Flag");
  flag2->SetNumberOfComponents(1);
  flag2->SetNumberOfTuples(numsSur-numflag);

  vtkSmartPointer<vtkFloatArray> size2 = vtkSmartPointer<vtkFloatArray>::New();
  size2->SetName("Size");
  size2->SetNumberOfComponents(1);
  size2->SetNumberOfTuples(numsSur-numflag);

  vtkSmartPointer<vtkFloatArray> velocity2 = vtkSmartPointer<vtkFloatArray>::New();
  velocity2->SetName("Velocity");
  velocity2->SetNumberOfComponents(3);
  velocity2->SetNumberOfTuples(numsSur-numflag);


  int n = 0, m = 0;
  for (int i = 0; i < numsSur; i++)
	{
    int index = surfaceIndex[i];
    auto pos =  fluidParticle[index].pos;
    auto vel =  fluidParticle[index].vel;

    if(fluidParticle[index].flag){
      FlagParticles1->InsertPoint(static_cast<vtkIdType>(n), pos.x, pos.y, pos.z);
      velocity1->InsertTuple3(static_cast<vtkIdType>(n), vel.x , vel.y, vel.z);
      // flag1->InsertTuple1(static_cast<vtkIdType>(n), fluidParticle[index].flag);
      size1->InsertTuple1(static_cast<vtkIdType>(n), params->mDiffuseParams.smoothingRadius/5);
      n++;
    }else{
      FlagParticles2->InsertPoint(static_cast<vtkIdType>(m), pos.x, pos.y, pos.z);
      velocity2->InsertTuple3(static_cast<vtkIdType>(m), vel.x , vel.y, vel.z);
      // flag2->InsertTuple1(static_cast<vtkIdType>(m), fluidParticle[index].flag);
      size2->InsertTuple1(static_cast<vtkIdType>(m), params->mDiffuseParams.smoothingRadius/5);
      m++;
    }
	}


  vtkSmartPointer<vtkPolyData> polydata1 = vtkSmartPointer<vtkPolyData>::New();
  polydata1->SetPoints(FlagParticles1);

  vtkSmartPointer<vtkPolyData> polydata2 = vtkSmartPointer<vtkPolyData>::New();
  polydata2->SetPoints(FlagParticles2);


  vtkSmartPointer<vtkCellArray> vertices1 = vtkSmartPointer<vtkCellArray>::New();
  for (long i = 0; i < numflag; ++i){
    vtkIdType pt[] = {i};
    vertices1->InsertNextCell(1, pt);
  }

  vtkSmartPointer<vtkCellArray> vertices2 = vtkSmartPointer<vtkCellArray>::New();
  for (long i = 0; i < numsSur-numflag; ++i){
    vtkIdType pt[] = {i};
    vertices1->InsertNextCell(1, pt);
  }

  polydata1->SetVerts(vertices1);
  polydata1->GetPointData()->AddArray(velocity1);
  // polydata1->GetPointData()->AddArray(flag1);
  polydata1->GetPointData()->AddArray(size1);

  polydata2->SetVerts(vertices2);
  polydata2->GetPointData()->AddArray(velocity2);
  // polydata2->GetPointData()->AddArray(flag2);
  polydata2->GetPointData()->AddArray(size2);

//---------------------------------------------------------------

  vtkSmartPointer<vtkPolyDataWriter> writer1 =
      vtkSmartPointer<vtkPolyDataWriter>::New();

  writer1->SetFileTypeToBinary();
  writer1->SetInputData(polydata1);

  frameIndex = frameIndex;
  ConfigParams mConfigParams1 = params->mConfigParams;

  mConfigParams1.diffusePrefix = "BlenderFlag1_";
  
  std::string seqnum(mConfigParams1.nzeros, '0');
  std::string formats1 = std::string("%.") + std::to_string(mConfigParams1.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats1.c_str(), frameIndex);
  std::string fileName1 = (fs::path(mConfigParams1.diffusePath) / 
  (mConfigParams1.diffusePrefix + seqnum + ".vtk")).generic_string();

  writer1->SetFileName(fileName1.c_str());
  writer1->Write();

//---------------------------------------------------------------

  vtkSmartPointer<vtkPolyDataWriter> writer2 =
    vtkSmartPointer<vtkPolyDataWriter>::New();

  writer2->SetFileTypeToBinary();
  writer2->SetInputData(polydata2);

  frameIndex = frameIndex;
  ConfigParams mConfigParams2 = params->mConfigParams;

  mConfigParams2.diffusePrefix = "BlenderFlag_";
  
  std::string seqnum2(mConfigParams2.nzeros, '0');
  std::string formats2 = std::string("%.") + std::to_string(mConfigParams2.nzeros) + std::string("d");
  std::sprintf(&seqnum2[0], formats2.c_str(), frameIndex);
  std::string fileName2 = (fs::path(mConfigParams2.diffusePath) / 
  (mConfigParams2.diffusePrefix + seqnum2 + ".vtk")).generic_string();

  writer2->SetFileName(fileName2.c_str());
  writer2->Write();

}



//--------------------------------------------------------------------------------

void FileData::saveDiffuseVTKfile(std::vector<DiffuseParticle> _diffuse, uint type)
{  
  std::vector<DiffuseParticle> diffuse = _diffuse;

  DiffuseParticles = vtkSmartPointer<vtkPoints>::New();
  DiffuseParticles->SetNumberOfPoints(diffuse.size());

  vtkSmartPointer<vtkFloatArray> velocity = vtkSmartPointer<vtkFloatArray>::New();
  velocity->SetName("Velocity");
  velocity->SetNumberOfComponents(3);
  velocity->SetNumberOfTuples(diffuse.size());

  vtkSmartPointer<vtkFloatArray> size = vtkSmartPointer<vtkFloatArray>::New();
  size->SetName("Size");
  size->SetNumberOfComponents(1);
  size->SetNumberOfTuples(diffuse.size());

  vtkSmartPointer<vtkFloatArray> energy = vtkSmartPointer<vtkFloatArray>::New();
  energy->SetName("energy");
  energy->SetNumberOfComponents(1);
  energy->SetNumberOfTuples(diffuse.size());

  vtkSmartPointer<vtkFloatArray> Ita = vtkSmartPointer<vtkFloatArray>::New();
  Ita->SetName("Ita");
  Ita->SetNumberOfComponents(1);
  Ita->SetNumberOfTuples(diffuse.size());

  vtkSmartPointer<vtkFloatArray> waveCrest = vtkSmartPointer<vtkFloatArray>::New();
  waveCrest->SetName("waveCrest");
  waveCrest->SetNumberOfComponents(1);
  waveCrest->SetNumberOfTuples(diffuse.size());

  for (int i = 0; i < diffuse.size(); i++)
	{
    auto pos =  diffuse[i].pos;
    auto vel =  diffuse[i].vel;

		DiffuseParticles->InsertPoint(static_cast<vtkIdType>(i), pos.x, pos.y, pos.z);
    velocity->InsertTuple3(static_cast<vtkIdType>(i), vel.x , vel.y, vel.z);
    size->InsertTuple1(static_cast<vtkIdType>(i), params->mDiffuseParams.smoothingRadius/10);

    energy->InsertTuple1(static_cast<vtkIdType>(i), diffuse[i].energy);
    Ita->InsertTuple1(static_cast<vtkIdType>(i), diffuse[i].Ita);
    waveCrest->InsertTuple1(static_cast<vtkIdType>(i), diffuse[i].waveCrest);
	}

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(DiffuseParticles);

  vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
  for (long i = 0; i < diffuse.size(); ++i){
    vtkIdType pt[] = {i};
    vertices->InsertNextCell(1, pt);
  }
  polydata->SetVerts(vertices);
  polydata->GetPointData()->AddArray(velocity);
  polydata->GetPointData()->AddArray(size);

  polydata->GetPointData()->AddArray(energy);
  polydata->GetPointData()->AddArray(Ita);
  polydata->GetPointData()->AddArray(waveCrest);

  vtkSmartPointer<vtkPolyDataWriter> writer =
      vtkSmartPointer<vtkPolyDataWriter>::New();

  writer->SetFileTypeToBinary();
  writer->SetInputData(polydata);

  frameIndex = frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;
  if (type == 0)
    mConfigParams.diffusePrefix = "PartSpary_";
  if (type == 1)
    mConfigParams.diffusePrefix = "PartFoam_";
  if (type == 2)
    mConfigParams.diffusePrefix = "PartBubble_";  

  if (type == 3)
    mConfigParams.diffusePrefix = "PartSurface_";

  if (type == 4)
    mConfigParams.diffusePrefix = "PartDebug_";
  
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.diffusePath) / 
  (mConfigParams.diffusePrefix + seqnum + ".vtk")).generic_string();

  writer->SetFileName(fileName.c_str());
  writer->Write();
}

void FileData::saveBoundDatfile()
{
  size_t numsFlu = gMemory->NumParticles;
  size_t numsSur = gMemory->NumSurfaceParticles;

  std::vector<FluidParticle> fluidParticle;
  // std::vector<Index> surfaceIndex;

  fluidParticle.resize(numsFlu);
  // surfaceIndex.resize(numsSur);

  checkCudaErrors(cudaMemcpy(static_cast<void*>(fluidParticle.data()), gMemory->dFluidParticle,
		sizeof(FluidParticle) * numsFlu, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(static_cast<void*>(surfaceIndex.data()), gMemory->dSurfaceParticlesIndex,
	// 	sizeof(Index) * numsSur, cudaMemcpyDeviceToHost));

  std::string basename = "Bound";
	std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
  std::ofstream file;
  file.open(path.c_str(), std::ios::out);

  file<< setw(10)<< "x" << " "
			<< setw(10)<< "y" << " "
			<< setw(10)<< "z" << " "
			<< setw(10)<< "vx" << " "
			<< setw(10)<< "vy" << " "
			<< setw(10)<< "vz" << " "
			<< setw(10)<< "nx" << " "
			<< setw(10)<< "ny" << " "
			<< setw(10)<< "nz" << " "
			<< setw(10)<< "rhop" << " "
			<< std::endl;

  std::cout << "numsSur = " << numsSur << std::endl;
  std::cout << "numsFlu = " << numsFlu << std::endl;

  for (int i = 0; i < numsFlu; i++)
  {
    // int index = surfaceIndex[i];
    int index = i;
    auto pos =  fluidParticle[index].pos;
    auto nor =  fluidParticle[index].nor;
    auto vel = fluidParticle[index].vel;
    auto rhop = fluidParticle[index].rhop;
    auto flag = fluidParticle[index].flag;

    if(flag < 0.5)
    {
      continue;
    }

    file  << setiosflags(ios::fixed)<<setprecision(6) 
          << setw(10)<< pos.x << " "
          << setw(10)<< pos.y << " "
          << setw(10)<< pos.z << " "
          << setw(10)<< vel.x << " "
          << setw(10)<< vel.y << " "
          << setw(10)<< vel.z << " "
          << setw(10)<< nor.x << " "
          << setw(10)<< nor.y << " "
          << setw(10)<< nor.z << " "
          << setw(10)<< rhop << " "
          << std::endl;
  }

}


void FileData::saveFluidDatfile()
{
  size_t numsFlu = gMemory->NumParticles;
  std::vector<FluidParticle> fluidParticle;

  fluidParticle.resize(numsFlu);

  checkCudaErrors(cudaMemcpy(static_cast<void*>(fluidParticle.data()), gMemory->dFluidParticle,
		sizeof(FluidParticle) * numsFlu, cudaMemcpyDeviceToHost));

  ConfigParams mConfigParams = params->mConfigParams;
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.boundPath) / 
  (mConfigParams.fluidPrefix + seqnum + ".dat")).generic_string();

  // std::string basename = "Fluid";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".dat";
  std::ofstream file;
  file.open(fileName.c_str(), std::ios::out);

  // file<< setw(10)<< "x" << " "
	// 		<< setw(10)<< "y" << " "
	// 		<< setw(10)<< "z" << " "
	// 		<< setw(10)<< "vx" << " "
	// 		<< setw(10)<< "vy" << " "
	// 		<< setw(10)<< "vz" << " "
	// 		<< setw(10)<< "rhop" << " "
	// 		<< std::endl;

  for (int i = 0; i < numsFlu; i++)
  {
    int index = i;
    auto pos =  fluidParticle[index].pos;
    auto vel = fluidParticle[index].vel;
    auto rhop = fluidParticle[index].rhop;
    auto flag = fluidParticle[index].flag;

    file  << setiosflags(ios::fixed)<<setprecision(6) 
          << setw(10)<< pos.x << " "
          << setw(10)<< pos.y << " "
          << setw(10)<< pos.z << " "
          << setw(10)<< vel.x << " "
          << setw(10)<< vel.y << " "
          << setw(10)<< vel.z << " "
          << setw(10)<< rhop << " "
          << std::endl;
  }
}

void FileData::saveSurfaceParticleVTKfile()
{
  size_t numsFlu = gMemory->NumParticles;
  size_t numsSur = gMemory->NumSurfaceParticles;

  std::vector<FluidParticle> fluidParticle;
  std::vector<Index> surfaceIndex;

  fluidParticle.resize(numsFlu);
  surfaceIndex.resize(numsSur);

  checkCudaErrors(cudaMemcpy(static_cast<void*>(fluidParticle.data()), gMemory->dFluidParticle,
		sizeof(FluidParticle) * numsFlu, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(static_cast<void*>(surfaceIndex.data()), gMemory->dSurfaceParticlesIndex,
		sizeof(Index) * numsSur, cudaMemcpyDeviceToHost));

  vtkSmartPointer<vtkPoints> FluidParticles;

  FluidParticles = vtkSmartPointer<vtkPoints>::New();
  FluidParticles->SetNumberOfPoints(numsSur);

  vtkSmartPointer<vtkFloatArray> Flag = vtkSmartPointer<vtkFloatArray>::New();
  Flag->SetName("Flag");
  Flag->SetNumberOfComponents(1);
  Flag->SetNumberOfTuples(numsSur);

  for (int i = 0; i < numsSur; i++)
	{
    int index = surfaceIndex[i];
    auto pos =  fluidParticle[index].pos;
    auto flag = fluidParticle[index].flag;

		FluidParticles->InsertPoint(static_cast<vtkIdType>(i), pos.x, pos.y, pos.z);
    Flag->InsertTuple1(static_cast<vtkIdType>(i), flag);

	}

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(FluidParticles);

  vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
  for (long i = 0; i < numsSur; ++i){
    vtkIdType pt[] = {i};
    vertices->InsertNextCell(1, pt);
  }

  polydata->SetVerts(vertices);
  polydata->GetPointData()->AddArray(Flag);

  vtkSmartPointer<vtkPolyDataWriter> writer =
      vtkSmartPointer<vtkPolyDataWriter>::New();

  writer->SetFileTypeToBinary();
  writer->SetInputData(polydata);

  frameIndex = frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;

  mConfigParams.diffusePrefix = "PartFlag_";
  
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.diffusePath) / 
  (mConfigParams.diffusePrefix + seqnum + ".vtk")).generic_string();

  writer->SetFileName(fileName.c_str());
  writer->Write();

}




void FileData::saveFluidVTKfile(std::vector<FluidParticle>& fParticle, int frame)
{  
  std::vector<FluidParticle> diffuse = fParticle;

  DiffuseParticles = vtkSmartPointer<vtkPoints>::New();
  DiffuseParticles->SetNumberOfPoints(diffuse.size());

  for (int i = 0; i < diffuse.size(); i++)
	{
    auto pos =  diffuse[i].pos;

		DiffuseParticles->InsertPoint(static_cast<vtkIdType>(i), pos.x, pos.y, pos.z);
	}

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(DiffuseParticles);

  vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
  for (long i = 0; i < diffuse.size(); ++i){
    vtkIdType pt[] = {i};
    vertices->InsertNextCell(1, pt);
  }
  polydata->SetVerts(vertices);


  vtkSmartPointer<vtkPolyDataWriter> writer =
      vtkSmartPointer<vtkPolyDataWriter>::New();

  writer->SetFileTypeToBinary();
  writer->SetInputData(polydata);

  frameIndex = frame;
  ConfigParams mConfigParams = params->mConfigParams;

  mConfigParams.diffusePrefix = "PFluid_";
  
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.fluidPath) / 
  (mConfigParams.diffusePrefix + seqnum + ".vtk")).generic_string();

  writer->SetFileName(fileName.c_str());
  writer->Write();
}

void FileData::saveSurfacePLYfile()
{
	size_t nums = gMemory->NumSurfaceMeshVertices;

	std::vector<Vertex> positions;
	std::vector<Normal> normals;
	std::vector<VertexIndex> verindex;

	positions.resize(nums);
	normals.resize(nums);
	verindex.resize(nums/3);

	if (nums == 0)
	{
		std::cerr << "Nothing produced.\n";
		return;
	}

	for (int i = 0; i < nums; i = i + 3)
	{
		verindex[i/3].x = i;
		verindex[i/3].y = i + 1;
		verindex[i/3].z = i + 2;
	}

	checkCudaErrors(cudaMemcpy(static_cast<void*>(positions.data()), gMemory->dVertex,
		sizeof(Vertex) * nums, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(static_cast<void*>(normals.data()), gMemory->dNormal,
		sizeof(Normal) * nums, cudaMemcpyDeviceToHost));

	std::cout << "Writing with ply format... " << std::endl;
	std::cout << "Number of vertices : " << nums << std::endl;
	std::cout << "Number of face: " << nums/3 << std::endl;

  frameIndex = frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.surfacePath) / 
  (mConfigParams.surfacePrefix + seqnum + ".ply")).generic_string();

	// std::string basename = "TestSurfaceR";
	// std::string path = params->mConfigParams.boundPath + std::string(basename) + ".ply";
  std::ofstream file(fileName.c_str());

	PlyFile file_ply;

  file_ply.add_properties_to_element("vertex", { "x", "y", "z" }, 
        Type::FLOAT32, positions.size(), reinterpret_cast<uint8_t*>(positions.data()), Type::INVALID, 0);

	file_ply.add_properties_to_element("face", { "vertex_indices" },
        Type::UINT32, verindex.size(), reinterpret_cast<uint8_t*>(verindex.data()), Type::UINT8, 3);

    file_ply.get_comments().push_back("generated by tinyply 2.3");

	file_ply.write(file, true);
}



void FileData::saveliuVtkfile(std::vector<FluidParticle>& _fParticle, int frame)
{
  std::vector<FluidParticle> fParticle = _fParticle;
  fParticle.resize(25);
  vtkSmartPointer<vtkPoints> FluidParticles;

  FluidParticles = vtkSmartPointer<vtkPoints>::New();
  FluidParticles->SetNumberOfPoints(fParticle.size());

  vtkSmartPointer<vtkFloatArray> velocity = vtkSmartPointer<vtkFloatArray>::New();
  velocity->SetName("Vel");
  velocity->SetNumberOfComponents(3);
  velocity->SetNumberOfTuples(fParticle.size());

  vtkSmartPointer<vtkFloatArray> rhop = vtkSmartPointer<vtkFloatArray>::New();
  rhop->SetName("Rhop");
  rhop->SetNumberOfComponents(1);
  rhop->SetNumberOfTuples(fParticle.size());

  for (int i = 0; i < fParticle.size(); i++)
	{
    auto pos =  fParticle[i].pos;
    auto vel =  fParticle[i].vel;

		FluidParticles->InsertPoint(static_cast<vtkIdType>(i), pos.x, pos.y, pos.z);
    velocity->InsertTuple3(static_cast<vtkIdType>(i), vel.x , vel.y, vel.z);

    rhop->InsertTuple1(static_cast<vtkIdType>(i), fParticle[i].rhop);
	}

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(FluidParticles);

  vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
  for (long i = 0; i < fParticle.size(); ++i){
    vtkIdType pt[] = {i};
    vertices->InsertNextCell(1, pt);
  }

  polydata->SetVerts(vertices);
  polydata->GetPointData()->AddArray(velocity);
  polydata->GetPointData()->AddArray(rhop);

  vtkSmartPointer<vtkPolyDataWriter> writer =
      vtkSmartPointer<vtkPolyDataWriter>::New();


  writer->SetFileTypeToASCII();
  writer->SetInputData(polydata);

  frameIndex = frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;

  mConfigParams.diffusePrefix = "PartFluid_";
  
  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.diffusePath) / 
  (mConfigParams.diffusePrefix + seqnum + ".vtk")).generic_string();

  writer->SetFileName(fileName.c_str());
  writer->Write();
}


void FileData::saveMotionVtkfile(std::vector<BoundParticle>& _bParticle, int frame)
{
  std::vector<BoundParticle> bParticle = _bParticle;

  vtkSmartPointer<vtkPoints> BoundParticles;

  BoundParticles = vtkSmartPointer<vtkPoints>::New();
  BoundParticles->SetNumberOfPoints(bParticle.size());

  vtkSmartPointer<vtkFloatArray> velocity = vtkSmartPointer<vtkFloatArray>::New();
  velocity->SetName("Vel");
  velocity->SetNumberOfComponents(3);
  velocity->SetNumberOfTuples(bParticle.size());

  for (int i = 0; i < bParticle.size(); i++)
	{
    auto pos =  bParticle[i].pos;
    auto vel =  bParticle[i].vel;

		BoundParticles->InsertPoint(static_cast<vtkIdType>(i), pos.x, pos.y, pos.z);
    velocity->InsertTuple3(static_cast<vtkIdType>(i), vel.x , vel.y, vel.z);

	}

  vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
  polydata->SetPoints(BoundParticles);

  vtkSmartPointer<vtkCellArray> vertices = vtkSmartPointer<vtkCellArray>::New();
  for (long i = 0; i < bParticle.size(); ++i){
    vtkIdType pt[] = {i};
    vertices->InsertNextCell(1, pt);
  }

  polydata->SetVerts(vertices);
  polydata->GetPointData()->AddArray(velocity);

  vtkSmartPointer<vtkPolyDataWriter> writer =
      vtkSmartPointer<vtkPolyDataWriter>::New();

  writer->SetFileTypeToASCII();
  writer->SetInputData(polydata);

  frameIndex = frameIndex;
  ConfigParams mConfigParams = params->mConfigParams;

  mConfigParams.diffusePrefix = "Motion_";

  std::string seqnum(mConfigParams.nzeros, '0');
  std::string formats = std::string("%.") + std::to_string(mConfigParams.nzeros) + std::string("d");
  std::sprintf(&seqnum[0], formats.c_str(), frameIndex);
  std::string fileName = (fs::path(mConfigParams.diffusePath) / 
  (mConfigParams.diffusePrefix + seqnum + ".vtk")).generic_string();

  writer->SetFileName(fileName.c_str());
  writer->Write();
}