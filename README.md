# **VisFluid** 

<p align="center">
    <img src="./assets/david_pacthes.png" width="80%"><br>
</p>

## **About**
VisFluid is a program for narrow band surface reconstruction designed for particle-based fluid and runs on the NVIDIA GPU.  
This repository provides source code to implementation the algorithm presented in the paper [Enhanced narrow band surface reconstruction with anisotropic kernel].  

### **Dependencies**
The code can be compiled on Ubuntu (GCC 6.3.0) and Ubuntu 16.04.1 LTS providing that CUDA (>=9.0) is installed. To run the executable(s), an NVIDIA GPU should be installed on the machine.  
- [VTK8.2.0](https://vtk.org/)  
The VTK has been installed!  

### **Run example**
To run the code, just do:  
> bash run.sh  

The input particle vtk file is located in test/watercrown/particle. After the program has been executed over, the output generated triangle meshes ply file is located test/watercrown/surface. This ply file can be opened through Paraview or Blender. To reproduce the rendering results in the paper, the blender file can be used, which has configured the rendering environment. And the only thing need to do is give the liquid material to the generated ply file.
