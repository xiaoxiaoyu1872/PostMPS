# **VisFluid** 

<p align="center">
    <img src="./assets/david_pacthes.png" width="80%"><br>
</p>

## **About**
VisFluid is a program for enhanced narrow band surface reconstruction designed for particle-based fluid and runs on the NVIDIA GPU. This program bring high-quality visual effects to scientific and engineering simulations, helping people to illustrate their simulations. This repository provides source code to implement the algorithm presented in the paper [Enhanced narrow band surface reconstruction with anisotropic kernel].  

### **Dependencies**
The code can be compiled on Ubuntu (GCC 6.3.0) providing that CUDA (>=9.0) is installed. To run the executable(s), an NVIDIA GPU should be installed on the machine.  
The dependent vtk8.2 has been installed in this repository.

### **Run example**
To run the code, just do:  
> bash run.sh  

The input particle vtk file is located in test/watercrown/particle. Once the program has been executed over, the output triangle meshes ply file will be generated in test/watercrown/surface. This ply file can be opened through Paraview (https://www.paraview.org/) or Blender (https://www.blender.org/). To reproduce the representative figure in the paper, the given blender file can be used, which has configured the rendering environment and is located in test/watercrown. The only thing need to do is giving the liquid material to the generated ply file.  
The simulation results can be obtained from partcile-based fluid solver, such as DualSPH (https://dual.sphysics.org/) and GPUSPH (http://www.gpusph.org/)
If you have any question about the code, contact me 11824048@zju.edu.cn.

