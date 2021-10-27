# **VisFluid** 

<p align="center">
    <img src="./paper/representative.png" width="40%"><br>
</p>

## **About**
VisFluid is a program used to reconstruct fluid surface from particle-based fluid solver and runs fully on NVIDIA GPU. This program brings high-quality visual effects to scientific and engineering computing, helping people to illustrate their simulation results. This repository provides source code to implement the algorithm presented in the paper [Enhanced narrow band surface reconstruction with anisotropic kernel].  

### **Dependencies**
The code can be compiled on Ubuntu (GCC 6.3.0) with CUDA (>=9.0). To run the code, an NVIDIA GPU should be installed on the machine.    
The test file is put on Google Drive as follows:
test.zip(https://drive.google.com/file/d/1HPbJa-htn2_YTEcl5P-dp-oTO-YOlhK2/view?usp=sharing)  

### **Run test**
To run the code, just do:  
> conda env create -f conda.yml  
> conda activate postmps  
> mkdir build  
> make && make post  


The input particle vtk file is located in test/watercrown/particle. Once the program has been executed over, the output triangle meshes ply file will be generated in test/watercrown/surface. This ply file can be opened through Paraview (https://www.paraview.org/) or Blender (https://www.blender.org/). To reproduce the representative figure, the blender file has given, which has already configured the rendering environment and is located in test/watercrown. The only thing need to do is giving the liquid material to the generated ply file.  
Other simulation results can be obtained from partcile-based fluid solver, such as DualSPH (https://dual.sphysics.org/) and GPUSPH (http://www.gpusph.org/).  
If you have any question, contact me 11824048@zju.edu.cn.

