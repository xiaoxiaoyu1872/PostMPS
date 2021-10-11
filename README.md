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
- 
All the dependencies have been installed. To run the code:
```
> cd PostMPS
> bash run.sh
