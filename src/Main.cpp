#include <iostream>
#include "GPUPOST.h"

int main()
{    
    GPUPOST GPOST;

    GPOST.initialize();
    GPOST.runSimulation();
    GPOST.finalize();
}


