cd lib
unzip VTK-8.2.0.zip
cd VTK-8.2.0
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../../VTK .. 
make -j4
make install
cd ../../
rm -r VTK-8.2.0
cd ..

unzip lib.zip
unzip test.zip
source env.sh
mkdir build
make
make post