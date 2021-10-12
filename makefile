compile:
	cd  ./build && cmake  .. && make
post:
	cd  ./bin && ./VisGPU
clean:
	cd  ./build && make clean