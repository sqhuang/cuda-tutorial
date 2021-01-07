CU_APPS=asyncAPI \
        simpleHyperqDepth \
		simpleHyperqBreadth \
        simpleHyperqOpenmp \
		simpleHyperqDependence \
        simpleCallback 

C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_35 -Xcompiler -fopenmp -o $@ $< -lgomp
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
