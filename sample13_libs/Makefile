CU_APPS=cublas \
		cufft \
		cusparse \
		rand-kernel \
        replace-rand-streams \
		replace-rand

all: ${C_APPS} ${CU_APPS}

cublas: cublas.cu
	nvcc -O2 -arch=sm_35 -lcublas -o cublas cublas.cu
cufft: cufft.cu
	nvcc -O2 -arch=sm_35 -lcufft -o cufft cufft.cu
cusparse: cusparse.cu
	nvcc -O2 -arch=sm_35 -lcusparse -o cusparse cusparse.cu
rand-kernel: rand-kernel.cu
	nvcc -O2 -arch=sm_35 -lcurand -o rand-kernel rand-kernel.cu
replace-rand-streams: replace-rand-streams.cu
	nvcc -O2 -arch=sm_35 -lcurand -o replace-rand-streams replace-rand-streams.cu
replace-rand: replace-rand.cu
	nvcc -O2 -arch=sm_35 -lcurand -o replace-rand replace-rand.cu
%: %.cu
	nvcc -O2 -arch=sm_35 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}