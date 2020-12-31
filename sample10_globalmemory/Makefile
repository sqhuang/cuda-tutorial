CU_APPS=sample10_globalvariable \
        sample10_memtransfer \
        sample10_pinmemtransfer \
        sample10_sumarrayzerocpy \
        sample10_sumarrayzerocpyUVA \
        sample10_readsegment \
        sample10_writesegment \
        sample10_simplemathAoS \
        sample10_simplemathSoA \
        sample10_transpose \
        sample10_summatrixGPUManaged \
        sample10_summatrixGPUManual

C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_35 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
