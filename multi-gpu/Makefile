
NVCC = nvcc
NVFLAGS = -I../h -O5 -keep -Xcompiler -fopenmp -Xcompiler -march=native

M2090FLAGS=-arch compute_20 -code sm_20 -DGPU_M2090 -DLINE_32_GF2_8_BITSLICE
K40FLAGS=-arch compute_30 -code sm_35 -DGPU_K40 -DLINE_32_GF2_8_BITSLICE
K80FLAGS=-ccbin /cvmfs/fgi.csc.fi/fgci/centos7/apps/GCC/4.8.2/bin/g++ -arch compute_37 -code sm_37 -DGPU_K80 -DLINE_32_GF2_8_BITSLICE
P100FLAGS=-arch compute_60 -code sm_60 -DGPU_P100 -DLINE_32_GF2_8_BITSLICE

all: lister-m2090 lister-m2090.s \
	 lister-k40 lister-k40.s \
	 lister-k80 lister-k80.s \
	 lister-p100 lister-p100.s

lister-m2090: lister.cu ../h/gf.h ../h/ffprng.h
	$(NVCC) $(NVFLAGS) $(M2090FLAGS) -o lister-m2090 lister.cu

lister-k40: lister.cu ../h/gf.h ../h/ffprng.h
	$(NVCC) $(NVFLAGS) $(K40FLAGS) -o lister-k40 lister.cu

lister-k80: lister.cu ../h/gf.h ../h/ffprng.h
	$(NVCC) $(NVFLAGS) $(K80FLAGS) -o lister-k80 lister.cu

lister-p100: lister.cu ../h/gf.h ../h/ffprng.h
	$(NVCC) $(NVFLAGS) $(P100FLAGS) -o lister-p100 lister.cu

lister-m2090.s: lister-m2090
	cuobjdump -sass lister-m2090 > lister-m2090.s

lister-k40.s: lister-k40
	cuobjdump -sass lister-k40 > lister-k40.s

lister-k80.s: lister-k80
	cuobjdump -sass lister-k80 > lister-k80.s

lister-p100.s: lister-p100
	cuobjdump -sass lister-p100 > lister-p100.s

clean:
	rm -f *.o *.a *.s *~ lister-m2090 lister-k40 lister-k80 lister-p100
	rm -f *.cudafe* *.fatbin* *.cubin* *.ptx *.reg* *.i *.ii *.module_id *.hash
