CC =/bin/gcc
LD = gcc
CFLAGS = -O3 -ffast-math -march=native -Wall -ftree-vectorize -fopenmp -Werror -flto  -fopt-info-vec  # -g  -fopt-info-vec-missed 

LDFLAGS = -lm 
RM = /bin/rm -f
OBJS = memory_pool.o mm_funcs.o main.o matrix_ops_fallback.o matrix_ops.o matrix_ops_avx.o 
EXECUTABLE = strassen

all:$(EXECUTABLE)

$(EXECUTABLE): $(OBJS) 
	$(LD) $(CFLAGS) -o $(EXECUTABLE) $(OBJS) $(LDFLAGS)

memory_pool.o: memory_pool.h memory_pool.c
	$(CC) $(CFLAGS) -c memory_pool.c

mm_funcs.o: mm_funcs.h mm_funcs.c matrix_data.h
	$(CC) $(CFLAGS)  -c  mm_funcs.c 

# for AVX supprts (my local machines and vitsippa)
matrix_ops_avx.o: matrix_ops_avx.c matrix_data.h
	$(CC) $(CFLAGS)  -mavx2 -mfma -c matrix_ops_avx.c

# not AVX support, fallback (fries)
matrix_ops_fallback.o: matrix_ops_fallback.c matrix_data.h
	$(CC) $(CFLAGS) -c matrix_ops_fallback.c

matrix_ops.o: matrix_ops.h matrix_ops.c matrix_data.h
	$(CC) $(CFLAGS) -c matrix_ops.c

main.o: main.c matrix_data.h
	$(CC) $(CFLAGS) -c main.c

clean:
	-$(RM) -f $(OBJS) $(EXECUTABLE) *.out *.csv r0* -r
