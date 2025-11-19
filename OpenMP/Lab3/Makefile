ifndef EXERCISE
EXERCISE=jacobi.c
endif

CC=clang
LD=ld
OBJDUMP=objdump

OPT=-O3 -g
OMP=-fopenmp=libomp -fopenmp-targets=nvptx64-nvidia-cuda
CFLAGS=$(OPT) $(OMP) -I. $(EXT_CFLAGS)
LDFLAGS=-lm $(EXT_LDFLAGS)

SRCS=utils.c
OBJS=$(SRCS:.c=.o) $(EXERCISE:.c=.o)
EXE=$(EXERCISE:.c=.exe)

$(EXE): $(OBJS)
        $(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

all: $(EXE)

.PHONY: run profile clean
run: $(EXE)
        ./$(EXE) $(EXT_ARGS)

profile: $(EXE)
        sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/ext/lib:${LD_LIBRARY_PATH} LIBRARY_PATH=/usr/ext/lib:${LIBRARY_PATH} /usr/local/cuda/bin/nvprof ./$(EXE) $(EXT_ARGS)

clean:
        rm -f $(OBJS) *.o *.exe *.out *~