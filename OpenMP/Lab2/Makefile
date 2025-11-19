ifndef EXERCISE
EXERCISE=jacobi.c
endif

CC=gcc
LD=ld
OBJDUMP=objdump

OPT=-O3 -g -fopenmp

CFLAGS=$(OPT) -I. $(EXT_CFLAGS)
LDFLAGS=-lm $(EXT_LDFLAGS)

SRCS=utils.c
OBJS=$(SRCS:.c=.o) $(EXERCISE:.c=.o)
EXE=$(EXERCISE:.c=.exe)

$(EXE): $(OBJS)
        $(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

all: $(EXE)

.PHONY: run clean
run: $(EXE)
        ./$(EXE) $(EXT_ARGS)

profile: $(EXE)
        valgrind --tool=callgrind --dump-instr=yes --simulate-cache=yes --collect-jumps=yes ./$(EXE) $(EXT_ARGS)

clean:
        rm -f $(OBJS) *.o *.exe *.out *~