ifndef EXERCISE
EXERCISE=exercise7.c
endif

CC=gcc
LD=ld
OBJDUMP=objdump

OPT=-O2 -g -fopenmp
CFLAGS=$(OPT) -I. $(EXT_CFLAGS)
LDFLAGS=-lm $(EXT_LDFLAGS)

SRCS=main.c utils.c
OBJS=$(SRCS:.c=.o) $(EXERCISE:.c=.o)
EXE=$(EXERCISE:.c=.exe)

$(EXE): $(OBJS)
        $(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

all: $(EXE)

.PHONY: run clean
run: $(EXE)
        ./$(EXE)

clean:
        rm -f $(OBJS) *.o *.exe *.out *~