OS := $(shell uname)
CFLAGS:= -O0 -static -D_GNU_SOURCE

ifeq ($(OS),Darwin)
	CC:= clang
	UTIL_OBJ := util/darwin-util.o
	ASM_CMD = otool -Vx
	MACH ?= 1
	KEXT ?= 0
else
	CC:= aarch64-linux-gnu-gcc
	# CC:= ~/ellcc/bin/ecc --target=arm64v8-linux
	UTIL_OBJ := util/linux-util.o
	LIBS:= -lpthread -lrt
	ASM_CMD = $(CC) -S -c $(CFLAGS)
	MACH ?= 1
endif

N2MB ?= 0
ifneq ($(N2MB),0)
	CFLAGS += -DN2MB
endif

PAC ?= 0
ifneq ($(PAC),0)
	CFLAGS += -arch arm64e
	LIBS += -arch arm64e
endif

FORCEADDR ?= 0
ifneq ($(FORCEADDR),0)
	CFLAGS += -DFORCE_ADDR
endif

ifneq ($(KEXT),0)
	CFLAGS += -DKEXT
else ifneq ($(MACH),0)
	CFLAGS += -DMACH
endif

_OBJ ?= augury imp slh-poc aslr-poc
OUTDIR=bin
OBJDIR=obj
N_OBJ = $(_OBJ:%=$(OUTDIR)/%)
D_OBJ = $(_OBJ:%=$(OUTDIR)/%-debug)

NA_OBJ = $(_OBJ:%=$(OUTDIR)/%.S)
DA_OBJ = $(_OBJ:%=$(OUTDIR)/%-debug.S)

all: debug native

$(OBJDIR)/slh-poc.o: CFLAGS += -mspeculative-load-hardening
$(OBJDIR)/slh-poc-debug.o: CFLAGS += -mspeculative-load-hardening
# $(OBJDIR)/slh-poc: CFLAGS += -mspeculative-load-hardening
# $(OBJDIR)/slh-poc-debug: CFLAGS += -mspeculative-load-hardening

$(OBJDIR)/%.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

$(OBJDIR)/%-debug.o: %.c
	$(CC) -c $(CFLAGS) -o $@ $<

$(OUTDIR)/%: $(OBJDIR)/%.o $(UTIL_OBJ)
	$(CC) -o $@ $^ $(LIBS)

$(OUTDIR)/%-debug: $(OBJDIR)/%-debug.o $(UTIL_OBJ)
	$(CC) -o $@ $^ $(LIBS)

$(OUTDIR)/%.S: $(OUTDIR)/%
ifeq ($(OS),Darwin)
	$(ASM_CMD) $^ > $@
else
	# $@
	$(ASM_CMD) $(notdir $(patsubst %.S,%.c,$(patsubst %-debug.S,%.S,$@))) > $@
endif

# $(OUTDIR)/%-debug.S: $(OUTDIR)/%-debug
# 	$(ASM_CMD) $(@:-debug=nope) > $@
# 	# $(ASM_CMD) $(notdir $(@:-debug.S=.c)) > $@

.PHONY: clean

debug: CFLAGS += -DNATIVE -DDEBUGG
debug: |$(OBJDIR)
debug: |$(OUTDIR) $(D_OBJ) $(DA_OBJ)

native: CFLAGS += -DNATIVE
native: |$(OBJDIR)
native: |$(OUTDIR) $(N_OBJ) $(NA_OBJ)

asm-native: CFLAGS += -DNATIVE
asm-native: |$(NA_OBJ)

asm-debug: CFLAGS += -DNATIVE -DDEBUGG
asm-debug: |$(DA_OBJ)

asm: asm-native asm-debug

$(OUTDIR):
	mkdir -p $(OUTDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf bin obj
	rm -rf util/*.o
