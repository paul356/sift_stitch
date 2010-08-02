###
###     Makefile for H.264/AVC encoder
###
###             generated for UNIX/LINUX environments
###             by H. Schwarz, Limin Wang
###



NAME=   sift-stitch

### include debug information: 1=yes, 0=no
DBG?= 1
### include MMX optimization : 1=yes, 0=no
M32?= 0

DEPEND= dependencies

BINDIR= .
INCDIR= .
SRCDIR= .
OBJDIR= .

ADDSRCDIR= ./
ADDINCDIR= ./

ifeq ($(M32),1)
CC=     $(shell which g++) -m32
else
CC=     $(shell which g++) 
endif

LIBS=   -lm $(shell pkg-config --libs opencv) $(shell gsl-config --libs)
AFLAGS=  
CFLAGS= $(shell pkg-config --cflags opencv) $(shell gsl-config --cflags)
FLAGS=  $(CFLAGS) -Wall -I$(INCDIR) -I$(ADDINCDIR)

ifeq ($(DBG),1)
SUFFIX=
FLAGS+= -g -O0
else
SUFFIX=
FLAGS+= -O3
endif

OBJSUF= .o$(SUFFIX)

SRC=    $(wildcard $(SRCDIR)/*.c) $(wildcard $(SRCDIR)/*.cpp)
ADDSRC= 
OBJ1 =  $(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o$(SUFFIX))
OBJ=    $(OBJ1:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o$(SUFFIX))
BIN=    $(BINDIR)/$(NAME)$(SUFFIX)

.PHONY: default distclean clean tags

default: messages $(DEPEND) $(BIN) 

messages:
ifeq ($(M32),1)
	@echo 'Compiling with M32 support...'
endif
ifeq ($(DBG),1)
	@echo 'Compiling with Debug support...'
endif

clean:
	@echo remove all objects
	@rm $(OBJDIR)/*.o

distclean: clean
	@rm -f $(DEPEND) tags
	@rm -f $(BIN)

tags:
	@echo update tag table
	@ctags inc/*.h src/*.c

$(BIN):    $(OBJ)
	@echo
	@echo 'creating binary "$(BIN)"'
	$(CC) $(FLAGS) -o $(BIN) $(OBJ) $(LIBS)
	@echo '... done'
	@echo

$(DEPEND): $(SRC)
	@echo
	@echo 'checking dependencies'
	$(SHELL) -ec '$(CC) $(AFLAGS) -MM $(FLAGS) -I$(INCDIR) -I$(ADDINCDIR) $(SRC) $(ADDSRC)                  \
         | sed '\''s@\(.*\)\.o[ :]@$(OBJDIR)/\1.o$(SUFFIX):@g'\''               \
         >$(DEPEND)'
	@echo

$(OBJDIR)/%.o$(SUFFIX): $(SRCDIR)/%.c
	@echo 'compiling object file "$@" ...'
	@$(CC) -c -o $@ $(FLAGS) $<

$(OBJDIR)/%.o$(SUFFIX): $(SRCDIR)/%.cpp
	@echo 'compiling object file "$@" ...'
	$(CC) -c -o $@ $(FLAGS) $<

$(OBJDIR)/%.o$(SUFFIX): $(ADDSRCDIR)/%.c
	@echo 'compiling object file "$@" ...'
	@$(CC) -c -o $@ $(FLAGS) $<

$(OBJDIR)/%.o$(SUFFIX): $(ADDSRCDIR)/%.cpp
	@echo 'compiling object file "$@" ...'
	@$(CC) -c -o $@ $(FLAGS) $<

-include $(DEPEND)

