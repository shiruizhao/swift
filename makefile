MAIN=swift

CXX=g++
CXX_CFLAGS=-std=c++11 -Wall
LIB_FLAGS=-larmadillo
EXEC=$(MAIN)
SRC=src/absyn/*.cpp \
  src/analyzer/*.cpp \
  src/analyzer/ConjugatePriors/*.cpp \
  src/ir/*.cpp \
  src/fabrica/*.cpp \
  src/msg/*.cpp \
  src/predecl/*.cpp \
  src/code/*.cpp \
  src/codegen/*.cpp \
  src/semant/*.cpp \
  src/random/*.cpp \
  src/printer/*.cpp \
  src/preprocess/*.cpp \
  src/util/*.cpp \
  src/main.cpp \
  src/parse/parser.cpp \
  src/parse/lexer.cpp
YACCDIR=lib/byacc-20130925

help:
	@echo 'Makefile for swift compiler                                            '
	@echo '                                                                       '
	@echo 'Usage:                                                                 '
	@echo '   make compile                     compile the whole project          '
	@echo '   make genparser                   re-generate the parser             '
	@echo '   ./run-target.sh [model name]     swift compile and run the model    '
	@echo '                                                                       '

compile: $(SRC)
	$(CXX) $(CXX_CFLAGS) $(SRC) -o $(EXEC) -L/users/micas/szhao/no_backup/software/anaconda3/envs/gem5/lib -I/users/micas/szhao/no_backup/software/anaconda3/envs/gem5/include $(LIB_FLAGS)

genparser: $(YACCDIR)/yacc
	cd src/parse; flex -o lexer.cpp blog.flex; ../../$(YACCDIR)/yacc -v -d -o parser.cpp blog.yacc

$(YACCDIR)/yacc:
	cd $(YACCDIR); ./configure; make
