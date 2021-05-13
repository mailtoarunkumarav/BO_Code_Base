# Makefile for SVMHeavy v6

# Available targets:
#
# basic: no optimisation, no debugging (default)
# opt: optimised version
# debug: basic debugging (-g)
# debugmore: basic debugging + optimiser debugging
# debugdeep: basic debugging + deep optimiser debugging
# clean: remove .o files
#
# - Need to uncomment component relevant to target system.
# - New files need to be included in SRC, the DEP tree and a recipe
#
# MLMUTATEFILE: touched to indicate changes in ML tree as make
#     churns and dies on the dependency tree otherwise.
# CBASEFLAGS: over-ride target specifics in basefn.h
# DEBUG_MEM: adds in some extra checks on memory new/delete operations



# absolute bare-bones version - threads, sockets and keyboard interupts disabled
# (for djgpp bare-bones use gxx and del)

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DDISABLE_KB_BY_DEF
#LIBFLAGS = -lm

# dos/djgpp version - no sockets or threads but has keyboard interupts

#CC = gxx
#CCC = gxx
#RM = del
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DDJGPP_MATHS -DHAVE_CONIO
#LIBFLAGS = -lm












# cygwin/gcc version modern - threadless and socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
##CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11
#CBASEFLAGS = -DCYGWIN10 -DSPECIFYSYSVIAMAKE -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11 -Wno-strict-overflow
#LIBFLAGS = -lm

# cygwin/gcc version modern

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
##CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DENABLE_THREADS -DCYGWIN_BUILD -DHAVE_TERMIOS -DHAVE_IOCTL -DIS_CPP11 -std=c++11 
#CBASEFLAGS = -DCYGWIN10 -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DENABLE_THREADS -DCYGWIN_BUILD -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11 -Wno-strict-overflow
#LIBFLAGS = -lm

# cygwin/gcc version modern - socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
##CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DENABLE_THREADS -DCYGWIN_BUILD -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11
#CBASEFLAGS = -DCYGWIN10 -DSPECIFYSYSVIAMAKE -DENABLE_THREADS -DCYGWIN_BUILD -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11 -Wno-strict-overflow
#LIBFLAGS = -lm

# cygwin/gcc version modern - threadless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
##CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11
#CBASEFLAGS = -DCYGWIN10 -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DHAVE_TERMIOS -DIS_CPP11 -DHAVE_IOCTL -std=c++11 -Wno-strict-overflow
#LIBFLAGS = -lm














# linux/gcc - socketless and socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DHAVE_TERMIOS  -DHAVE_IOCTL -DNOPURGE
#LIBFLAGS = -lm

# linux/gcc version

CC = g++
CCC = g++
RM = rm -f
MLMUTATEFILE = POSTSIG.TXT
CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DENABLE_THREADS -DHAVE_TERMIOS -DHAVE_IOCTL -DNOPURGE
LIBFLAGS = -pthread -lm

# linux/gcc - threadless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DALLOW_SOCKETS -DHAVE_TERMIOS -DHAVE_IOCTL -DNOPURGE
#LIBFLAGS = -lm

# linux/gcc - socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DENABLE_THREADS -DHAVE_TERMIOS -DHAVE_IOCTL -DNOPURGE
#LIBFLAGS = -pthread -lm














# visual studio version - socketless and threadless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DHAVE_CONIO
#LIBFLAGS = -lm

# visual studio version

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DHAVE_CONIO -DENABLE_THREADS -DALLOW_SOCKETS
#LIBFLAGS = -lm

# visual studio version - socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DENABLE_THREADS -DHAVE_CONIO
#LIBFLAGS = -lm

# visual studio version - threadless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DHAVE_CONIO -DALLOW_SOCKETS
#LIBFLAGS = -lm

# visual studio (older) version

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DVISUAL_STU_OLD -DVISUAL_STU_NOERF -DHAVE_CONIO -DENABLE_THREADS -DALLOW_SOCKETS
#LIBFLAGS = -lm

# visual studio (older) version - socketless

#CC = g++
#CCC = g++
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = -DSPECIFYSYSVIAMAKE -DVISUAL_STU -DVISUAL_STU_OLD -DVISUAL_STU_NOERF -DHAVE_CONIO -DENABLE_THREADS 
#LIBFLAGS = -lm















# mex creator script (make clean then make will simply create a text file mexmake.m that you can cut and paste into Matlab to make mex version)

#CC = ./mexmakeobjcmd.sh
#CCC = ./mexmakemexcmd.sh
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = USE_MEX VISUAL_STU VISUAL_STU_NOERF HAVE_CONIO IGNOREMEM ALLOW_SOCKETS ENABLE_THREADS IS_CPP11
#LIBFLAGS = -lm

# mex creator script (make clean then make will simply create a text file mexmake.m that you can cut and paste into Matlab to make mex version) - threadless

#CC = ./mexmakeobjcmd.sh
#CCC = ./mexmakemexcmd.sh
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = USE_MEX VISUAL_STU VISUAL_STU_NOERF HAVE_CONIO IGNOREMEM ALLOW_SOCKETS IS_CPP11
#LIBFLAGS = -lm

# mex creator script (make clean then make will simply create a text file mexmake.m that you can cut and paste into Matlab to make mex version) - socketless

#CC = ./mexmakeobjcmd.sh
#CCC = ./mexmakemexcmd.sh
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = USE_MEX VISUAL_STU VISUAL_STU_NOERF HAVE_CONIO IGNOREMEM ENABLE_THREADS IS_CPP11
#LIBFLAGS = -lm

# mex creator script (make clean then make will simply create a text file mexmake.m that you can cut and paste into Matlab to make mex version) - threadless and socketless

#CC = ./mexmakeobjcmd.sh
#CCC = ./mexmakemexcmd.sh
#RM = rm -f
#MLMUTATEFILE = POSTSIG.TXT
#CBASEFLAGS = USE_MEX VISUAL_STU VISUAL_STU_NOERF HAVE_CONIO IGNOREMEM IS_CPP11
#LIBFLAGS = -lm










# source files

SRC = basefn.cc gslrefs.cc zerocross.cc \
      anion.cc numbase.cc dynarray.cc strfns.cc opttest.cc paretotest.cc idstore.cc smatrix.cc gentype.cc mercer.cc mlcommon.cc kcache.cc FNVector.cc \
      hyper_psc.cc hyper_debug.cc hyper_base.cc hyper_opt.cc hyper_alt.cc \
      optcontext.cc optlinbasecontext.cc optlincontext.cc optlinstate.cc \
      sQbase.cc sQsLsAsWs.cc sQsmo.cc sQd2c.cc sQgraddesc.cc linsolve.cc \
      direct_direct.cc neldermead.cc \
      ml_base.cc \
      svm_generic.cc \
      svm_scalar.cc svm_binary.cc svm_single.cc svm_biscor.cc svm_scscor.cc svm_densit.cc svm_pfront.cc \
      svm_vector_atonce_template.cc \
      svm_multic_redbin.cc svm_multic_atonce.cc svm_vector_redbin.cc svm_vector_atonce.cc svm_vector_mredbin.cc svm_vector_matonce.cc \
      svm_multic.cc svm_vector.cc svm_anions.cc svm_autoen.cc svm_gentyp.cc svm_planar.cc svm_mvrank.cc svm_mulbin.cc \
      svm_simlrn.cc svm_cyclic.cc \
      onn_generic.cc \
      onn_scalar.cc onn_binary.cc onn_vector.cc onn_anions.cc onn_autoen.cc onn_gentyp.cc \
      knn_generic.cc \
      knn_scalar.cc knn_binary.cc knn_multic.cc knn_vector.cc knn_anions.cc knn_autoen.cc knn_gentyp.cc knn_densit.cc \
      blk_generic.cc \
      blk_nopnop.cc blk_consen.cc blk_avesca.cc blk_avevec.cc blk_aveani.cc blk_usrfna.cc blk_usrfnb.cc blk_userio.cc blk_calbak.cc \
      blk_mexfna.cc blk_mexfnb.cc blk_mercer.cc blk_conect.cc blk_system.cc blk_kernel.cc blk_bernst.cc blk_batter.cc \
      imp_generic.cc \
      imp_expect.cc imp_parsvm.cc \
      lsv_generic.cc \
      lsv_scalar.cc lsv_vector.cc lsv_anions.cc lsv_scscor.cc lsv_autoen.cc lsv_gentyp.cc lsv_planar.cc lsv_mvrank.cc \
      gpr_generic.cc \
      gpr_scalar.cc gpr_vector.cc gpr_anions.cc gpr_gentyp.cc gpr_binary.cc \
      ssv_generic.cc \
      ssv_scalar.cc ssv_binary.cc ssv_single.cc \
      mlm_generic.cc \
      mlm_scalar.cc mlm_binary.cc mlm_vector.cc \
      ml_mutable.cc \
      fuzzyml.cc xferml.cc errortest.cc addData.cc analyseAnomaly.cc balc.cc hillclimb.cc \
      globalopt.cc gridopt.cc directopt.cc nelderopt.cc smboopt.cc bayesopt.cc \
      awarestream.cc \
      mlinter.cc
OBJS = $(SRC:.cc=.o)








# Flags and stuff

BASEFLAGS = -W -Wall
LIBFLAG = -lm
OPTFLAGS = -O3 -DIGNOREMEM -DNDEBUG
DEBUGFLAGSA = -g -DDEBUG_MEM_CHEAP
DEBUGFLAGSB = -DDEBUGOPT
DEBUGFLAGSC = -DDEBUGDEEP
TARGNAME = svmheavyv7.exe
TARGSRC = svmheavyv7.cc













basic: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS)
basic: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

opt: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(OPTFLAGS)
opt: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debug: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSA)
debug: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugalt: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSB)
debugalt: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugmore: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSA) $(DEBUGFLAGSB)
debugmore: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugopt: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSA) $(DEBUGFLAGSB)
debugopt: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugoptdeep: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSB) $(DEBUGFLAGSC)
debugoptdeep: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

debugdeep: CFLAGS = $(BASEFLAGS) $(CBASEFLAGS) $(DEBUGFLAGSA) $(DEBUGFLAGSB) $(DEBUGFLAGSC)
debugdeep: $(OBJS)
	$(CCC) $(TARGSRC) -o $(TARGNAME) $(OBJS) $(CFLAGS) $(LIBFLAGS)

.PHONY: clean

clean: 
	$(RM) *.o

mex: CFLAGS = 
mex: $(OBJS)
	$(CCC) svmmatlab.cpp *.obj




# Define macros to simplify include trees for header files only.  Each
# macro specifies the dependencies in the relevant header in terms of
# both headers included and previous macros for headers included from
# said header.  Headers with no dependencies are not defined here

DYNARRAYDEP     = basefn.h
VECTORDEP       = numbase.h dynarray.h $(DYNARRAYDEP)
SPARSEVECTORDEP = strfns.h vector.h $(VECTORDEP) 
MATRIXDEP       = sparsevector.h $(SPARSEVECTORDEP)
VECSTACKDEP     = vector.h $(VECTORDEP)
NONZEROINTDEP   = basefn.h
SETDEP          = matrix.h $(MATRIXDEP)
DGRAPHDEP       = matrix.h $(MATRIXDEP)


NUMBASEDEP    = basefn.h gslrefs.h
ANIONDEP      = basefn.h numbase.h $(NUMBASEDEP)
PARETOTESTDEP = vector.h $(VECTORDEP)
OPTTESTDEP    = vector.h $(VECTORDEP) $(MATRIXDEP)
IDSTOREDEP    = nonzeroint.h $(NONZEROINTDEP) sparsevector.h $(SPARSEVECTORDEP)
SMATRIXDEP    = matrix.h $(MATRIXDEP)
GENTYPEDEP    = basefn.h numbase.h $(NUMBASEDEP) anion.h $(ANIONDEP) \
                vector.h $(VECTORDEP) sparsevector.h $(SPARSEVECTORDEP) \
                matrix.h $(MATRIXDEP) set.h $(SETDEP) dgraph.h $(DGRAPHDEP) \
                opttest.h $(OPTTESTDEP) paretotest.h $(PARETOTESTDEP)
MERCERDEP     = vector.h $(VECTORDEP) sparsevector.h $(SPARSEVECTORDEP) \
                matrix.h $(MATRIXDEP) gentype.h $(GENTYPEDEP) numbase.h \
                awarestream.h $(AWARESTREAMDEP)
MLCOMMONDEP   = sparsevector.h $(SPARSEVECTORDEP) anion.h $(ANIONDEP) \
                mercer.h $(MERCERDEP)
KCACHEDEP     = vector.h $(VECTORDEP) gentype.h $(GENTYPEDEP)
FNVECTORDEP   = vector.h $(VECTORDEP) sparsevector.h $(SPARSEVECTORDEP) \
                gentype.h $(GENTYPEDEP) mercer.h $(MERCERDEP) \
                mlcommon.h $(MLCOMMONDEP)

CHOLDEP = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) mlcommon.h $(MLCOMMONDEP)

OPTCONTEXTDEP        = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) \
                       chol.h $(CHOLDEP) numbase.h $(NUMBASEDEP)
OPTLINBASECONTEXTDEP = optcontext.h $(OPTCONTEXTDEP)
OPTLINCONTEXTDEP     = optlinbasecontext.h $(OPTLINBASECONTEXTDEP)
OPTLINSTATEDEP       = optlincontext.h $(OPTLINCONTEXTDEP) mlcommon.h $(MLCOMMONDEP)
OPTSTATEDEP          = optcontext.h $(OPTCONTEXTDEP) mlcommon.h $(MLCOMMONDEP)

SQBASEDEP     = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) optstate.h $(OPTSTATEDEP) \
                smatrix.h $(SMATRIXDEP)
SQSLSQSWSDEP  = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) optstate.h $(OPTSTATEDEP) \
                sQbase.h $(SQBASEDEP)
SQSMODEP      = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) optstate.h $(OPTSTATEDEP) \
                sQbase.h $(SQBASEDEP)
SQD2CDEP      = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) optstate.h $(OPTSTATEDEP) \
                sQbase.h $(SQBASEDEP)
SQSMOVECTDEP  = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) optstate.h $(OPTSTATEDEP) \
                smatrix.h $(SMATRIXDEP) zerocross.h sQbase.h $(SQBASEDEP)
SQGRADDESCDEP = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) optstate.h $(OPTSTATEDEP) \
                sQbase.h $(SQBASEDEP) 
LINSOLVEDEP   = vector.h $(VECTORDEP) matrix.h $(MATRIXDEP) optstate.h $(OPTSTATEDEP) \
                sQbase.h $(SQBASEDEP)

DIRECT_DIRECTDEP = basefn.h
NELDERMEADDEP    = basefn.h

ML_BASEDEP = vector.h $(VECTORDEP) sparsevector.h $(SPARSEVECTORDEP) \
             matrix.h $(MATRIXDEP) mercer.h $(MERCERDEP) \
             set.h $(SETDEP) gentype.h $(GENTYPEDEP) \
             mlcommon.h $(MLCOMMONDEP) numbase.h $(NUMBASEDEP) \
             basefn.h

SVM_GENERICDEP = ml_base.h $(ML_BASEDEP)

SVM_SCALARDEP = svm_generic.h $(SVM_GENERICDEP) kcache.h $(KCACHEDEP) \
                optstate.h $(OPTSTATEDEP) sQgraddesc.h $(SQGRADDESCDEP)
SVM_BINARYDEP = svm_scalar.h $(SVM_SCALARDEP)
SVM_SIMLRNDEP = svm_scalar.h $(SVM_SCALARDEP)
SVM_SINGLEDEP = svm_binary.h $(SVM_BINARYDEP)
SVM_BISCORDEF = svm_binary.h $(SVM_BINARYDEP)
SVM_SCSCORDEF = svm_scalar.h $(SVM_SCALARDEP)
SVM_DENSITDEF = svm_scalar.h $(SVM_SCALARDEP)
SVM_PFRONTDEP = svm_binary.h $(SVM_BINARYDEP)

SVM_VECTOR_ATONCE_TEMPLATEDEP = svm_generic.h $(SVM_GENERICDEP) kcache.h $(KCACHEDEP) \
                                optstate.h $(OPTSTATEDEP) sQsLsAsWs.h $(SQSLSASWSDEP) \
                                sQsmo.h $(SQSMODEP) sQd2c.h $(SQD2CDEP) \
                                sQsmoVect.h $(SQSMOVECTDEP)

SVM_MULTIC_REDBINDEP  = svm_generic.h $(SVM_GENERICDEP) svm_binary.h $(SVM_BINARYDEP) \
                        svm_single.h $(SVM_SINGLEDEP) kcache.h $(KCACHEDEP) \
                        idstore.h $(IDSTOREDEP)
SVM_MULTIC_ATONCEDEP  = svm_generic.h $(SVM_GENERICDEP) svm_binary.h $(SVM_BINARYDEP) \
                        svm_single.h $(SVM_SINGLEDEP) kcache.h $(KCACHEDEP) \
                        idstore.h $(IDSTOREDEP)
SVM_VECTOR_REDBINDEP  = svm_generic.h $(SVM_GENERICDEP) svm_scalar.h $(SVM_SCALARDEP) \
                        svm_scscor.h $(SVM_SCSCORDEP) kcache.h $(KCACHEDEP)
SVM_VECTOR_ATONCEDEP  = svm_vector_atonce_template.h $(SVM_VECTOR_ATONCE_TEMPLATEDEP)
SVM_VECTOR_MREDBINDEP = svm_generic.h $(SVM_GENERICDEP) svm_scalar.h $(SVM_SCALARDEP)
SVM_VECTOR_MATONCEDEP = svm_vector_atonce_template.h $(SVM_VECTOR_ATONCE_TEMPLATEDEP)

SVM_MULTICDEP = svm_generic.h $(SVM_GENERICDEP) \
                svm_multic_redbin.h $(SVM_MULTIC_REDBINDEP) svm_multic_atonce.h $(SVM_MULTIC_ATONCEDEP)
SVM_VECTORDEP = svm_generic.h $(SVM_GENERICDEP) svm_scalar.h $(SVM_SCALARDEP) \
                svm_vector_redbin.h $(SVM_MULTIC_REDBINDEP) svm_vector_atonce.h $(SVM_MULTIC_ATONCEDEP) \
                svm_vector_mredbin.h $(SVM_MULTIC_MREDBINDEP) svm_vector_matonce.h $(SVM_MULTIC_MATONCEDEP)
SVM_ANIONSDEP = svm_vector.h $(SVM_VECTORDEP)
SVM_AUTOENDEP = svm_vector.h $(SVM_VECTORDEP)
SVM_GENTYPDEP = svm_vector.h $(SVM_VECTORDEP)
SVM_PLANARDEP = svm_planar.h $(SVM_SCALARDEP)
SVM_MVRANKDEP = svm_mvrank.h $(SVM_PLANARDEP)
SVM_MULBINDEP = svm_mulbin.h $(SVM_MVRANKDEP)
SVM_CYCLICDEP = svm_cyclic.h $(SVM_PLANARDEP)

ONN_GENERICDEP = ml_base.h $(ML_BASEDEP)

ONN_SCALARDEP = onn_generic.h $(ONN_GENERICDEP)
ONN_BINARYDEP = onn_generic.h $(ONN_GENERICDEP)
ONN_VECTORDEP = onn_generic.h $(ONN_GENERICDEP)
ONN_ANIONSDEP = onn_generic.h $(ONN_GENERICDEP)
ONN_AUTOENDEP = onn_vector.h  $(ONN_VECTORDEP)
ONN_GENTYPDEP = onn_generic.h $(ONN_GENERICDEP)

KNN_GENERICDEP = ml_base.h $(ML_BASEDEP) kcache.h $(KCACHEDEP)

KNN_SCALARDEP = knn_generic.h $(KNN_GENERICDEP)
KNN_BINARYDEP = knn_generic.h $(KNN_GENERICDEP)
KNN_MULTICDEP = knn_generic.h $(KNN_GENERICDEP)
KNN_VECTORDEP = knn_generic.h $(KNN_GENERICDEP)
KNN_ANIONSDEP = knn_generic.h $(KNN_GENERICDEP)
KNN_AUTOENDEP = knn_vector.h  $(KNN_VECTORDEP)
KNN_GENTYPDEP = knn_generic.h $(KNN_GENERICDEP)
KNN_DENSITDEP = knn_generic.h $(KNN_GENERICDEP)

BLK_GENERICDEP = ml_base.h $(ML_BASEDEP)

BLK_NOPNOPDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_MERCERDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_CONSENDEP = blk_generic.h $(BLK_GENERICDEP) idstore.h $(IDSTOREDEP)
BLK_BERNSTDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_BATTERDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_AVESCADEP = blk_generic.h $(BLK_GENERICDEP)
BLK_AVEVECDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_AVEANIDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_USRFNADEP = blk_generic.h $(BLK_GENERICDEP)
BLK_USRFNBDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_USERIODEP = blk_generic.h $(BLK_GENERICDEP)
BLK_CALBAKDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_MEXFNADEP = blk_generic.h $(BLK_GENERICDEP)
BLK_MEXFNBDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_CONECTDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_SYSTEMDEP = blk_generic.h $(BLK_GENERICDEP)
BLK_KERNELDEP = blk_generic.h $(BLK_GENERICDEP)

IMP_GENERICDEP = ml_base.h $(ML_BASEDEP)

IMP_EXPECTDEP = imp_generic.h $(IMP_GENERICDEP) hyper_opt.h
IMP_PARSVMDEP = imp_generic.h $(IMP_GENERICDEP) svm_pfront.h $(SVM_PFRONTDEP)

LSV_GENERICDEP = ml_base.h $(ML_BASEDEP) svm_scalar.h $(SVM_SCALARDEP)

LSV_SCALARDEP = lsv_generic.h $(LSV_GENERICDEP)
LSV_VECTORDEP = lsv_generic.h $(LSV_GENERICDEP)
LSV_ANIONSDEP = lsv_generic.h $(LSV_GENERICDEP)
LSV_SCSCORDEP = lsv_generic.h $(LSV_GENERICDEP) svm_scscor.h $(SVM_SCSCORDEP)
LSV_AUTOENDEP = lsv_vector.h $(LSV_VECTORDEP)
LSV_GENTYPDEP = lsv_generic.h $(LSV_GENERICDEP) svm_gentyp.h $(SVM_GENTYPDEP)
LSV_PLANARDEP = lsv_generic.h $(LSV_GENERICDEP) svm_planar.h $(SVM_PLANARDEP)
LSV_MVRANKDEP = lsv_generic.h $(LSV_GENERICDEP) svm_mvrank.h $(SVM_MVRANKDEP)

GPR_GENERICDEP = ml_base.h $(ML_BASEDEP) lsv_generic.h $(LSV_GENERICDEP)

GPR_SCALARDEP = gpr_generic.h $(GPR_GENERICDEP) lsv_scalar.h $(LSV_SCALARDEP)
GPR_VECTORDEP = gpr_generic.h $(GPR_GENERICDEP) lsv_vector.h $(LSV_VECTORDEP)
GPR_ANIONSDEP = gpr_generic.h $(GPR_GENERICDEP) lsv_anions.h $(LSV_ANIONSDEP)
GPR_GENTYPDEP = gpr_generic.h $(GPR_GENERICDEP) lsv_gentyp.h $(LSV_GENTYPDEP)
GPR_BINARYDEP = gpr_scalar.h $(GPR_SCALARDEP)

SSV_GENERICDEP = ml_base.h $(ML_BASEDEP) svm_scalar.h $(SVM_SCALARDEP)

SSV_SCALARDEP = ssv_generic.h $(SSV_GENERICDEP)
SSV_BINARYDEP = ssv_generic.h $(SSV_GENERICDEP) ssv_scalar.h $(SSV_SCALARDEP)
SSV_SINGLEDEP = ssv_generic.h $(SSV_GENERICDEP) ssv_binary.h $(SSV_BINAYRDEP)

MLM_GENERICDEP = ml_base.h $(ML_BASEDEP) svm_generic.h $(SVM_GENERICDEP) svm_scalar.h $(SVM_SCALARDEP)

MLM_SCALARDEP = mlm_generic.h $(MLM_GENERICDEP) svm_scalar.h $(SVM_SCALARDEP)
MLM_BINARYDEP = mlm_generic.h $(MLM_GENERICDEP) svm_binary.h $(SVM_BINARYDEP)
MLM_VECTORDEP = mlm_generic.h $(MLM_GENERICDEP) svm_vector.h $(SVM_VECTORDEP)

# Simplified version so as to not kill make!

ML_MUTABLEDEP = $(MLMUTATEFILE)
#ML_MUTABLEDEP = ml_base.h $(ML_BASEDEP) \
#                svm_generic.h $(SVM_GENERICDEP) onn_generic.h $(ONN_GENERICDEP) \
#                knn_generic.h $(KNN_GENERICDEP) gpr_generic.h $(GPR_GENERICDEP) \
#                lsv_generic.h $(LSV_GENERICDEP) blk_generic.h $(BLK_GENERICDEP) \
#                imp_generic.h $(IMP_GENERICDEP) ssv_generic.h $(SSV_GENERICDEP) \
#                mlm_generic.h $(MLM_GENERICDEP) \
#                svm_single.h $(SVM_SINGLEDEP) svm_binary.h $(SVM_BINARYDEP) \
#                svm_scalar.h $(SVM_SCALARDEP) svm_multic.h $(SVM_MULTICDEP) \
#                svm_vector.h $(SVM_VECTORDEP) svm_anions.h $(SVM_ANIONSDEP) \
#                svm_autoen.h $(SVM_AUTOENDEP) svm_densit.h $(SVM_DENSITDEP) \
#                svm_pfront.h $(SVM_PFRONTDEP) \
#                svm_biscor.h $(SVM_BISCORDEP) svm_scscor.h $(SVM_SCSCORDEP) \
#                onn_scalar.h $(ONN_SCALARDEP) onn_vector.h $(ONN_VECTORDEP) \
#                onn_anions.h $(ONN_ANIONSDEP) onn_binary.h $(ONN_BINARYDEP) \
#                onn_autoen.h $(ONN_AUTOENDEP) onn_gentyp.h $(ONN_GENTYPDEP) \
#                knn_densit.h $(KNN_DENSITDEP) knn_binary.h $(KNN_BINARYDEP) \
#                knn_gentyp.h $(KNN_GENTYPDEP) knn_scalar.h $(KNN_SCALARDEP) \
#                knn_vector.h $(KNN_VECTORDEP) knn_anions.h $(KNN_ANIONSDEP) \
#                knn_autoen.h $(KNN_AUTOENDEP) knn_multic.h $(KNN_MULTICDEP) \
#                gpr_scalar.h $(GPR_SCALARDEP) gpr_vector.h $(GPR_VECTORDEP) \
#                gpr_anions.h $(GPR_ANIONSDEP) gpr_binary.h $(GPR_BINARYDEP) \
#                lsv_scalar.h $(LSV_SCALARDEP) lsv_vector.h $(LSV_VECTORDEP) \
#                lsv_anions.h $(LSV_ANIONSDEP) lsv_scscor.h $(LSV_SCSCORDEP) \
#                lsv_autoen.h $(LSV_AUTOENDEP) \
#                blk_nopnop.h $(BLK_NOPNOPDEP) blk_consen.h $(BLK_CONSENDEP) \
#                blk_avesca.h $(BLK_AVESCADEP) blk_avevec.h $(BLK_AVEVECDEP) \
#                blk_aveani.h $(BLK_AVEANIDEP) blk_usrfna.h $(BLK_USRFNADEP) \
#                blk_usrfnb.h $(BLK_USRFNBDEP) blk_userio.h $(BLK_USERIODEP) \
#                blk_calbak.h $(BLK_CALBAKDEP) blk_mexfna.h $(BLK_MEXFNADEP) \
#                blk_mexfnb.h $(BLK_MEXFNBDEP) blk_mercer.h $(BLK_MERCERDEP) \
#                blk_conect.h $(BLK_CONECTDEP) blk_system.h $(BLK_SYSTEMDEP) \
#                blk_kernel.h $(BLK_KERNELDEP) blk_bernst.h $(BLK_BERNSTDEP) \
#                blk_batter.h $(BLK_BATTERDEP) \
#                imp_expect.h $(IMP_EXPECTDEP) imp_parsvm.h $(IMP_PARSVMDEP) \
#                ssv_scalar.h $(SSV_SCALARDEP) ssv_binary.h $(SSV_BINARYDEP) \
#                ssv_single.h $(SSV_SINGLEDEP) \
#                mlm_single.h $(MLM_SINGLEDEP)

FUZZYMLDEP        = ml_base.h $(ML_BASEDEP)
XFERMLDEP         = svm_generic.h $(SVM_GENERICDEP) svm_binary.h $(SVM_BINARYDEP) numbase.h $(NUMBASEDEP)
ERRORTESTDEP      = ml_base.h $(ML_BASEDEP)
ADDDATADEP        = ml_base.h $(ML_BASEDEP) basefn.h
ANALYSEANOMALYDEP = ml_mutable.h $(ML_MUTABLEDEP)
BALCDEP           = ml_base.h $(ML_BASEDEP)
HILLCLIMBDEP      = ml_base.h $(ML_BASEDEP)

GLOBALOPTDEP = FNVector.h $(FNVECTORDEP) ml_base.h $(ML_BASEDEP) \
               ml_mutable.h $(ML_MUTABLE) $(ML_MUTABLEDEP) \
               mlcommon.h $(MLCOMMONDEP)
GRIDOPTDEP   = globalopt.h $(GLOBALOPTDEP)
DIRECTOPTDEP = globalopt.h $(GLOBALOPTDEP) direct_direct.h $(DIRECT_DIRECTDEP)
NELDEROPTDEP = globalopt.h $(GLOBALOPTDEP) neldermead.h $(NELDERMEADDEP)
SMBOOPTDEP   = ml_base.h $(ML_BASEDEP) ml_mutable.h $(ML_MUTABLE) \
               globalopt.h $(GLOBALOPTDEP) gridopt.h $(GRIDOPTDEP) \
               directopt.h $(DIRECTOPTDEP) nelderopt.h $(NELDEROPTDEP) \
               gpr_scalar.h $(GPR_SCALARDEP) gpr_vector.h $(GPR_VECTORDEP) \
               errortest.h $(ERRORTESTDEP) addData.h $(ADDDATADEP)
BAYESOPTDEP  = smboopt.h $(SMBOOPTDEP) directopt.h $(DIRECTOPTDEP) \
               imp_generic.h $(IMP_GENERICDEP)
AWARESTREAMDEP = basefn.h 
MLINTERDEP   = ml_base.h $(ML_BASEDEP) mlcommon.h $(MLCOMMONDEP) \
               ml_mutable.h $(ML_MUTABLEDEP) gentype.h $(GENTYPEDEP) \
               ofiletype.h $(OFILETYPEDEP) vecstack.h $(VECSTACKDEP) \
               awarestream.h $(AWARESTREAMDEP)









# Object file creation dependencies.
#
# Basic template for what follows:
#
# x.o: x.cc blah1.h blah2.h ...
#      $(CC) $< -o $@ -c $(CFLAGS)
#
# These are headers included from the .cc file only.  If DEP macros are
# defined for a given header then this must also be included as it will
# expand to the complete dependency tree.
#
# Note that the ML tree pipes to $(MLMUTATEFILE) as the dependency tree
# is too complicated for make.


basefn.o    : basefn.cc basefn.h
	$(CC) $< -o $@ -c $(CFLAGS)
gslrefs.o   : gslrefs.cc gslrefs.h
	$(CC) $< -o $@ -c $(CFLAGS)
zerocross.o : zerocross.cc zerocross.h
	$(CC) $< -o $@ -c $(CFLAGS)

anion.o      : anion.cc anion.h numbase.h $(NUMBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
numbase.o    : numbase.cc numbase.h $(NUMBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
dynarray.o   : dynarray.cc dynarray.h $(DYNARRAYDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
strfns.o     : strfns.cc strfns.h numbase.h $(NUMBASEDEP) vecstack.h $(VECSTACKDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
opttest.o : opttest.cc opttest.h $(OPTTESTDEP) numbase.h $(NUMBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
paretotest.o : paretotest.cc paretotest.h $(PARETOTESTDEP) numbase.h $(NUMBASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
idstore.o    : idstore.cc idstore.h $(IDSTOREDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
smatrix.o    : smatrix.cc smatrix.h $(SMATRIXDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
gentype.o    : gentype.cc gentype.h $(GENTYPEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
mercer.o     : mercer.cc mercer.h $(MERCERDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
mlcommon.o   : mlcommon.cc mlcommon.h $(MLCOMMONDEP) vecstack.h $(VECSTACKDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
kcache.o     : kcache.cc kcache.h $(KCACHEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
FNVector.o   : FNVector.cc FNVector.h $(FNVECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

hyper_psc.o   : hyper_psc.cc hyper_psc.h basefn.h
	$(CC) $< -o $@ -c $(CFLAGS)
hyper_debug.o : hyper_debug.cc hyper_debug.h basefn.h
	$(CC) $< -o $@ -c $(CFLAGS)
hyper_base.o  : hyper_base.cc hyper_base.h hyper_psc.h hyper_debug.h basefn.h numbase.h
	$(CC) $< -o $@ -c $(CFLAGS)
hyper_opt.o   : hyper_opt.cc hyper_opt.h hyper_base.h hyper_psc.h basefn.h numbase.h
	$(CC) $< -o $@ -c $(CFLAGS)
hyper_alt.o   : hyper_alt.cc hyper_alt.h hyper_opt.h hyper_base.h hyper_psc.h basefn.h numbase.h
	$(CC) $< -o $@ -c $(CFLAGS)

optcontext.o        : optcontext.cc optcontext.h $(OPTCONTEXTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
optlinbasecontext.o : optlinbasecontext.cc optlinbasecontext.h $(OPTLINBASECONTEXTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
optlincontext.o     : optlincontext.cc optlincontext.h $(OPTLINCONTEXTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
optlinstate.o       : optlinstate.cc optlinstate.h $(OPTLINSTATEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

sQbase.o : sQbase.cc sQbase.h $(SQBASEDEP) kcache.h $(KCACHEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
sQsLsAsWs.o : sQsLsAsWs.cc sQsLsAsWs.h $(SQSLSQSWSDEP) kcache.h $(KCACHEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
sQsmo.o     : sQsmo.cc sQsmo.h $(SQSMODEP)
	$(CC) $< -o $@ -c $(CFLAGS)
sQd2c.o     : sQd2c.cc sQd2c.h $(SQD2CDEP) sQsmo.h $(SQSMODEP)
	$(CC) $< -o $@ -c $(CFLAGS)
sQgraddesc.o : sQgraddesc.cc sQgraddesc.h $(SQGRADDESCDEP) sQsLsAsWs.h $(SQSLSQSWSDEP) sQd2c.h $(SQD2CDEP) smatrix.h $(SMATRIXDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
linsolve.o  : linsolve.cc linsolve.h $(LINSOLVEDEP) optlinstate.h $(OPTLINSTATEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

direct_direct.o  : direct_direct.cc $(DIRECT_DIRECTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
neldermead.o  : neldermead.cc $(NELDERMEADDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

ml_base.o : ml_base.cc ml_base.h $(ML_BASEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_generic.o : svm_generic.cc svm_generic.h $(SVM_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_scalar.o : svm_scalar.cc svm_scalar.h $(SVM_SCALARDEP) sQsLsAsWs.h $(SQSLSASWSDEP) sQsmo.h $(SQSMODEP) sQd2c.h $(SQD2CDEP) sQgraddesc.h $(SQGRADDESCDEP) linsolve.h $(LINSOLVEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_binary.o : svm_binary.cc svm_binary.h $(SVM_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_simlrn.o : svm_simlrn.cc svm_simlrn.h $(SVM_SIMLRNDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_single.o : svm_single.cc svm_single.h $(SVM_SINGLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_biscor.o : svm_biscor.cc svm_biscor.h $(SVM_BISCORDEF)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_scscor.o : svm_scscor.cc svm_scscor.h $(SVM_SCSCORDEF)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_densit.o : svm_densit.cc svm_densit.h $(SVM_DENSITDEF)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_pfront.o : svm_pfront.cc svm_pfront.h $(SVM_PFRONTDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_vector_atonce_template.o : svm_vector_atonce_template.cc svm_vector_atonce_template.h $(SVM_VECTOR_ATONCE_TEMPLATEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_multic_redbin.o  : svm_multic_redbin.cc svm_multic_redbin.h $(SVM_MULTIC_REDBINDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_multic_atonce.o  : svm_multic_atonce.cc svm_multic_atonce.h $(SVM_MULTIC_ATONCEDEP) smatrix.h $(SMATRIXDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector_redbin.o  : svm_vector_redbin.cc svm_vector_redbin.h $(SVM_VECTOR_REDBINDEP) smatrix.h $(SMATRIXDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector_atonce.o  : svm_vector_atonce.cc svm_vector_atonce.h $(SVM_VECTOR_ATONCEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector_mredbin.o : svm_vector_mredbin.cc svm_vector_mredbin.h $(SVM_VECTOR_MREDBINDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector_matonce.o : svm_vector_matonce.cc svm_vector_matonce.h $(SVM_VECTOR_MATONCEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

svm_multic.o : svm_multic.cc svm_multic.h $(SVM_MULTICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_vector.o : svm_vector.cc svm_vector.h $(SVM_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_anions.o : svm_anions.cc svm_anions.h $(SVM_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_gentyp.o : svm_gentyp.cc svm_gentyp.h $(SVM_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_autoen.o : svm_autoen.cc svm_autoen.h $(SVM_AUTOENDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_planar.o : svm_planar.cc svm_planar.h $(SVM_PLANARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_mvrank.o : svm_mvrank.cc svm_mvrank.h $(SVM_MVRANKDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_mulbin.o : svm_mulbin.cc svm_mulbin.h $(SVM_MULBINDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
svm_cyclic.o : svm_cyclic.cc svm_cyclic.h $(SVM_CYCLICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

onn_generic.o : onn_generic.cc onn_generic.h $(ONN_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

onn_scalar.o : onn_scalar.cc onn_scalar.h $(ONN_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
onn_binary.o : onn_binary.cc onn_binary.h $(ONN_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
onn_vector.o : onn_vector.cc onn_vector.h $(ONN_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
onn_anions.o : onn_anions.cc onn_anions.h $(ONN_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
onn_autoen.o : onn_autoen.cc onn_autoen.h $(ONN_AUTOENDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
onn_gentyp.o : onn_gentyp.cc onn_gentyp.h $(ONN_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

knn_generic.o : knn_generic.cc knn_generic.h $(KNN_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

knn_scalar.o : knn_scalar.cc knn_scalar.h $(KNN_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_binary.o : knn_binary.cc knn_binary.h $(KNN_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_multic.o : knn_multic.cc knn_multic.h $(KNN_MULTICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_vector.o : knn_vector.cc knn_vector.h $(KNN_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_anions.o : knn_anions.cc knn_anions.h $(KNN_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_autoen.o : knn_autoen.cc knn_autoen.h $(KNN_AUTOENDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_gentyp.o : knn_gentyp.cc knn_gentyp.h $(KNN_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
knn_densit.o : knn_densit.cc knn_densit.h $(KNN_DENSITDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

blk_generic.o : blk_generic.cc blk_generic.h $(BLK_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

blk_nopnop.o : blk_nopnop.cc blk_nopnop.h $(BLK_NOPNOPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_consen.o : blk_consen.cc blk_consen.h $(BLK_CONSENDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_avesca.o : blk_avesca.cc blk_avesca.h $(BLK_AVESCADEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_avevec.o : blk_avevec.cc blk_avevec.h $(BLK_AVEVECDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_aveani.o : blk_aveani.cc blk_aveani.h $(BLK_AVEANIDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_usrfna.o : blk_usrfna.cc blk_usrfna.h $(BLK_USRFNADEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_usrfnb.o : blk_usrfnb.cc blk_usrfnb.h $(BLK_USRFNBDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_userio.o : blk_userio.cc blk_userio.h $(BLK_USERIODEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_calbak.o : blk_calbak.cc blk_calbak.h $(BLK_CALBAKDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_mexfna.o : blk_mexfna.cc blk_mexfna.h $(BLK_MEXFNADEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_mexfnb.o : blk_mexfnb.cc blk_mexfnb.h $(BLK_MEXFNBDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_mercer.o : blk_mercer.cc blk_mercer.h $(BLK_MERCERDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_conect.o : blk_conect.cc blk_conect.h $(BLK_CONECTDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_system.o : blk_system.cc blk_system.h $(BLK_SYSTEMDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_kernel.o : blk_kernel.cc blk_kernel.h $(BLK_KERNELDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_bernst.o : blk_bernst.cc blk_bernst.h $(BLK_BERNSTDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
blk_batter.o : blk_batter.cc blk_batter.h $(BLK_BATTERDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

imp_generic.o : imp_generic.cc imp_generic.h $(IMP_GENERICDEP) hyper_base.h 
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

imp_expect.o : imp_expect.cc imp_expect.h $(IMP_EXPECTDEP) hyper_alt.h hyper_base.h
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
imp_parsvm.o : imp_parsvm.cc imp_parsvm.h $(IMP_PARSVMDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

lsv_generic.o : lsv_generic.cc lsv_generic.h $(LSV_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

lsv_scalar.o : lsv_scalar.cc lsv_scalar.h $(LSV_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_vector.o : lsv_vector.cc lsv_vector.h $(LSV_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_anions.o : lsv_anions.cc lsv_anions.h $(LSV_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_scscor.o : lsv_scscor.cc lsv_scscor.h $(LSV_SCSCORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_autoen.o : lsv_autoen.cc lsv_autoen.h $(LSV_AUTOENDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_gentyp.o : lsv_gentyp.cc lsv_gentyp.h $(LSV_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_planar.o : lsv_planar.cc lsv_planar.h $(LSV_PLANARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
lsv_mvrank.o : lsv_mvrank.cc lsv_mvrank.h $(LSV_MVRANKDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

gpr_generic.o : gpr_generic.cc gpr_generic.h $(GPR_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

gpr_scalar.o : gpr_scalar.cc gpr_scalar.h $(GPR_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_vector.o : gpr_vector.cc gpr_vector.h $(GPR_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_anions.o : gpr_anions.cc gpr_anions.h $(GPR_ANIONSDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_gentyp.o : gpr_gentyp.cc gpr_gentyp.h $(GPR_GENTYPDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
gpr_binary.o : gpr_binary.cc gpr_binary.h $(GPR_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

ssv_generic.o : ssv_generic.cc ssv_generic.h $(SSV_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

ssv_scalar.o : ssv_scalar.cc ssv_scalar.h $(SSV_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
ssv_binary.o : ssv_binary.cc ssv_binary.h $(SSV_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
ssv_single.o : ssv_single.cc ssv_single.h $(SSV_SINGLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

mlm_generic.o : mlm_generic.cc mlm_generic.h $(MLM_GENERICDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

mlm_scalar.o : mlm_scalar.cc mlm_scalar.h $(MLM_SCALARDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
mlm_binary.o : mlm_binary.cc mlm_binary.h $(MLM_BINARYDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)
mlm_vector.o : mlm_vector.cc mlm_vector.h $(MLM_VECTORDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

ml_mutable.o : ml_mutable.cc ml_mutable.h $(ML_MUTABLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS) > $(MLMUTATEFILE)

fuzzyml.o        : fuzzyml.cc fuzzyml.h $(FUZZYMLDEP) svm_single.h $(SVM_SINGLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
xferml.o         : xferml.cc xferml.h $(XFERMLDEP) svm_scalar.h $(SVM_SCALARDEP) mlcommon.h $(MLCOMMONDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
errortest.o      : errortest.cc errortest.h $(ERRORTESTDEP) ml_mutable.h $(ML_MUTABLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
addData.o        : addData.cc addData.h $(ADDDATADEP) ml_mutable.h $(ML_MUTABLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
analyseAnomaly.o : analyseAnomaly.cc analyseAnomaly.h $(ANALYSEANOMALYDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
balc.o           : balc.cc balc.h $(BALCDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
hillclimb.o      : hillclimb.cc hillclimb.h $(HILLCLIMBDEP) errortest.h $(ERRORTESTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

globalopt.o : globalopt.cc globalopt.h $(GLOBALOPTDEP) hyper_base.h
	$(CC) $< -o $@ -c $(CFLAGS)
gridopt.o   : gridopt.cc gridopt.h $(GRIDOPTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
directopt.o : directopt.cc directopt.h $(DIRECTOPTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
nelderopt.o : nelderopt.cc nelderopt.h $(NELDEROPTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
smboopt.o : smboopt.cc smboopt.h $(SMBOOPTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)
bayesopt.o  : bayesopt.cc bayesopt.h $(BAYESOPTDEP) ml_mutable.h $(ML_MUTABLEDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

awarestream.o : awarestream.cc awarestream.h $(AWARESTREAMDEP)
	$(CC) $< -o $@ -c $(CFLAGS)

mlinter.o   : mlinter.cc mlinter.h $(MLINTERDEP) hillclimb.h $(HILLCLIMBDEP) fuzzyml.h $(FUZZYMLDEP) xferml.h $(XFERMLDEP) errortest.h $(ERRORTESTDEP) addData.h $(ADDDATADEP) analyseAnomaly.h $(ANALYSEANOMALYDEP) balc.h $(BALCDEP) gridopt.h $(GRIDOPTDEP) directopt.h $(DIRECTOPTDEP) nelderopt.h $(NELDEROPTDEP) smboopt.h $(SMBOOPTDEP) bayesopt.h $(BAYESOPTDEP) globalopt.h $(GLOBALOPTDEP) opttest.h $(OPTTESTDEP) paretotest.h $(PARETOTESTDEP)
	$(CC) $< -o $@ -c $(CFLAGS)



