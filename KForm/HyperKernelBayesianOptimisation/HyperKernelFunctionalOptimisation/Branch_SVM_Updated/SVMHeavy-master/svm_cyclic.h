
//
// Cyclic regression SVM
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

//
// Basic idea: take a dataset
//
// (x_i vector, y_i n-dim unit vector, d_i) i=1,2,...,N
//
// gets transformed into (epsilon being the usual margin term):
//
// (x_i, y_i, 0) first N 
// (::: 0:i 7:v_i1, epsilon, +1) second N
// (::: 0:i 7:v_i2, epsilon, +1) third N
// ...
// (::: 0:i 7:v_in, epsilon, +1) n-th N
//
// where v_ij is the j'th plane supporting the cone pointed
// on the axis y_i - that is, if the cone pointed on axis 
// (1,1,...,1) is supported by:
//
// v_1,v_2,...,v_n
//
// where:
//
// v_j = ( eps.delta_j + (1-eps).( delta_j - (1,1,...,1)/||(1,1,...,1)||^2 ) )   (then normalised)
// v_j = ( eps.delta_j + (1-eps).( delta_j - (1,1,...,1)/n ) )   (then normalised)
//
// (so eps = 1 makes the cone just the upper quadrant shifted 
// so the point lies on to (1,1,...,1), and if eps = 0 then this
// is simply a line in the direction (1,1,...,1), then:
//
// v_ij = ( I - 2.( y - (1,1,...,1)/sqrt(n) ).( y - (1,1,...,1)/sqrt(n) )'/|| y - (1,1,...,1)/sqrt(n) ||_2^2 ).v_j (which is already normalised)
//
// (that is, v_j reflected in the hyperplane separating (1,1,...,1) and y)
//
//
// The constraints have the form:
//
// v_ij' g(x) >= q
//
// For the templates v_1 we want
//
// v_i' 1/sqrt(n) = q
//
// (that is, the unit vector in the + quadrant lies on the relevant
// margins).  So we require
//
// q = (1' v_i)/sqrt(n)
//

//
// Notes:
//
// - the VV basis needs to be the standard basis for the n-dim Euclidean space
// - need to store y locally
// - need to ensure default axis is set (except during evaluation).  That way 
//   Gp(i,j) can evaluate to double everywhere as required.  Note that 
//   ghTrainingVector(i>=0) will need to be calculated each time as there is
//   no "pre-calculated" version waiting to be used (see ghEvalFull in svm_planar).
//

#ifndef _svm_cyclic_h
#define _svm_cyclic_h

#include <iostream>
#include <string>
#include "svm_planar.h"



class SVM_Cyclic;

// Swap function

inline void qswap(SVM_Cyclic &a, SVM_Cyclic &b);


class SVM_Cyclic : public SVM_Planar
{
public:

    // Constructors, destructors, assignment etc..

    SVM_Cyclic();
    SVM_Cyclic(const SVM_Cyclic &src);
    SVM_Cyclic(const SVM_Cyclic &src, const ML_Base *xsrc);
    SVM_Cyclic &operator=(const SVM_Cyclic &src) { assign(src); return *this; }
    virtual ~SVM_Cyclic();

    virtual int prealloc(int expectedN);
    virtual int preallocsize(void) const { return SVM_Planar::preallocsize(); }

    virtual void assign(const ML_Base &src, int onlySemiCopy = 0);
    virtual void semicopy(const ML_Base &src);
    virtual void qswapinternal(ML_Base &b);

    virtual std::ostream &printstream(std::ostream &output, int dep) const;
    virtual std::istream &inputstream(std::istream &input);

    // Information functions (training data):

    virtual int N(void)       const { return locy.size(); }
    virtual int type(void)    const { return 20; }
    virtual int subtype(void) const { return 0;  }

    virtual int tspaceDim(void)    const { return locy.size() ? locy(0).size() : 0; }
    virtual int tspaceSparse(void) const { return 0; }

    virtual char gOutType(void) const { return 'V'; }
    virtual char hOutType(void) const { return 'V'; }
    virtual char targType(void) const { return 'V'; }
    virtual double calcDist(const gentype &ha, const gentype &hb, int i = -1, int db = 2) const;

    virtual int isUnderlyingScalar(void) const { return 1; }
    virtual int isUnderlyingVector(void) const { return 0; }
    virtual int isUnderlyingAnions(void) const { return 0; }

    virtual double eps(void) const { return cyceps; }

    virtual const Vector<gentype> &y        (void) const { return locyg;        }
    virtual const Vector<int>     &d        (void) const { return locd;         }
    virtual const Vector<double>  &epsweight(void) const { return cycepsweight; }

    // Kernel stuff

    virtual int resetKernel(int modind = 1, int onlyChangeRowI = -1, int updateInfo = 1);
    virtual int setKernel(const MercerKernel &xkernel, int modind = 1, int onlyChangeRowI = -1);

    // Add/remove data:

    virtual int addTrainingVector (int i, const gentype &z, const SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);
    virtual int qaddTrainingVector(int i, const gentype &z,       SparseVector<gentype> &x, double Cweigh = 1, double epsweigh = 1);

    virtual int addTrainingVector (int i, const Vector<gentype> &z, const Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);
    virtual int qaddTrainingVector(int i, const Vector<gentype> &z,       Vector<SparseVector<gentype> > &x, const Vector<double> &Cweigh, const Vector<double> &epsweigh);

    virtual int removeTrainingVector(int i) { SparseVector<gentype> x; gentype y; return removeTrainingVector(i,y,x); }
    virtual int removeTrainingVector(int i, gentype &y, SparseVector<gentype> &x);
    virtual int removeTrainingVector(int i, int num);

    virtual int setx(int i, const SparseVector<gentype> &x);
    virtual int setx(const Vector<int> &i, const Vector<SparseVector<gentype> > &x);
    virtual int setx(const Vector<SparseVector<gentype> > &x);

    virtual int sety(int i, const gentype &z);
    virtual int sety(const Vector<int> &i, const Vector<gentype> &z);
    virtual int sety(const Vector<gentype> &z);

    virtual int setd(const Vector<int> &i, const Vector<int> &d);
    virtual int setd(const Vector<int> &d);
    virtual int setd(int i, int d);

    virtual int seteps(double xeps);

    virtual int setepsweight(int i, double xepsweight);
    virtual int setepsweight(const Vector<int> &i, const Vector<double> &xepsweight);
    virtual int setepsweight(const Vector<double> &xepsweight);

    virtual const gentype &y(int i) const { return y()(i); }

    // Modification

    virtual int restart(void) { SVM_Cyclic temp; *this = temp; return 1; }

    // Evaluation:

    virtual int ghTrainingVector(gentype &resh, gentype &resg, int i, int retaltg, gentype ***pxyprodi = NULL) const;

    // Other functions

    virtual int disable(int i);
    virtual int disable(const Vector<int> &i);

private:

    // Local basis and factorisation

    Vector<Vector<double> > locy;
    Vector<gentype> locyg;
    Vector<int> locd;
    Vector<double> cycepsweight;
    double cyceps;

    // Distance required for given eps

    double qval;

    // Fix reflections due to y or cyceps change (-1 means all)

    gentype &calcvij(gentype &res, double &q, int i, int j);

    // thisthisthisthisthisthisthatthisthisthis

    SVM_Cyclic *thisthis;
    SVM_Cyclic **thisthisthis;
};

inline void qswap(SVM_Cyclic &a, SVM_Cyclic &b)
{
    a.qswapinternal(b);

    return;
}

inline void SVM_Cyclic::qswapinternal(ML_Base &bb)
{
    NiceAssert( isQswapCompat(*this,bb) );

    SVM_Cyclic &b = dynamic_cast<SVM_Cyclic &>(bb.getML());

    SVM_Planar::qswapinternal(b);
    
    qswap(locy        ,b.locy        );
    qswap(locyg       ,b.locyg       );
    qswap(locd        ,b.locd        );
    qswap(cycepsweight,b.cycepsweight);
    qswap(cyceps      ,b.cyceps      );
    qswap(qval        ,b.qval        );

    return;
}

inline void SVM_Cyclic::semicopy(const ML_Base &bb)
{
    NiceAssert( isSemicopyCompat(*this,bb) );

    const SVM_Cyclic &b = dynamic_cast<const SVM_Cyclic &>(bb.getMLconst());

    SVM_Planar::semicopy(b);

    //locy         = b.locy;
    //locyg        = b.locyg;
    locd         = b.locd;
    cycepsweight = b.cycepsweight;
    cyceps       = b.cyceps;
    qval         = b.qval;

    return;
}

inline void SVM_Cyclic::assign(const ML_Base &bb, int onlySemiCopy)
{
    NiceAssert( isAssignCompat(*this,bb) );

    const SVM_Cyclic &src = dynamic_cast<const SVM_Cyclic &>(bb.getMLconst());

    SVM_Planar::assign(static_cast<const SVM_Planar &>(src),onlySemiCopy);

    locy         = src.locy;
    locyg        = src.locyg;
    locd         = src.locd;
    cycepsweight = src.cycepsweight;
    cyceps       = src.cyceps;
    qval         = src.qval;

    return;
}

#endif
