
//
// Serial ML module
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include "ml_serial.h"




ML_Base *makeMLSerial(void)
{
    ML_Base *res;

    MEMNEW(res,ML_Serial());

    return res;
}

void assigntoMLSerial(ML_Base **dest, const ML_Base *src, int onlySemiCopy)
{
    dynamic_cast<ML_Serial  &>((**dest).getML()).assign(dynamic_cast<const ML_Serial  &>((*src).getMLconst()),onlySemiCopy);

    return;
}

void xferMLSerial(ML_Base &dest, ML_Base &src)
{
    dynamic_cast<ML_Serial  &>(dest) = dynamic_cast<const ML_Serial  &>(src);

    return;
}


ML_Serial::ML_Serial() : ML_Mutable(), datastore(this)
{
    ML_Serial::resizetheML(0);

    //NB: by default the parent initialises the elements in theML to be
    //    pointers to type SVM_Scalar, SVM_Binary, ... rather than the
    //    generic ML_Mutable.  We actually want them to point to ML_Mutable,
    //    so we need to specifically construct them as such!

    mlType = -2;
    mlind  = 0;
    theML.resize(1);
    MEMNEW(theML("&",mlind),ML_Mutable); //NB specified mutable type!

    setaltx(NULL);

    return;
}

ML_Serial::ML_Serial(const ML_Serial &src) : ML_Mutable(), datastore(this,src.datastore)
{
    ML_Serial::resizetheML(0);

    mlType = src.mlType;
    mlind  = src.mlind;
    theML.resize((src.theML).size());

    if ( theML.size() )
    {
        int i;

        for ( i = 0 ; i < theML.size() ; i++ )
        {
            MEMNEW(theML("&",i),ML_Mutable(dynamic_cast<const ML_Mutable &>(*((src.theML)(i)))));
        }
    }

    setaltx(NULL);

    return;
}

ML_Serial::ML_Serial(const ML_Serial &src, const ML_Base *srcx) : ML_Mutable(), datastore(this,src.datastore,srcx)
{
    ML_Serial::resizetheML(0);

    mlType = src.mlType;
    mlind  = src.mlind;
    theML.resize((src.theML).size());

    if ( theML.size() )
    {
        int i;

        for ( i = 0 ; i < theML.size() ; i++ )
        {
            MEMNEW(theML("&",i),ML_Mutable(dynamic_cast<const ML_Mutable &>(*((src.theML)(i)))));
        }
    }

    setaltx(NULL);

    return;
}

ML_Serial::~ML_Serial()
{
    return;
}

int ML_Serial::addLayer(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= numLayers() );

    theML.add(i);

    MEMNEW(theML("&",i),ML_Mutable);

    return 1;
}

int ML_Serial::removeLayer(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < numLayers() );

    MEMDEL(theML("&",i));

    theML.remove(i);

    return 1;
}

int ML_Serial::setNumLayers(int n)
{
    int res = 0;

    while ( numLayers() > n )
    {
        res |= removeLayer(numLayers()-1);
    }

    while ( numLayers() < n )
    {
        res |= addLayer(numLayers());
    }

    return res;
}

int ML_Serial::setActiveLayer(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < numLayers() );

    mlind = i;

    return 0;
}





























































int SerialBackcall::tspaceDim(void) const
{
    return (owner->lastMLconst)().tspaceDim();
}

int SerialBackcall::xspaceDim(void)  const
{
    return (owner->firstMLconst)().xspaceDim();
}

int SerialBackcall::tspaceSparse(void) const
{
    return (owner->lastMLconst)().tspaceSparse();
}

int SerialBackcall::xspaceSparse(void) const
{
    return (owner->firstMLconst)().xspaceSparse();
}

int SerialBackcall::numClasses(void) const
{
    return (owner->lastMLconst)().numClasses();
}

int SerialBackcall::order(void) const
{
    return (owner->lastMLconst)().order();
}

char SerialBackcall::gOutType(void) const
{
    return (owner->lastMLconst)().gOutType();
}

char SerialBackcall::hOutType(void) const
{
    return (owner->lastMLconst)().hOutType();
}

char SerialBackcall::targType(void) const
{
    return (owner->lastMLconst)().targType();
}

double SerialBackcall::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    return (owner->lastMLconst)().calcDist(ha,hb,ia,db);
}

const Vector<int> &SerialBackcall::ClassLabels(void) const
{
    return (owner->lastMLconst)().ClassLabels();
}

int SerialBackcall::getInternalClass(const gentype &y) const
{
    return (owner->lastMLconst)().getInternalClass(y);
}

int SerialBackcall::numInternalClasses(void) const
{
    return (owner->lastMLconst)().numInternalClasses();
}

int SerialBackcall::isClassifier(void) const
{
    return (owner->lastMLconst)().isClassifier();
}

int SerialBackcall::isRegression(void) const
{
    return (owner->lastMLconst)().isRegression();
}


SerialBackcall::SerialBackcall(ML_Serial *xowner) : BLK_Nopnop()
{
    NiceAssert( xowner );

    owner = xowner;

    return;
}

SerialBackcall::SerialBackcall(ML_Serial *xowner, const SerialBackcall &src) : BLK_Nopnop()
{
    NiceAssert( xowner );

    owner = xowner;

    setaltx(NULL);

    assign(src,0);

    return;
}

SerialBackcall::SerialBackcall(ML_Serial *xowner, const SerialBackcall &src, const ML_Base *xsrcx) : BLK_Nopnop()
{
    NiceAssert( xowner );

    owner = xowner;

    setaltx(xsrcx);

    assign(src,0);

    return;
}

SerialBackcall::~SerialBackcall()
{
    return;
}

std::ostream &SerialBackcall::printstream(std::ostream &output) const
{
    output << "Serial Network\n\n";

    output << "Module type:    " << owner->mlType    << "\n";
    output << "Module index:   " << owner->mlind     << "\n";

    if ( (owner->theML).size() )
    {
        int i;

        for ( i = 0 ; i < (owner->theML).size() ; i++ )
        {
            output << "Module number " << i << ": " << *((owner->theML)(i)) << "\n";
        }
    }

    output << "\n";

    return BLK_Nopnop::printstream(output);
}

std::istream &SerialBackcall::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> owner->mlType;
    input >> dummy; input >> owner->mlind;

    int n;

    input >> dummy; input >> n;

    setNumLayers(n);

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            input >> dummy; input >> *((owner->theML)(i));
        }
    }

    return BLK_Nopnop::inputstream(input);
}

int SerialBackcall::isTrained(void) const
{
    int res = 1;

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            if ( !((*((owner->theML)(i))).isTrained()) )
            {
                res = 0;
                break;
            }
        }
    }

    return res;
}

double SerialBackcall::sparlvl(void) const
{
    double res = 0;

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res += ((*((owner->theML)(i)))).sparlvl();
        }

        res /= numLayers();
    }

    return res;
}

int SerialBackcall::randomise(double sparsity)
{
    int res = BLK_Nopnop::randomise(sparsity);

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).randomise(sparsity);
        }
    }

    return res;
}

int SerialBackcall::autoen(void)
{
    int res = BLK_Nopnop::autoen();

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).autoen();
        }
    }

    return res;
}

int SerialBackcall::renormalise(void)
{
    int res = BLK_Nopnop::renormalise();

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).renormalise();
        }
    }

    return res;
}

int SerialBackcall::realign(void)
{
    int res = BLK_Nopnop::realign();

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).realign();
        }
    }

    return res;
}

int SerialBackcall::scale(double a)
{
    int res = BLK_Nopnop::scale(a);

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).scale(a);
        }
    }

    return res;
}

int SerialBackcall::reset(void)
{
    int res = BLK_Nopnop::reset();

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).reset();
        }
    }

    return res;
}

int SerialBackcall::restart(void)
{
    int res = BLK_Nopnop::restart();

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).restart();
        }
    }

    return res;
}

int SerialBackcall::home(void)
{
    owner->mlind = -1;

    int res = 0;

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).home();
        }
    }

    return res;
}

int SerialBackcall::train(int &res, svmvolatile int &killSwitch)
{
    int retcode = 0;

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            retcode |= ((*((owner->theML)("&",i)))).train(res,killSwitch);
        }
    }

    return retcode;
}

void SerialBackcall::eTrainingVector(gentype &res, int i) const
{
    gg(res,i);
    res -= y(i);

    return;
}

int SerialBackcall::gvTrainingVector(gentype &resv, int ii) const
{
    int res = 0;

    if ( numLayers() == 1 )
    {
        res = ((*((owner->theML)(zeroint())))).gv(resv,BLK_Nopnop::getx(ii));
    }

    else
    {
        NiceAssert( numLayers() > 1 );

        gentype temph,tempg;
        SparseVector<gentype> tempx;

        res = ((*((owner->theML)(zeroint())))).gh(temph,tempg,BLK_Nopnop::getx(ii));

        int i;

        if ( numLayers() > 2 )
        {
            for ( i = 1 ; i < numLayers()-1 ; i++ )
            {
                if ( ((*((owner->theML)(i-1)))).gOutType() == 'V' )
                {
                    tempx = tempg.cast_vector();
                }

                else
                {
                    tempx.zero();
                    tempx("&",0) = tempg;
                }

                res = ((*((owner->theML)(i)))).gh(temph,tempg,tempx);
            }
        }

        if ( ((*((owner->theML)(numLayers()-2)))).gOutType() == 'V' )
        {
            tempx = tempg.cast_vector();
        }

        else
        {
            tempx.zero();
            tempx("&",0) = tempg;
        }

        res = ((*((owner->theML)(numLayers()-1)))).gv(resv,tempx);
    }

    return res;
}

/*int SerialBackcall::gvTrainingVector(Matrix<gentype> &resv, int Nx, int dummy) const
{
    int res = 0;

    if ( numLayers() == 1 )
    {
        res = ((*((owner->theML)(zeroint())))).gv(resv,Nx,dummy);
    }

    else
    {
        NiceAssert( numLayers() > 1 );

        gentype temph,tempg;
        SparseVector<gentype> tempx;

        res = ((*((owner->theML)(zeroint())))).gh(temph,tempg,BLK_Nopnop::getx(ii));

        int i;

        if ( numLayers() > 2 )
        {
            for ( i = 1 ; i < numLayers()-1 ; i++ )
            {
                if ( ((*((owner->theML)(i-1)))).gOutType() == 'V' )
                {
                    tempx = tempg.cast_vector();
                }

                else
                {
                    tempx.zero();
                    tempx("&",0) = tempg;
                }

                res = ((*((owner->theML)(i)))).gh(temph,tempg,tempx);
            }
        }

        if ( ((*((owner->theML)(numLayers()-2)))).gOutType() == 'V' )
        {
            tempx = tempg.cast_vector();
        }

        else
        {
            tempx.zero();
            tempx("&",0) = tempg;
        }

        res = ((*((owner->theML)(numLayers()-1)))).gv(resv,Nx,dummy);
    }

    return res;
}*/

int SerialBackcall::ghTrainingVector(gentype &resh, gentype &resg, int ii, int retaltg) const
{
    int res = 0;

    if ( numLayers() == 1 )
    {
        res = ((*((owner->theML)(zeroint())))).gh(resh,resg,BLK_Nopnop::getx(ii),retaltg);
    }

    else
    {
        NiceAssert( numLayers() > 1 );

        Vector<double> tempg;
        SparseVector<gentype> tempx;

        res = ((*((owner->theML)(zeroint())))).gg(tempg,BLK_Nopnop::getx(ii));

        int i,j;

        tempx.zero();

        if ( tempg.size() )
        {
            for ( j = 0 ; j < tempg.size() ; j++ )
            {
                tempx("&",j) = tempg(j);
            }
        }

        if ( numLayers() > 2 )
        {
            for ( i = 1 ; i < numLayers()-1 ; i++ )
            {
                res = ((*((owner->theML)(i)))).gg(tempg,tempx);

                tempx.zero();

                if ( tempg.size() )
                {
                    for ( j = 0 ; j < tempg.size() ; j++ )
                    {
                        tempx("&",j) = tempg(j);
                    }
                }
            }
        }

        res = ((*((owner->theML)(numLayers()-1)))).gh(resh,resg,tempx,retaltg);
    }

    return res;
}

int SerialBackcall::ggTrainingVector(double &resg, int ii, int retaltg) const
{
    int res = 0;

    if ( numLayers() == 1 )
    {
        res = ((*((owner->theML)(zeroint())))).gg(resg,BLK_Nopnop::getx(ii),retaltg);
    }

    else
    {
        NiceAssert( numLayers() > 1 );

        Vector<double> tempg;
        SparseVector<gentype> tempx;

        res = ((*((owner->theML)(zeroint())))).gg(tempg,BLK_Nopnop::getx(ii));

        int i,j;

        tempx.zero();

        if ( tempg.size() )
        {
            for ( j = 0 ; j < tempg.size() ; j++ )
            {
                tempx("&",j) = tempg(j);
            }
        }

        if ( numLayers() > 2 )
        {
            for ( i = 1 ; i < numLayers()-1 ; i++ )
            {
                res = ((*((owner->theML)(i)))).gg(tempg,tempx);

                tempx.zero();

                if ( tempg.size() )
                {
                    for ( j = 0 ; j < tempg.size() ; j++ )
                    {
                        tempx("&",j) = tempg(j);
                    }
                }
            }
        }

        res = ((*((owner->theML)(numLayers()-1)))).gg(resg,tempx,retaltg);
    }

    return res;
}

int SerialBackcall::ggTrainingVector(Vector<double> &resg, int ii, int retaltg) const
{
    int res = 0;

    if ( numLayers() == 1 )
    {
        res = ((*((owner->theML)(zeroint())))).gg(resg,BLK_Nopnop::getx(ii),retaltg);
    }

    else
    {
        NiceAssert( numLayers() > 1 );

        Vector<double> tempg;
        SparseVector<gentype> tempx;

        res = ((*((owner->theML)(zeroint())))).gg(tempg,BLK_Nopnop::getx(ii));

        int i,j;

        tempx.zero();

        if ( tempg.size() )
        {
            for ( j = 0 ; j < tempg.size() ; j++ )
            {
                tempx("&",j) = tempg(j);
            }
        }

        if ( numLayers() > 2 )
        {
            for ( i = 1 ; i < numLayers()-1 ; i++ )
            {
                res = ((*((owner->theML)(i)))).gg(tempg,tempx);

                tempx.zero();

                if ( tempg.size() )
                {
                    for ( j = 0 ; j < tempg.size() ; j++ )
                    {
                        tempx("&",j) = tempg(j);
                    }
                }
            }
        }

        res = ((*((owner->theML)(numLayers()-1)))).gg(resg,tempx,retaltg);
    }

    return res;
}

int SerialBackcall::ggTrainingVector(d_anion &resg, int ii, int retaltg) const
{
    int res = 0;

    if ( numLayers() == 1 )
    {
        res = ((*((owner->theML)(zeroint())))).gg(resg,BLK_Nopnop::getx(ii),retaltg);
    }

    else
    {
        NiceAssert( numLayers() > 1 );

        Vector<double> tempg;
        SparseVector<gentype> tempx;

        res = ((*((owner->theML)(zeroint())))).gg(tempg,BLK_Nopnop::getx(ii));

        int i,j;

        tempx.zero();

        if ( tempg.size() )
        {
            for ( j = 0 ; j < tempg.size() ; j++ )
            {
                tempx("&",j) = tempg(j);
            }
        }

        if ( numLayers() > 2 )
        {
            for ( i = 1 ; i < numLayers()-1 ; i++ )
            {
                res = ((*((owner->theML)(i)))).gg(tempg,tempx);

                tempx.zero();

                if ( tempg.size() )
                {
                    for ( j = 0 ; j < tempg.size() ; j++ )
                    {
                        tempx("&",j) = tempg(j);
                    }
                }
            }
        }

        res = ((*((owner->theML)(numLayers()-1)))).gg(resg,tempx,retaltg);
    }

    return res;
}

void SerialBackcall::dgTrainingVector(SparseVector<gentype> &resx, int ii) const
{
    if ( numLayers() == 1 )
    {
        ((*((owner->theML)(zeroint())))).dg(resx,BLK_Nopnop::y(ii),BLK_Nopnop::getx(ii));
    }

    else
    {
        NiceAssert( numLayers() > 1 );

        gentype temph,tempg;
        SparseVector<gentype> tempx;
        SparseVector<gentype> tempresx;

        ((*((owner->theML)(zeroint())))).dg(resx,BLK_Nopnop::y(ii),BLK_Nopnop::getx(ii));
        ((*((owner->theML)(zeroint())))).gh(temph,tempg,BLK_Nopnop::getx(ii));

        int i,j;

        if ( numLayers() > 2 )
        {
            for ( i = 1 ; i < numLayers()-1 ; i++ )
            {
                if ( ((*((owner->theML)(i-1)))).gOutType() == 'V' )
                {
                    tempx = tempg.cast_vector();
                }

                else
                {
                    tempx.zero();
                    tempx("&",0) = tempg;
                }

                ((*((owner->theML)(i)))).dg(tempresx,BLK_Nopnop::y(ii),tempx);
                ((*((owner->theML)(i)))).gh(temph,tempg,tempx);

                if ( ((*((owner->theML)(i-1)))).gOutType() == 'V' )
                {
                    // Notes:
                    //
                    // - tempresx() is just tempresx in (non-sparse) vector
                    // - resx("&",j) is the gradient of x(j) wrt the output
                    //   of layer i-1, which is a vector.
                    // - multiplying these two results in an inner product,
                    //   the output of which is the gradient of x(j) wrt
                    //   the output of layer i, which may or may no be a 
                    //   vector itself.

                    for ( j = 0 ; j < resx.nearindsize() ; j++ )
                    {
                        resx("&",j) = (resx(j).cast_vector())*tempresx();
                    }
                }

                else
                {
                    // Notes:
                    //
                    // - tempresx is a vector with a single element, and we
                    //   know that the index of this element must be zero
                    //   as there are no offsets in ML_Serial and it is just
                    //   the output of the previous layer.
                    // - resx is the gradient of x wrt to the output of
                    //   lyaer i-1, which is a sparse vector containing
                    //   just one value at index 0.
                    // - this produce will multiply each element in resx
                    //   by the gradient of the next layer, which is what
                    //   we require.

                    resx *= tempresx(zeroint());
                }
            }
        }

        if ( ((*((owner->theML)(numLayers()-2)))).gOutType() == 'V' )
        {
            tempx = tempg.cast_vector();
        }

        else
        {
            tempx.zero();
            tempx("&",0) = tempg;
        }

        ((*((owner->theML)(numLayers()-1)))).dg(tempresx,BLK_Nopnop::y(ii),tempx);
        ((*((owner->theML)(numLayers()-1)))).gh(temph,tempg,tempx);

        if ( ((*((owner->theML)(numLayers()-2)))).gOutType() == 'V' )
        {
            for ( j = 0 ; j < resx.nearindsize() ; j++ )
            {
                resx("&",j) = (resx(j).cast_vector())*tempresx();
            }
        }

        else
        {
            resx *= tempresx(zeroint());
        }
    }

    return;
}


