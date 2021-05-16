
//
// Parallel ML module
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
#include "ml_parall.h"





ML_Base *makeMLParall(void)
{
    ML_Base *res;

    MEMNEW(res,ML_Parall());

    return res;
}

void assigntoMLParall(ML_Base **dest, const ML_Base *src, int onlySemiCopy)
{
    dynamic_cast<ML_Parall  &>((**dest).getML()).assign(dynamic_cast<const ML_Parall  &>((*src).getMLconst()),onlySemiCopy);

    return;
}

void xferMLParall(ML_Base &dest, ML_Base &src)
{
    dynamic_cast<ML_Parall  &>(dest) = dynamic_cast<const ML_Parall  &>(src);

    return;
}





ML_Parall::ML_Parall() : ML_Mutable(), datastore(this)
{
    ML_Parall::resizetheML(0);

    resoffset.resize(1);
    resoffset = zeroint();

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

ML_Parall::ML_Parall(const ML_Parall &src) : ML_Mutable(), datastore(this,src.datastore)
{
    ML_Parall::resizetheML(0);

    resoffset = src.resoffset;

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

ML_Parall::ML_Parall(const ML_Parall &src, const ML_Base *srcx) : ML_Mutable(), datastore(this,src.datastore,srcx)
{
    ML_Parall::resizetheML(0);

    resoffset = src.resoffset;

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

ML_Parall::~ML_Parall()
{
    return;
}

int ML_Parall::addLayer(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i <= numLayers() );

    int oldn = numLayers();

    theML.add(i);
    resoffset.add(i);

    MEMNEW(theML("&",i),ML_Mutable);
    resoffset("&",i) = ( i < oldn ) ? resoffset(i+1) : resoffset(zeroint());
    resoffset("&",i+1,1,oldn) += 1;

    return 1;
}

int ML_Parall::removeLayer(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < numLayers() );

    MEMDEL(theML("&",i));
    resoffset("&",i+1,1,resoffset.size()-1) -= 1;

    theML.remove(i);
    resoffset.remove(i);

    return 1;
}

int ML_Parall::setNumLayers(int n)
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

int ML_Parall::setActiveLayer(int i)
{
    NiceAssert( i >= 0 );
    NiceAssert( i < numLayers() );

    mlind = i;

    return 0;
}

int ML_Parall::setloffset(int i)
{
    if ( mlind == -1 )
    {
        throw("Parallel wrapper has no offset");
    }

    resoffset("&",mlind) = i;

    return 1;
}































































ParallBackcall::ParallBackcall(ML_Parall *xowner) : BLK_Nopnop()
{
    NiceAssert( xowner );

    owner = xowner;

    return;
}

ParallBackcall::ParallBackcall(ML_Parall *xowner, const ParallBackcall &src) : BLK_Nopnop()
{
    NiceAssert( xowner );

    owner = xowner;

    setaltx(NULL);

    assign(src,0);

    return;
}

ParallBackcall::ParallBackcall(ML_Parall *xowner, const ParallBackcall &src, const ML_Base *xsrcx) : BLK_Nopnop()
{
    NiceAssert( xowner );

    owner = xowner;

    setaltx(xsrcx);

    assign(src,0);

    return;
}

ParallBackcall::~ParallBackcall()
{
    return;
}

std::ostream &ParallBackcall::printstream(std::ostream &output) const
{
    output << "Parallel Network\n\n";

    output << "Module offsets: " << owner->resoffset << "\n";
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

std::istream &ParallBackcall::inputstream(std::istream &input )
{
    wait_dummy dummy;

    input >> dummy; input >> owner->resoffset;
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

int ParallBackcall::tspaceDim(void) const
{
    int i = ( owner->mlind == -1 ) ? numLayers()-1 : owner->mlind;

    return ((*((owner->theML)(i))).tspaceDim())+(owner->resoffset)(i);
}

int ParallBackcall::xspaceDim(void) const
{
    int i = ( owner->mlind == -1 ) ? numLayers()-1 : owner->mlind;

    return ((*((owner->theML)(i))).xspaceDim());
}

int ParallBackcall::isTrained(void) const
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

double ParallBackcall::calcDist(const gentype &ha, const gentype &hb, int ia, int db) const
{
    (void) ia;

    double res = 0;

    if ( db )
    {
        res = (double) norm(ha-hb);
    }

    return res;
}

double ParallBackcall::sparlvl(void) const
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

int ParallBackcall::randomise(double sparsity)
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

int ParallBackcall::autoen(void)
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

int ParallBackcall::renormalise(void)
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

int ParallBackcall::realign(void)
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

int ParallBackcall::scale(double a)
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

int ParallBackcall::reset(void)
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

int ParallBackcall::restart(void)
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

int ParallBackcall::home(void)
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

int ParallBackcall::train(int &res, svmvolatile int &killSwitch)
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

void ParallBackcall::eTrainingVector(gentype &res, int i) const
{
    gg(res,i);
    res -= y(i);

    return;
}

int ParallBackcall::gvTrainingVector(gentype &resv, int ii) const
{
    setzero((resv.force_vector()).resize(tspaceDim()));

    int res = 0;
    gentype tempv;

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).gv(tempv,BLK_Nopnop::getx(ii));

            (resv.dir_vector())("&",(owner->resoffset)(i)) = tempv;
        }
    }

    return res;
}

/*int ParallBackcall::gvTrainingVector(Matrix<gentype> &resv, int Nx, int dummy) const
{
    setzero((resv.force_vector()).resize(tspaceDim()));

    int res = 0;
    gentype tempv;

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)("&",i)))).gv(tempv,Nx,dummy);

            (resv.dir_vector())("&",(owner->resoffset)(i)) = tempv;
        }
    }

    return res;
}*/

int ParallBackcall::ghTrainingVector(gentype &resh, gentype &resg, int ii, int retaltg) const
{
    setzero((resh.force_vector()).resize(tspaceDim()));
    setzero((resg.force_vector()).resize(tspaceDim()));

    int res = 0;
    gentype temph,tempg;

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)(i)))).gh(temph,tempg,BLK_Nopnop::getx(ii),retaltg);

            if ( ((*((owner->theML)(i)))).hOutType() == 'V' )
            {
                (resh.dir_vector())("&",(owner->resoffset)(i),1,(owner->resoffset)(i)+((*((owner->theML)(i)))).tspaceDim()-1) = temph.cast_vector();
            }

            else
            {
                (resh.dir_vector())("&",(owner->resoffset)(i)) = temph;
            }

            if ( ((*((owner->theML)(i)))).gOutType() == 'V' )
            {
                (resg.dir_vector())("&",(owner->resoffset)(i),1,(owner->resoffset)(i)+((*((owner->theML)(i)))).tspaceDim()-1) = tempg.cast_vector();
            }

            else
            {
                (resg.dir_vector())("&",(owner->resoffset)(i)) = tempg;
            }
        }
    }

    return res;
}

int ParallBackcall::ggTrainingVector(Vector<double> &resg, int ii, int retaltg) const
{
    setzero(resg.resize(tspaceDim()));

    int res = 0;
    Vector<double> tempg;

    if ( numLayers() )
    {
        int i;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            res |= ((*((owner->theML)(i)))).gg(resg("&",(owner->resoffset)(i),1,(owner->resoffset)(i)+((*((owner->theML)(i)))).tspaceDim()-1),BLK_Nopnop::getx(ii),retaltg);
        }
    }

    return res;
}


void ParallBackcall::dgTrainingVector(SparseVector<gentype> &resx, int ii) const
{
    gentype temph,tempg;
    SparseVector<gentype> tempresx;

    // Set indexing for resx

    resx = BLK_Nopnop::getx(ii);

    // Resize all elements in resx appropriately

    if ( resx.nearindsize() )
    {
        int j;

        for ( j = 0 ; j < resx.nearindsize() ; j++ )
        {
            setzero(((resx.direref(j)).force_vector()).resize(tspaceDim()));
        }
    }

    if ( numLayers() )
    {
        int i,j;

        for ( i = 0 ; i < numLayers() ; i++ )
        {
            ((*((owner->theML)(i)))).dg(tempresx,BLK_Nopnop::y(ii),BLK_Nopnop::getx(ii));

            if ( resx.nearindsize() )
            {
                for ( j = 0 ; j < resx.nearindsize() ; j++ )
                {
                    if ( ((*((owner->theML)(i)))).gOutType() == 'V' )
                    {
                        ((resx.direref(j)).dir_vector())("&",(owner->resoffset)(i),1,(owner->resoffset)(i)+((*((owner->theML)(i)))).tspaceDim()-1) = ((tempresx.direcref(j)).cast_vector())(0,1,((*((owner->theML)(i)))).tspaceDim()-1);
                    }

                    else
                    {
                        ((resx.direref(j)).dir_vector())("&",(owner->resoffset)(i)) = tempresx.direcref(j);
                    }
                }
            }
        }
    }

    return;
}
