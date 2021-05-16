
//
// Mutable ML
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include "ml_mutable.h"
#include <iostream>
#include <sstream>
#include <string>

//
// xfer:    transfers data from src to dest, leaving src empty
// assign:  calls member assign function.  If *dest type does not match it
//          is deleted and a new one constructed prior to calling.  Note
//          that the address pointed to by dest is not changed.
//
// NB: for non-trivially different classes there will be some loss
//     of information in the data transfer.  For example, anomaly
//     classes may be not from SVM_MultiC, and class-wise weights
//     may be lost when moving from a classifier to a regressor.
//

ML_Base &xfer(ML_Base &dest, ML_Base &src);
//Assign defined above because function definition required at that point
//ML_Base &assign(ML_Base **dest, const ML_Base *src, int onlySemiCopy = 0);

void xferInfo(ML_Base &dest, const ML_Base &src);


ML_Mutable::ML_Mutable() : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    mlType = 1; // Default to binary SVM classifier
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = makeNewML(mlType);

    isfirst = 1;
    inftype = -1;

    setaltx(NULL);

    return;
}

ML_Mutable::ML_Mutable(const ML_Mutable &src) : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    mlType = 1; // Default to binary SVM classifier
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = makeNewML(mlType);

    isfirst = 1;
    inftype = -1;

    assign(src,0);
    setaltx(NULL);

    return;
}

ML_Mutable::ML_Mutable(const ML_Mutable &src, const ML_Base *xsrc) : ML_Base()
{
    thisthis = this;
    thisthisthis = &thisthis;

    mlType = 1; // Default to binary SVM classifier
    mlind  = 0;
    theML.resize(1);
    theML("&",mlind) = makeNewML(mlType);

    isfirst = 1;
    inftype = -1;

    assign(src,1);
    setaltx(xsrc);

    return;
}

ML_Mutable::~ML_Mutable()
{
    resizetheML(0);

    return;
}

void ML_Mutable::setMLTypeMorph(int newmlType)
{
    if ( mlType != newmlType )
    {
        // Assume a simple ML, must deal with general case elsewhere

        mlType = newmlType;
        mlind  = 0;

        resizetheML(1);

        ML_Base *src  = theML("&",mlind);
        ML_Base *dest = makeNewML(mlType);

        xfer(*dest,*src);

        MEMDEL(src);
    }

    return;
}

void ML_Mutable::setMLTypeClean(int newmlType)
{
    if ( mlType != newmlType )
    {
        // Assume a simple ML, must deal with general case elsewhere

        resizetheML(0);  // deletes everything
        theML.resize(1); // naive resize, don't allocate anything yet

        mlType = newmlType;
        mlind  = 0;

        theML("&",mlind) = makeNewML(mlType);
    }

    else
    {
        restart();
    }

    return;
}


std::istream &ML_Mutable::inputstream(std::istream &input)
{
    std::string keytype;

    // Mutate the type based on the first word in the stream (the removal
    // of which will not effect the underlying class), then let polymorphism
    // take care of the rest.

    input >> keytype;

    setMLTypeClean(convIDToType(keytype));

    return getML().inputstream(input);
}

void ML_Mutable::resizetheML(int newsize)
{
    NiceAssert( newsize >= 0 );

    int oldsize = theML.size();

    if ( newsize > oldsize )
    {
        theML.resize(newsize);

        int i;
        int newmlType = oldsize ? (*(theML(mlind))).type() : 1;

        for ( i = oldsize ; i < newsize ; i++ )
        {
            theML("&",i) = makeNewML(newmlType);
        }
    }

    else if ( newsize < oldsize )
    {
        int i;

        for ( i = newsize ; i < oldsize ; i++ )
        {
            MEMDEL(theML("&",i));
        }

        theML.resize(newsize);
    }

    return;
}





// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
//
// Begin helper functions
//
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// -----------------------------------------------------------------------

// IMPORTANT: if the machine is mutable then we are actually dealing
// with class ML_Mutable, although it will return type indicating
// the encased type.  Now, attempting to dynamic cast from ML_Mutable
// to SVM_Scalar (or whatever) will fail at runtime because you can't
// (apparently) safely cast to sibling cast.  To get around this, use
// the funciton getMLconst

int convIDToType(const std::string &keytype)
{
    int type = -1;

         if ( keytype == "SVM_Scalar" ) { type =   0; }
    else if ( keytype == "SVM_Binary" ) { type =   1; }
    else if ( keytype == "SVM_Single" ) { type =   2; }
    else if ( keytype == "SVM_MultiC" ) { type =   3; }
    else if ( keytype == "SVM_Vector" ) { type =   4; }
    else if ( keytype == "SVM_Anions" ) { type =   5; }
    else if ( keytype == "SVM_AutoEn" ) { type =   6; }
    else if ( keytype == "SVM_Densit" ) { type =   7; }
    else if ( keytype == "SVM_PFront" ) { type =   8; }
    else if ( keytype == "SVM_BiScor" ) { type =  12; }
    else if ( keytype == "SVM_ScScor" ) { type =  13; }
    else if ( keytype == "SVM_Gentyp" ) { type =  15; }
    else if ( keytype == "SVM_Planar" ) { type =  16; }
    else if ( keytype == "SVM_MvRank" ) { type =  17; }
    else if ( keytype == "SVM_MulBin" ) { type =  18; }
    else if ( keytype == "SVM_SimLrn" ) { type =  19; }
    else if ( keytype == "SVM_Cyclic" ) { type =  20; }

    else if ( keytype == "ONN_Scalar" ) { type = 100; }
    else if ( keytype == "ONN_Vector" ) { type = 101; }
    else if ( keytype == "ONN_Anions" ) { type = 102; }
    else if ( keytype == "ONN_Binary" ) { type = 103; }
    else if ( keytype == "ONN_AutoEn" ) { type = 104; }
    else if ( keytype == "ONN_Gentyp" ) { type = 105; }

    else if ( keytype == "BLK_Nopnop" ) { type = 200; }
    else if ( keytype == "BLK_Consen" ) { type = 201; }
    else if ( keytype == "BLK_AveSca" ) { type = 202; }
    else if ( keytype == "BLK_UsrFnA" ) { type = 203; }
    else if ( keytype == "BLK_UserIO" ) { type = 204; }
    else if ( keytype == "BLK_AveVec" ) { type = 205; }
    else if ( keytype == "BLK_AveAni" ) { type = 206; }
    else if ( keytype == "BLK_UsrFnB" ) { type = 207; }
    else if ( keytype == "BLK_CalBak" ) { type = 208; }
    else if ( keytype == "BLK_MexFnA" ) { type = 209; }
    else if ( keytype == "BLK_MexFnA" ) { type = 210; }
    else if ( keytype == "BLK_Mercer" ) { type = 211; }
    else if ( keytype == "BLK_Conect" ) { type = 212; }
    else if ( keytype == "BLK_System" ) { type = 213; }
    else if ( keytype == "BLK_Kernel" ) { type = 214; }
    else if ( keytype == "BLK_Bernst" ) { type = 215; }
    else if ( keytype == "BLK_Batter" ) { type = 216; }

    else if ( keytype == "KNN_Densit" ) { type = 300; }
    else if ( keytype == "KNN_Binary" ) { type = 301; }
    else if ( keytype == "KNN_Gentyp" ) { type = 302; }
    else if ( keytype == "KNN_Scalar" ) { type = 303; }
    else if ( keytype == "KNN_Vector" ) { type = 304; }
    else if ( keytype == "KNN_Anions" ) { type = 305; }
    else if ( keytype == "KNN_AutoEn" ) { type = 306; }
    else if ( keytype == "KNN_MultiC" ) { type = 307; }

    else if ( keytype == "GPR_Scalar" ) { type = 400; }
    else if ( keytype == "GPR_Vector" ) { type = 401; }
    else if ( keytype == "GPR_Anions" ) { type = 402; }
    else if ( keytype == "GPR_Gentyp" ) { type = 408; }
    else if ( keytype == "GPR_Binary" ) { type = 409; }

    else if ( keytype == "LSV_Scalar" ) { type = 500; }
    else if ( keytype == "LSV_Vector" ) { type = 501; }
    else if ( keytype == "LSV_Anions" ) { type = 502; }
    else if ( keytype == "LSV_ScScor" ) { type = 505; }
    else if ( keytype == "LSV_AutoEn" ) { type = 507; }
    else if ( keytype == "LSV_Gentyp" ) { type = 508; }
    else if ( keytype == "LSV_Planar" ) { type = 509; }
    else if ( keytype == "LSV_MvRank" ) { type = 510; }

    else if ( keytype == "IMP_Expect" ) { type = 600; }
    else if ( keytype == "IMP_ParSVM" ) { type = 601; }

    else if ( keytype == "SSV_Scalar" ) { type = 700; }
    else if ( keytype == "SSV_Binary" ) { type = 701; }
    else if ( keytype == "SSV_Single" ) { type = 702; }

    else if ( keytype == "MLM_Scalar" ) { type = 800; }
    else if ( keytype == "MLM_Binary" ) { type = 801; }
    else if ( keytype == "MLM_Vector" ) { type = 802; }

    else
    {
        throw("Error: unrecognised ID string");
    }

    return type;
}

int convTypeToID(std::string &res, int id)
{
    switch ( id )
    {
        case   0: { res = "SVM_Scalar"; break; }
        case   1: { res = "SVM_Binary"; break; }
        case   2: { res = "SVM_Single"; break; }
        case   3: { res = "SVM_MultiC"; break; }
        case   4: { res = "SVM_Vector"; break; }
        case   5: { res = "SVM_Anions"; break; }
        case   6: { res = "SVM_AutoEn"; break; }
        case   7: { res = "SVM_Densit"; break; }
        case   8: { res = "SVM_PFront"; break; }
        case  12: { res = "SVM_BiScor"; break; }
        case  13: { res = "SVM_ScScor"; break; }
        case  15: { res = "SVM_Gentyp"; break; }
        case  16: { res = "SVM_Planar"; break; }
        case  17: { res = "SVM_MvRank"; break; }
        case  18: { res = "SVM_MulBin"; break; }
        case  19: { res = "SVM_SimLrn"; break; }
        case  20: { res = "SVM_Cyclic"; break; }

        case 100: { res = "ONN_Scalar"; break; }
        case 101: { res = "ONN_Vector"; break; }
        case 102: { res = "ONN_Anions"; break; }
        case 103: { res = "ONN_Binary"; break; }
        case 104: { res = "ONN_AutoEn"; break; }
        case 105: { res = "ONN_Gentyp"; break; }

        case 200: { res = "BLK_Nopnop"; break; }
        case 201: { res = "BLK_Consen"; break; }
        case 202: { res = "BLK_AveSca"; break; }
        case 203: { res = "BLK_UsrFnA"; break; }
        case 204: { res = "BLK_UserIO"; break; }
        case 205: { res = "BLK_AveVec"; break; }
        case 206: { res = "BLK_AveAni"; break; }
        case 207: { res = "BLK_UsrFnB"; break; }
        case 208: { res = "BLK_CalBak"; break; }
        case 209: { res = "BLK_MexFnA"; break; }
        case 210: { res = "BLK_MexFnB"; break; }
        case 211: { res = "BLK_Mercer"; break; }
        case 212: { res = "BLK_Conect"; break; }
        case 213: { res = "BLK_System"; break; }
        case 214: { res = "BLK_Kernel"; break; }
        case 215: { res = "BLK_Bernst"; break; }
        case 216: { res = "BLK_Batter"; break; }

        case 300: { res = "KNN_Densit"; break; }
        case 301: { res = "KNN_Binary"; break; }
        case 302: { res = "KNN_Gentyp"; break; }
        case 303: { res = "KNN_Scalar"; break; }
        case 304: { res = "KNN_Vector"; break; }
        case 305: { res = "KNN_Anions"; break; }
        case 306: { res = "KNN_AutoEn"; break; }
        case 307: { res = "KNN_MultiC"; break; }

        case 400: { res = "GPR_Scalar"; break; }
        case 401: { res = "GPR_Vector"; break; }
        case 402: { res = "GPR_Anions"; break; }
        case 408: { res = "GPR_Gentyp"; break; }
        case 409: { res = "GPR_Binary"; break; }

        case 500: { res = "LSV_Scalar"; break; }
        case 501: { res = "LSV_Vector"; break; }
        case 502: { res = "LSV_Anions"; break; }
        case 505: { res = "LSV_ScScor"; break; }
        case 507: { res = "LSV_AutoEn"; break; }
        case 508: { res = "LSV_Gentyp"; break; }
        case 509: { res = "LSV_Planar"; break; }
        case 510: { res = "LSV_MvRank"; break; }

        case 600: { res = "IMP_Expect"; break; }
        case 601: { res = "IMP_ParSVM"; break; }

        case 700: { res = "SSV_Scalar"; break; }
        case 701: { res = "SSV_Binary"; break; }
        case 702: { res = "SSV_Single"; break; }

        case 800: { res = "MLM_Scalar"; break; }
        case 801: { res = "MLM_Binary"; break; }
        case 802: { res = "MLM_Vector"; break; }

        default:
        {
            throw("Error: unrecognised ID");
            break;
        }
    }

    return id;
}

ML_Base *makeNewML(int type, int subtype)
{
    ML_Base *res = NULL;

    switch ( type )
    {
        case   0: { MEMNEW(res,SVM_Scalar()); break; }
        case   1: { MEMNEW(res,SVM_Binary()); break; }
        case   2: { MEMNEW(res,SVM_Single()); break; }
        case   3: { MEMNEW(res,SVM_MultiC()); break; }
        case   4: { MEMNEW(res,SVM_Vector()); break; }
        case   5: { MEMNEW(res,SVM_Anions()); break; }
        case   6: { MEMNEW(res,SVM_AutoEn()); break; }
        case   7: { MEMNEW(res,SVM_Densit()); break; }
        case   8: { MEMNEW(res,SVM_PFront()); break; }
        case  12: { MEMNEW(res,SVM_BiScor()); break; }
        case  13: { MEMNEW(res,SVM_ScScor()); break; }
        case  15: { MEMNEW(res,SVM_Gentyp()); break; }
        case  16: { MEMNEW(res,SVM_Planar()); break; }
        case  17: { MEMNEW(res,SVM_MvRank()); break; }
        case  18: { MEMNEW(res,SVM_MulBin()); break; }
        case  19: { MEMNEW(res,SVM_SimLrn()); break; }
        case  20: { MEMNEW(res,SVM_Cyclic()); break; }

        case 100: { MEMNEW(res,ONN_Scalar()); break; }
        case 101: { MEMNEW(res,ONN_Vector()); break; }
        case 102: { MEMNEW(res,ONN_Anions()); break; }
        case 103: { MEMNEW(res,ONN_Binary()); break; }
        case 104: { MEMNEW(res,ONN_AutoEn()); break; }
        case 105: { MEMNEW(res,ONN_Gentyp()); break; }

        case 200: { MEMNEW(res,BLK_Nopnop()); break; }
        case 201: { MEMNEW(res,BLK_Consen()); break; }
        case 202: { MEMNEW(res,BLK_AveSca()); break; }
        case 203: { MEMNEW(res,BLK_UsrFnA()); break; }
        case 204: { MEMNEW(res,BLK_UserIO()); break; }
        case 205: { MEMNEW(res,BLK_AveVec()); break; }
        case 206: { MEMNEW(res,BLK_AveAni()); break; }
        case 207: { MEMNEW(res,BLK_UsrFnB()); break; }
        case 208: { MEMNEW(res,BLK_CalBak()); break; }
        case 209: { MEMNEW(res,BLK_MexFnA()); break; }
        case 210: { MEMNEW(res,BLK_MexFnB()); break; }
        case 211: { MEMNEW(res,BLK_Mercer()); break; }
        case 212: { MEMNEW(res,BLK_Conect()); break; }
        case 213: { MEMNEW(res,BLK_System()); break; }
        case 214: { MEMNEW(res,BLK_Kernel()); break; }
        case 215: { MEMNEW(res,BLK_Bernst()); break; }
        case 216: { MEMNEW(res,BLK_Batter()); break; }

        case 300: { MEMNEW(res,KNN_Densit()); break; }
        case 301: { MEMNEW(res,KNN_Binary()); break; }
        case 302: { MEMNEW(res,KNN_Gentyp()); break; }
        case 303: { MEMNEW(res,KNN_Scalar()); break; }
        case 304: { MEMNEW(res,KNN_Vector()); break; }
        case 305: { MEMNEW(res,KNN_Anions()); break; }
        case 306: { MEMNEW(res,KNN_AutoEn()); break; }
        case 307: { MEMNEW(res,KNN_MultiC()); break; }

        case 400: { MEMNEW(res,GPR_Scalar()); break; }
        case 401: { MEMNEW(res,GPR_Vector()); break; }
        case 402: { MEMNEW(res,GPR_Anions()); break; }
        case 408: { MEMNEW(res,GPR_Gentyp()); break; }
        case 409: { MEMNEW(res,GPR_Binary()); break; }

        case 500: { MEMNEW(res,LSV_Scalar()); break; }
        case 501: { MEMNEW(res,LSV_Vector()); break; }
        case 502: { MEMNEW(res,LSV_Anions()); break; }
        case 505: { MEMNEW(res,LSV_ScScor()); break; }
        case 507: { MEMNEW(res,LSV_AutoEn()); break; }
        case 508: { MEMNEW(res,LSV_Gentyp()); break; }
        case 509: { MEMNEW(res,LSV_Planar()); break; }
        case 510: { MEMNEW(res,LSV_MvRank()); break; }

        case 600: { MEMNEW(res,IMP_Expect()); break; }
        case 601: { MEMNEW(res,IMP_ParSVM()); break; }

        case 700: { MEMNEW(res,SSV_Scalar()); break; }
        case 701: { MEMNEW(res,SSV_Binary()); break; }
        case 702: { MEMNEW(res,SSV_Single()); break; }

        case 800: { MEMNEW(res,MLM_Scalar()); break; }
        case 801: { MEMNEW(res,MLM_Binary()); break; }
        case 802: { MEMNEW(res,MLM_Vector()); break; }

        default: { throw("Error: type unknown in makeNewML."); break; }
    }

    NiceAssert(res);

    if ( subtype != -42 )
    {
        res->setsubtype(subtype);
    }

    return res;
}

ML_Base &assign(ML_Base **dest, const ML_Base *src, int onlySemiCopy)
{
    if ( (*dest)->type() != src->type() )
    {
        MEMDEL(*dest);
        *dest = makeNewML(src->type());
    }

    switch ( src->type() )
    {
        case   0: { dynamic_cast<SVM_Scalar &>((**dest).getML()).assign(dynamic_cast<const SVM_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case   1: { dynamic_cast<SVM_Binary &>((**dest).getML()).assign(dynamic_cast<const SVM_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case   2: { dynamic_cast<SVM_Single &>((**dest).getML()).assign(dynamic_cast<const SVM_Single &>((*src).getMLconst()),onlySemiCopy); break; }
        case   3: { dynamic_cast<SVM_MultiC &>((**dest).getML()).assign(dynamic_cast<const SVM_MultiC &>((*src).getMLconst()),onlySemiCopy); break; }
        case   4: { dynamic_cast<SVM_Vector &>((**dest).getML()).assign(dynamic_cast<const SVM_Vector &>((*src).getMLconst()),onlySemiCopy); break; }
        case   5: { dynamic_cast<SVM_Anions &>((**dest).getML()).assign(dynamic_cast<const SVM_Anions &>((*src).getMLconst()),onlySemiCopy); break; }
        case   6: { dynamic_cast<SVM_AutoEn &>((**dest).getML()).assign(dynamic_cast<const SVM_AutoEn &>((*src).getMLconst()),onlySemiCopy); break; }
        case   7: { dynamic_cast<SVM_Densit &>((**dest).getML()).assign(dynamic_cast<const SVM_Densit &>((*src).getMLconst()),onlySemiCopy); break; }
        case   8: { dynamic_cast<SVM_PFront &>((**dest).getML()).assign(dynamic_cast<const SVM_PFront &>((*src).getMLconst()),onlySemiCopy); break; }
        case  12: { dynamic_cast<SVM_BiScor &>((**dest).getML()).assign(dynamic_cast<const SVM_BiScor &>((*src).getMLconst()),onlySemiCopy); break; }
        case  13: { dynamic_cast<SVM_ScScor &>((**dest).getML()).assign(dynamic_cast<const SVM_ScScor &>((*src).getMLconst()),onlySemiCopy); break; }
        case  15: { dynamic_cast<SVM_Gentyp &>((**dest).getML()).assign(dynamic_cast<const SVM_Gentyp &>((*src).getMLconst()),onlySemiCopy); break; }
        case  16: { dynamic_cast<SVM_Planar &>((**dest).getML()).assign(dynamic_cast<const SVM_Planar &>((*src).getMLconst()),onlySemiCopy); break; }
        case  17: { dynamic_cast<SVM_MvRank &>((**dest).getML()).assign(dynamic_cast<const SVM_MvRank &>((*src).getMLconst()),onlySemiCopy); break; }
        case  18: { dynamic_cast<SVM_MulBin &>((**dest).getML()).assign(dynamic_cast<const SVM_MulBin &>((*src).getMLconst()),onlySemiCopy); break; }
        case  19: { dynamic_cast<SVM_SimLrn &>((**dest).getML()).assign(dynamic_cast<const SVM_SimLrn &>((*src).getMLconst()),onlySemiCopy); break; }
        case  20: { dynamic_cast<SVM_Cyclic &>((**dest).getML()).assign(dynamic_cast<const SVM_Cyclic &>((*src).getMLconst()),onlySemiCopy); break; }

        case 100: { dynamic_cast<ONN_Scalar &>((**dest).getML()).assign(dynamic_cast<const ONN_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 101: { dynamic_cast<ONN_Vector &>((**dest).getML()).assign(dynamic_cast<const ONN_Vector &>((*src).getMLconst()),onlySemiCopy); break; }
        case 102: { dynamic_cast<ONN_Anions &>((**dest).getML()).assign(dynamic_cast<const ONN_Anions &>((*src).getMLconst()),onlySemiCopy); break; }
        case 103: { dynamic_cast<ONN_Binary &>((**dest).getML()).assign(dynamic_cast<const ONN_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case 104: { dynamic_cast<ONN_AutoEn &>((**dest).getML()).assign(dynamic_cast<const ONN_AutoEn &>((*src).getMLconst()),onlySemiCopy); break; }
        case 105: { dynamic_cast<ONN_Gentyp &>((**dest).getML()).assign(dynamic_cast<const ONN_Gentyp &>((*src).getMLconst()),onlySemiCopy); break; }

        case 200: { dynamic_cast<BLK_Nopnop &>((**dest).getML()).assign(dynamic_cast<const BLK_Nopnop &>((*src).getMLconst()),onlySemiCopy); break; }
        case 201: { dynamic_cast<BLK_Consen &>((**dest).getML()).assign(dynamic_cast<const BLK_Consen &>((*src).getMLconst()),onlySemiCopy); break; }
        case 202: { dynamic_cast<BLK_AveSca &>((**dest).getML()).assign(dynamic_cast<const BLK_AveSca &>((*src).getMLconst()),onlySemiCopy); break; }
        case 203: { dynamic_cast<BLK_UsrFnA &>((**dest).getML()).assign(dynamic_cast<const BLK_UsrFnA &>((*src).getMLconst()),onlySemiCopy); break; }
        case 204: { dynamic_cast<BLK_UserIO &>((**dest).getML()).assign(dynamic_cast<const BLK_UserIO &>((*src).getMLconst()),onlySemiCopy); break; }
        case 205: { dynamic_cast<BLK_AveVec &>((**dest).getML()).assign(dynamic_cast<const BLK_AveVec &>((*src).getMLconst()),onlySemiCopy); break; }
        case 206: { dynamic_cast<BLK_AveAni &>((**dest).getML()).assign(dynamic_cast<const BLK_AveAni &>((*src).getMLconst()),onlySemiCopy); break; }
        case 207: { dynamic_cast<BLK_UsrFnB &>((**dest).getML()).assign(dynamic_cast<const BLK_UsrFnB &>((*src).getMLconst()),onlySemiCopy); break; }
        case 208: { dynamic_cast<BLK_CalBak &>((**dest).getML()).assign(dynamic_cast<const BLK_CalBak &>((*src).getMLconst()),onlySemiCopy); break; }
        case 209: { dynamic_cast<BLK_MexFnA &>((**dest).getML()).assign(dynamic_cast<const BLK_MexFnA &>((*src).getMLconst()),onlySemiCopy); break; }
        case 210: { dynamic_cast<BLK_MexFnB &>((**dest).getML()).assign(dynamic_cast<const BLK_MexFnB &>((*src).getMLconst()),onlySemiCopy); break; }
        case 211: { dynamic_cast<BLK_Mercer &>((**dest).getML()).assign(dynamic_cast<const BLK_Mercer &>((*src).getMLconst()),onlySemiCopy); break; }
        case 212: { dynamic_cast<BLK_Conect &>((**dest).getML()).assign(dynamic_cast<const BLK_Conect &>((*src).getMLconst()),onlySemiCopy); break; }
        case 213: { dynamic_cast<BLK_System &>((**dest).getML()).assign(dynamic_cast<const BLK_System &>((*src).getMLconst()),onlySemiCopy); break; }
        case 214: { dynamic_cast<BLK_Kernel &>((**dest).getML()).assign(dynamic_cast<const BLK_Kernel &>((*src).getMLconst()),onlySemiCopy); break; }
        case 215: { dynamic_cast<BLK_Bernst &>((**dest).getML()).assign(dynamic_cast<const BLK_Bernst &>((*src).getMLconst()),onlySemiCopy); break; }
        case 216: { dynamic_cast<BLK_Batter &>((**dest).getML()).assign(dynamic_cast<const BLK_Batter &>((*src).getMLconst()),onlySemiCopy); break; }

        case 300: { dynamic_cast<KNN_Densit &>((**dest).getML()).assign(dynamic_cast<const KNN_Densit &>((*src).getMLconst()),onlySemiCopy); break; }
        case 301: { dynamic_cast<KNN_Binary &>((**dest).getML()).assign(dynamic_cast<const KNN_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case 302: { dynamic_cast<KNN_Gentyp &>((**dest).getML()).assign(dynamic_cast<const KNN_Gentyp &>((*src).getMLconst()),onlySemiCopy); break; }
        case 303: { dynamic_cast<KNN_Scalar &>((**dest).getML()).assign(dynamic_cast<const KNN_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 304: { dynamic_cast<KNN_Vector &>((**dest).getML()).assign(dynamic_cast<const KNN_Vector &>((*src).getMLconst()),onlySemiCopy); break; }
        case 305: { dynamic_cast<KNN_Anions &>((**dest).getML()).assign(dynamic_cast<const KNN_Anions &>((*src).getMLconst()),onlySemiCopy); break; }
        case 306: { dynamic_cast<KNN_AutoEn &>((**dest).getML()).assign(dynamic_cast<const KNN_AutoEn &>((*src).getMLconst()),onlySemiCopy); break; }
        case 307: { dynamic_cast<KNN_MultiC &>((**dest).getML()).assign(dynamic_cast<const KNN_MultiC &>((*src).getMLconst()),onlySemiCopy); break; }

        case 400: { dynamic_cast<GPR_Scalar &>((**dest).getML()).assign(dynamic_cast<const GPR_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 401: { dynamic_cast<GPR_Vector &>((**dest).getML()).assign(dynamic_cast<const GPR_Vector &>((*src).getMLconst()),onlySemiCopy); break; }
        case 402: { dynamic_cast<GPR_Anions &>((**dest).getML()).assign(dynamic_cast<const GPR_Anions &>((*src).getMLconst()),onlySemiCopy); break; }
        case 408: { dynamic_cast<GPR_Gentyp &>((**dest).getML()).assign(dynamic_cast<const GPR_Gentyp &>((*src).getMLconst()),onlySemiCopy); break; }
        case 409: { dynamic_cast<GPR_Binary &>((**dest).getML()).assign(dynamic_cast<const GPR_Binary &>((*src).getMLconst()),onlySemiCopy); break; }

        case 500: { dynamic_cast<LSV_Scalar &>((**dest).getML()).assign(dynamic_cast<const LSV_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 501: { dynamic_cast<LSV_Vector &>((**dest).getML()).assign(dynamic_cast<const LSV_Vector &>((*src).getMLconst()),onlySemiCopy); break; }
        case 502: { dynamic_cast<LSV_Anions &>((**dest).getML()).assign(dynamic_cast<const LSV_Anions &>((*src).getMLconst()),onlySemiCopy); break; }
        case 505: { dynamic_cast<LSV_ScScor &>((**dest).getML()).assign(dynamic_cast<const LSV_ScScor &>((*src).getMLconst()),onlySemiCopy); break; }
        case 507: { dynamic_cast<LSV_AutoEn &>((**dest).getML()).assign(dynamic_cast<const LSV_AutoEn &>((*src).getMLconst()),onlySemiCopy); break; }
        case 508: { dynamic_cast<LSV_Gentyp &>((**dest).getML()).assign(dynamic_cast<const LSV_Gentyp &>((*src).getMLconst()),onlySemiCopy); break; }
        case 509: { dynamic_cast<LSV_Planar &>((**dest).getML()).assign(dynamic_cast<const LSV_Planar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 510: { dynamic_cast<LSV_MvRank &>((**dest).getML()).assign(dynamic_cast<const LSV_MvRank &>((*src).getMLconst()),onlySemiCopy); break; }

        case 600: { dynamic_cast<IMP_Expect &>((**dest).getML()).assign(dynamic_cast<const IMP_Expect &>((*src).getMLconst()),onlySemiCopy); break; }
        case 601: { dynamic_cast<IMP_ParSVM &>((**dest).getML()).assign(dynamic_cast<const IMP_ParSVM &>((*src).getMLconst()),onlySemiCopy); break; }

        case 700: { dynamic_cast<SSV_Scalar &>((**dest).getML()).assign(dynamic_cast<const SSV_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 701: { dynamic_cast<SSV_Binary &>((**dest).getML()).assign(dynamic_cast<const SSV_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case 702: { dynamic_cast<SSV_Single &>((**dest).getML()).assign(dynamic_cast<const SSV_Single &>((*src).getMLconst()),onlySemiCopy); break; }

        case 800: { dynamic_cast<MLM_Scalar &>((**dest).getML()).assign(dynamic_cast<const MLM_Scalar &>((*src).getMLconst()),onlySemiCopy); break; }
        case 801: { dynamic_cast<MLM_Binary &>((**dest).getML()).assign(dynamic_cast<const MLM_Binary &>((*src).getMLconst()),onlySemiCopy); break; }
        case 802: { dynamic_cast<MLM_Vector &>((**dest).getML()).assign(dynamic_cast<const MLM_Vector &>((*src).getMLconst()),onlySemiCopy); break; }

        default:
        {
            throw("Error: unknown type error in assignxfer assign");

            break;
        }
    }

    return **dest;
}

// Notes on xfer functions:
//
// - by using qaddtrainingvector, we remove the actual data from the source
//   and place it into the dest, leaving a placeholder.  As the added vector
//   is by default set zero, and the Gp matrix is a callback cache, the memory
//   load from all this adding is minimal.  We then simply resize the src to
//   zero to remove all the data.

ML_Base &xfer(ML_Base &dest, ML_Base &src)
{
    if ( dest.type() == src.type() )
    {
        switch ( dest.type() )
        {
            case   0: { dynamic_cast<SVM_Scalar &>(dest) = dynamic_cast<const SVM_Scalar &>(src); break; }
            case   1: { dynamic_cast<SVM_Binary &>(dest) = dynamic_cast<const SVM_Binary &>(src); break; }
            case   2: { dynamic_cast<SVM_Single &>(dest) = dynamic_cast<const SVM_Single &>(src); break; }
            case   3: { dynamic_cast<SVM_MultiC &>(dest) = dynamic_cast<const SVM_MultiC &>(src); break; }
            case   4: { dynamic_cast<SVM_Vector &>(dest) = dynamic_cast<const SVM_Vector &>(src); break; }
            case   5: { dynamic_cast<SVM_Anions &>(dest) = dynamic_cast<const SVM_Anions &>(src); break; }
            case   6: { dynamic_cast<SVM_AutoEn &>(dest) = dynamic_cast<const SVM_AutoEn &>(src); break; }
            case   7: { dynamic_cast<SVM_Densit &>(dest) = dynamic_cast<const SVM_Densit &>(src); break; }
            case   8: { dynamic_cast<SVM_PFront &>(dest) = dynamic_cast<const SVM_PFront &>(src); break; }
            case  12: { dynamic_cast<SVM_BiScor &>(dest) = dynamic_cast<const SVM_BiScor &>(src); break; }
            case  13: { dynamic_cast<SVM_ScScor &>(dest) = dynamic_cast<const SVM_ScScor &>(src); break; }
            case  15: { dynamic_cast<SVM_Gentyp &>(dest) = dynamic_cast<const SVM_Gentyp &>(src); break; }
            case  16: { dynamic_cast<SVM_Planar &>(dest) = dynamic_cast<const SVM_Planar &>(src); break; }
            case  17: { dynamic_cast<SVM_MvRank &>(dest) = dynamic_cast<const SVM_MvRank &>(src); break; }
            case  18: { dynamic_cast<SVM_MulBin &>(dest) = dynamic_cast<const SVM_MulBin &>(src); break; }
            case  19: { dynamic_cast<SVM_SimLrn &>(dest) = dynamic_cast<const SVM_SimLrn &>(src); break; }
            case  20: { dynamic_cast<SVM_Cyclic &>(dest) = dynamic_cast<const SVM_Cyclic &>(src); break; }

            case 100: { dynamic_cast<ONN_Scalar &>(dest) = dynamic_cast<const ONN_Scalar &>(src); break; }
            case 101: { dynamic_cast<ONN_Vector &>(dest) = dynamic_cast<const ONN_Vector &>(src); break; }
            case 102: { dynamic_cast<ONN_Anions &>(dest) = dynamic_cast<const ONN_Anions &>(src); break; }
            case 103: { dynamic_cast<ONN_Binary &>(dest) = dynamic_cast<const ONN_Binary &>(src); break; }
            case 104: { dynamic_cast<ONN_AutoEn &>(dest) = dynamic_cast<const ONN_AutoEn &>(src); break; }
            case 105: { dynamic_cast<ONN_Gentyp &>(dest) = dynamic_cast<const ONN_Gentyp &>(src); break; }

            case 200: { dynamic_cast<BLK_Nopnop &>(dest) = dynamic_cast<const BLK_Nopnop &>(src); break; }
            case 201: { dynamic_cast<BLK_Consen &>(dest) = dynamic_cast<const BLK_Consen &>(src); break; }
            case 202: { dynamic_cast<BLK_AveSca &>(dest) = dynamic_cast<const BLK_AveSca &>(src); break; }
            case 203: { dynamic_cast<BLK_UsrFnA &>(dest) = dynamic_cast<const BLK_UsrFnA &>(src); break; }
            case 204: { dynamic_cast<BLK_UserIO &>(dest) = dynamic_cast<const BLK_UserIO &>(src); break; }
            case 205: { dynamic_cast<BLK_AveVec &>(dest) = dynamic_cast<const BLK_AveVec &>(src); break; }
            case 206: { dynamic_cast<BLK_AveAni &>(dest) = dynamic_cast<const BLK_AveAni &>(src); break; }
            case 207: { dynamic_cast<BLK_UsrFnB &>(dest) = dynamic_cast<const BLK_UsrFnB &>(src); break; }
            case 208: { dynamic_cast<BLK_CalBak &>(dest) = dynamic_cast<const BLK_CalBak &>(src); break; }
            case 209: { dynamic_cast<BLK_MexFnA &>(dest) = dynamic_cast<const BLK_MexFnA &>(src); break; }
            case 210: { dynamic_cast<BLK_MexFnB &>(dest) = dynamic_cast<const BLK_MexFnB &>(src); break; }
            case 211: { dynamic_cast<BLK_Mercer &>(dest) = dynamic_cast<const BLK_Mercer &>(src); break; }
            case 212: { dynamic_cast<BLK_Conect &>(dest) = dynamic_cast<const BLK_Conect &>(src); break; }
            case 213: { dynamic_cast<BLK_System &>(dest) = dynamic_cast<const BLK_System &>(src); break; }
            case 214: { dynamic_cast<BLK_Kernel &>(dest) = dynamic_cast<const BLK_Kernel &>(src); break; }
            case 215: { dynamic_cast<BLK_Bernst &>(dest) = dynamic_cast<const BLK_Bernst &>(src); break; }
            case 216: { dynamic_cast<BLK_Batter &>(dest) = dynamic_cast<const BLK_Batter &>(src); break; }

            case 300: { dynamic_cast<KNN_Densit &>(dest) = dynamic_cast<const KNN_Densit &>(src); break; }
            case 301: { dynamic_cast<KNN_Binary &>(dest) = dynamic_cast<const KNN_Binary &>(src); break; }
            case 302: { dynamic_cast<KNN_Gentyp &>(dest) = dynamic_cast<const KNN_Gentyp &>(src); break; }
            case 303: { dynamic_cast<KNN_Scalar &>(dest) = dynamic_cast<const KNN_Scalar &>(src); break; }
            case 304: { dynamic_cast<KNN_Vector &>(dest) = dynamic_cast<const KNN_Vector &>(src); break; }
            case 305: { dynamic_cast<KNN_Anions &>(dest) = dynamic_cast<const KNN_Anions &>(src); break; }
            case 306: { dynamic_cast<KNN_AutoEn &>(dest) = dynamic_cast<const KNN_AutoEn &>(src); break; }
            case 307: { dynamic_cast<KNN_MultiC &>(dest) = dynamic_cast<const KNN_MultiC &>(src); break; }

            case 400: { dynamic_cast<GPR_Scalar &>(dest) = dynamic_cast<const GPR_Scalar &>(src); break; }
            case 401: { dynamic_cast<GPR_Vector &>(dest) = dynamic_cast<const GPR_Vector &>(src); break; }
            case 402: { dynamic_cast<GPR_Anions &>(dest) = dynamic_cast<const GPR_Anions &>(src); break; }
            case 408: { dynamic_cast<GPR_Gentyp &>(dest) = dynamic_cast<const GPR_Gentyp &>(src); break; }
            case 409: { dynamic_cast<GPR_Binary &>(dest) = dynamic_cast<const GPR_Binary &>(src); break; }

            case 500: { dynamic_cast<LSV_Scalar &>(dest) = dynamic_cast<const LSV_Scalar &>(src); break; }
            case 501: { dynamic_cast<LSV_Vector &>(dest) = dynamic_cast<const LSV_Vector &>(src); break; }
            case 502: { dynamic_cast<LSV_Anions &>(dest) = dynamic_cast<const LSV_Anions &>(src); break; }
            case 505: { dynamic_cast<LSV_ScScor &>(dest) = dynamic_cast<const LSV_ScScor &>(src); break; }
            case 507: { dynamic_cast<LSV_AutoEn &>(dest) = dynamic_cast<const LSV_AutoEn &>(src); break; }
            case 508: { dynamic_cast<LSV_Gentyp &>(dest) = dynamic_cast<const LSV_Gentyp &>(src); break; }
            case 509: { dynamic_cast<LSV_Planar &>(dest) = dynamic_cast<const LSV_Planar &>(src); break; }
            case 510: { dynamic_cast<LSV_MvRank &>(dest) = dynamic_cast<const LSV_MvRank &>(src); break; }

            case 600: { dynamic_cast<IMP_Expect &>(dest) = dynamic_cast<const IMP_Expect &>(src); break; }
            case 601: { dynamic_cast<IMP_ParSVM &>(dest) = dynamic_cast<const IMP_ParSVM &>(src); break; }

            case 700: { dynamic_cast<SSV_Scalar &>(dest) = dynamic_cast<const SSV_Scalar &>(src); break; }
            case 701: { dynamic_cast<SSV_Binary &>(dest) = dynamic_cast<const SSV_Binary &>(src); break; }
            case 702: { dynamic_cast<SSV_Single &>(dest) = dynamic_cast<const SSV_Single &>(src); break; }

            case 800: { dynamic_cast<MLM_Scalar &>(dest) = dynamic_cast<const MLM_Scalar &>(src); break; }
            case 801: { dynamic_cast<MLM_Binary &>(dest) = dynamic_cast<const MLM_Binary &>(src); break; }
            case 802: { dynamic_cast<MLM_Vector &>(dest) = dynamic_cast<const MLM_Vector &>(src); break; }

            default:
            {
                throw("Error: Unknown source/destination type in ML xfer.");

                break;
            }
        }
    }

    else if ( isSVMBinary(dest) && isSVMSingle(src) )
    {
        dynamic_cast<SVM_Binary &>(dest) = dynamic_cast<const SVM_Single &>(src);
    }

    else
    {
        xferInfo(dest,src);

        int i,j;
        int d;
        double Cweight;
        double epsweight;
        gentype y;
        SparseVector<gentype> x;

        for ( i = (src.N())-1 ; i >= 0 ; i-- )
        {
            j = dest.N();

            d         = src.isenabled(i);
            Cweight   = (src.Cweight())(i);
            Cweight   = (src.Cweight())(i);
            epsweight = (src.epsweight())(i);

            src.removeTrainingVector(i,y,x);
            dest.qaddTrainingVector(j,y,x,Cweight,epsweight);

            if ( !d )
            {
                dest.disable(j);
            }
        }

        if ( ( isSVMScalar(dest) || isSVMBinary(dest) || isSVMSingle(dest) ) && isSVM(src) )
        {
            (dynamic_cast<SVM_Generic &>(dest)).setCweightfuzz((dynamic_cast<const SVM_Generic &>(src)).Cweightfuzz());
        }
    }

    src.restart();

    return dest;
}

void xferInfo(ML_Base &mldest, const ML_Base &mlsrc)
{
    // Clear the destination

    mldest.restart();
    mldest.setmemsize(mlsrc.memsize());

    // Transfer variables for *all* types of ML

    mldest.setC(mlsrc.C());
    mldest.seteps(mlsrc.eps());

    mldest.setzerotol(mlsrc.zerotol());
    mldest.setOpttol(mlsrc.Opttol());
    mldest.setmaxitcnt(mlsrc.maxitcnt());
    mldest.setmaxtraintime(mlsrc.maxtraintime());

    mldest.setKernel(mlsrc.getKernel());

    if ( isONN(mlsrc) && !(isONN(mldest)) )
    {
        mldest.getKernel_unsafe().setLeftNormal();
        mldest.resetKernel();
    }

    if ( !(isONN(mlsrc)) && isONN(mldest) )
    {
        mldest.getKernel_unsafe().setLeftPlain();
        mldest.resetKernel();
    }

    // Type specifics follow

    if ( isSVMScalar(mldest) && isSVM(mlsrc) )
    {
              SVM_Scalar  &dest = dynamic_cast<      SVM_Scalar  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.set1NormCost();     }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMBinary(mldest) && isSVM(mlsrc) )
    {
              SVM_Binary  &dest = dynamic_cast<      SVM_Binary  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMSingle(mldest) && isSVM(mlsrc) )
    {
              SVM_Single  &dest = dynamic_cast<      SVM_Single  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMMultiC(mldest) && isSVM(mlsrc) )
    {
              SVM_MultiC  &dest = dynamic_cast<      SVM_MultiC  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());

        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMVector(mldest) && isSVM(mlsrc) )
    {
              SVM_Vector  &dest = dynamic_cast<      SVM_Vector  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());

        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
    }

    else if ( isSVMAnions(mldest) && isSVM(mlsrc) )
    {
              SVM_Anions  &dest = dynamic_cast<      SVM_Anions  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());

        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
    }

    else if ( isSVMAutoEn(mldest) && isSVM(mlsrc) )
    {
              SVM_AutoEn  &dest = dynamic_cast<      SVM_AutoEn  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setouterlr (src.outerlr());
        dest.setoutertol(src.outertol());

        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
    }

    else if ( isSVMDensit(mldest) && isSVM(mlsrc) )
    {
              SVM_Densit  &dest = dynamic_cast<      SVM_Densit  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMPFront(mldest) && isSVM(mlsrc) )
    {
              SVM_Single  &dest = dynamic_cast<      SVM_Single  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMBiScor(mldest) && isSVM(mlsrc) )
    {
              SVM_BiScor  &dest = dynamic_cast<      SVM_BiScor  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMScScor(mldest) && isSVM(mlsrc) )
    {
              SVM_ScScor  &dest = dynamic_cast<      SVM_ScScor  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        dest.setm(src.m());
        dest.setouterlr(src.outerlr());
        dest.setoutertol(src.outertol());
        dest.setmaxiterfuzzt(src.maxiterfuzzt());
        dest.setusefuzzt(src.usefuzzt());
        dest.setlrfuzzt(src.lrfuzzt());
        dest.setztfuzzt(src.ztfuzzt());
        dest.setcostfnfuzzt(src.costfnfuzzt());
        dest.setLinBiasForce(src.LinBiasForce());
        dest.setQuadBiasForce(src.QuadBiasForce());
        dest.setnu(src.nu());
        dest.setnuQuad(src.nuQuad());

        if ( src.isOptActive()     ) { dest.setOptActive();     }
        if ( src.isOptSMO()        ) { dest.setOptSMO();        }
        if ( src.isOptD2C()        ) { dest.setOptD2C();        }
        if ( src.isLinearCost()    ) { dest.setLinearCost();    }
        if ( src.isQuadraticCost() ) { dest.setQuadraticCost(); }
        if ( src.is1NormCost()     ) { dest.is1NormCost();      }
        if ( src.isFixedTube()     ) { dest.setFixedTube();     }
        if ( src.isShrinkTube()    ) { dest.setShrinkTube();    }

        if ( src.isautosetCscaled()      ) { dest.autosetCscaled(src.autosetCval());                         }
        if ( src.isautosetCKmean()       ) { dest.autosetCKmean();                                           }
        if ( src.isautosetCKmedian()     ) { dest.autosetCKmedian();                                         }
        if ( src.isautosetCNKmean()      ) { dest.autosetCNKmean();                                          }
        if ( src.isautosetCNKmedian()    ) { dest.autosetCNKmedian();                                        }
        if ( src.isautosetLinBiasForce() ) { dest.autosetLinBiasForce(src.autosetnuval(),src.autosetCval()); }
    }

    else if ( isSVMGentyp(mldest) && isSVM(mlsrc) )
    {
              SVM_Gentyp  &dest = dynamic_cast<      SVM_Gentyp  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMPlanar(mldest) && isSVM(mlsrc) )
    {
              SVM_Planar  &dest = dynamic_cast<      SVM_Planar  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMMvRank(mldest) && isSVM(mlsrc) )
    {
              SVM_MvRank  &dest = dynamic_cast<      SVM_MvRank  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMCyclic(mldest) && isSVM(mlsrc) )
    {
              SVM_Cyclic  &dest = dynamic_cast<      SVM_Cyclic  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMMulBin(mldest) && isSVM(mlsrc) )
    {
              SVM_MulBin  &dest = dynamic_cast<      SVM_MulBin  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSVMSimLrn(mldest) && isSVM(mlsrc) )
    {
              SVM_SimLrn  &dest = dynamic_cast<      SVM_SimLrn  &>(mldest);
        const SVM_Generic &src  = dynamic_cast<const SVM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isONNScalar(mldest) && isONN(mlsrc) )
    {
              ONN_Scalar  &dest = dynamic_cast<      ONN_Scalar  &>(mldest);
        const ONN_Generic &src  = dynamic_cast<const ONN_Generic &>(mlsrc );

        dest.setlr(src.lr());
    }

    else if ( isONNVector(mldest) && isONN(mlsrc) )
    {
              ONN_Vector  &dest = dynamic_cast<      ONN_Vector  &>(mldest);
        const ONN_Generic &src  = dynamic_cast<const ONN_Generic &>(mlsrc );

        dest.setlr(src.lr());
    }

    else if ( isONNAnions(mldest) && isONN(mlsrc) )
    {
              ONN_Anions  &dest = dynamic_cast<      ONN_Anions  &>(mldest);
        const ONN_Generic &src  = dynamic_cast<const ONN_Generic &>(mlsrc );

        dest.setlr(src.lr());
    }

    else if ( isONNBinary(mldest) && isONN(mlsrc) )
    {
              ONN_Binary  &dest = dynamic_cast<      ONN_Binary  &>(mldest);
        const ONN_Generic &src  = dynamic_cast<const ONN_Generic &>(mlsrc );

        dest.setlr(src.lr());
    }

    else if ( isONNAutoEn(mldest) && isONN(mlsrc) )
    {
              ONN_AutoEn  &dest = dynamic_cast<      ONN_AutoEn  &>(mldest);
        const ONN_Generic &src  = dynamic_cast<const ONN_Generic &>(mlsrc );

        dest.setlr(src.lr());
    }

    else if ( isONNGentyp(mldest) && isONN(mlsrc) )
    {
              ONN_Gentyp  &dest = dynamic_cast<      ONN_Gentyp  &>(mldest);
        const ONN_Generic &src  = dynamic_cast<const ONN_Generic &>(mlsrc );

        dest.setlr(src.lr());
    }

    else if ( isBLKNopnop(mldest) && isBLK(mlsrc) )
    {
              BLK_Nopnop  &dest = dynamic_cast<      BLK_Nopnop  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKConsen(mldest) && isBLK(mlsrc) )
    {
              BLK_Consen  &dest = dynamic_cast<      BLK_Consen  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKAveSca(mldest) && isBLK(mlsrc) )
    {
              BLK_AveSca  &dest = dynamic_cast<      BLK_AveSca  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKUsrFnA(mldest) && isBLK(mlsrc) )
    {
              BLK_UsrFnA  &dest = dynamic_cast<      BLK_UsrFnA  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKUserIO(mldest) && isBLK(mlsrc) )
    {
              BLK_UserIO  &dest = dynamic_cast<      BLK_UserIO  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKAveVec(mldest) && isBLK(mlsrc) )
    {
              BLK_AveVec  &dest = dynamic_cast<      BLK_AveVec  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKAveAni(mldest) && isBLK(mlsrc) )
    {
              BLK_AveAni  &dest = dynamic_cast<      BLK_AveAni  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKUsrFnB(mldest) && isBLK(mlsrc) )
    {
              BLK_UsrFnB  &dest = dynamic_cast<      BLK_UsrFnB  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKCalBak(mldest) && isBLK(mlsrc) )
    {
              BLK_CalBak  &dest = dynamic_cast<      BLK_CalBak  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKMexFnA(mldest) && isBLK(mlsrc) )
    {
              BLK_MexFnA  &dest = dynamic_cast<      BLK_MexFnA  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKMexFnB(mldest) && isBLK(mlsrc) )
    {
              BLK_MexFnB  &dest = dynamic_cast<      BLK_MexFnB  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKMercer(mldest) && isBLK(mlsrc) )
    {
              BLK_Mercer  &dest = dynamic_cast<      BLK_Mercer  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKConect(mldest) && isBLK(mlsrc) )
    {
              BLK_Conect  &dest = dynamic_cast<      BLK_Conect  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKSystem(mldest) && isBLK(mlsrc) )
    {
              BLK_System  &dest = dynamic_cast<      BLK_System  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKKernel(mldest) && isBLK(mlsrc) )
    {
              BLK_Kernel  &dest = dynamic_cast<      BLK_Kernel  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKBernst(mldest) && isBLK(mlsrc) )
    {
              BLK_Bernst  &dest = dynamic_cast<      BLK_Bernst  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isBLKBatter(mldest) && isBLK(mlsrc) )
    {
              BLK_Batter  &dest = dynamic_cast<      BLK_Batter  &>(mldest);
        const BLK_Generic &src  = dynamic_cast<const BLK_Generic &>(mlsrc );

        dest.setoutfn(src.outfn());
    }

    else if ( isKNNDensit(mldest) && isKNN(mlsrc) )
    {
              KNN_Densit  &dest = dynamic_cast<      KNN_Densit  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNBinary(mldest) && isKNN(mlsrc) )
    {
              KNN_Binary  &dest = dynamic_cast<      KNN_Binary  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNGentyp(mldest) && isKNN(mlsrc) )
    {
              KNN_Gentyp  &dest = dynamic_cast<      KNN_Gentyp  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNScalar(mldest) && isKNN(mlsrc) )
    {
              KNN_Scalar  &dest = dynamic_cast<      KNN_Scalar  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNVector(mldest) && isKNN(mlsrc) )
    {
              KNN_Vector  &dest = dynamic_cast<      KNN_Vector  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNAnions(mldest) && isKNN(mlsrc) )
    {
              KNN_Anions  &dest = dynamic_cast<      KNN_Anions  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNAutoEn(mldest) && isKNN(mlsrc) )
    {
              KNN_AutoEn  &dest = dynamic_cast<      KNN_AutoEn  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isKNNMultiC(mldest) && isKNN(mlsrc) )
    {
              KNN_MultiC  &dest = dynamic_cast<      KNN_MultiC  &>(mldest);
        const KNN_Generic &src  = dynamic_cast<const KNN_Generic &>(mlsrc );

        dest.setk(src.k());
    }

    else if ( isGPRScalar(mldest) && isGPR(mlsrc) )
    {
              GPR_Scalar  &dest = dynamic_cast<      GPR_Scalar  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRVector(mldest) && isGPR(mlsrc) )
    {
              GPR_Scalar  &dest = dynamic_cast<      GPR_Scalar  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRAnions(mldest) && isGPR(mlsrc) )
    {
              GPR_Anions  &dest = dynamic_cast<      GPR_Anions  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRGentyp(mldest) && isGPR(mlsrc) )
    {
              GPR_Gentyp  &dest = dynamic_cast<      GPR_Gentyp  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isGPRBinary(mldest) && isGPR(mlsrc) )
    {
              GPR_Binary  &dest = dynamic_cast<      GPR_Binary  &>(mldest);
        const GPR_Generic &src  = dynamic_cast<const GPR_Generic &>(mlsrc );

        dest.setsigma(src.sigma());
    }

    else if ( isLSVScalar(mldest) && isLSV(mlsrc) )
    {
              LSV_Scalar  &dest = dynamic_cast<      LSV_Scalar  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVVector(mldest) && isLSV(mlsrc) )
    {
              LSV_Scalar  &dest = dynamic_cast<      LSV_Scalar  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVAnions(mldest) && isLSV(mlsrc) )
    {
              LSV_Scalar  &dest = dynamic_cast<      LSV_Scalar  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVScScor(mldest) && isLSV(mlsrc) )
    {
              LSV_ScScor  &dest = dynamic_cast<      LSV_ScScor  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVAutoEn(mldest) && isLSV(mlsrc) )
    {
              LSV_AutoEn  &dest = dynamic_cast<      LSV_AutoEn  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVGentyp(mldest) && isLSV(mlsrc) )
    {
              LSV_Gentyp  &dest = dynamic_cast<      LSV_Gentyp  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVPlanar(mldest) && isLSV(mlsrc) )
    {
              LSV_Planar  &dest = dynamic_cast<      LSV_Planar  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isLSVMvRank(mldest) && isLSV(mlsrc) )
    {
              LSV_MvRank  &dest = dynamic_cast<      LSV_MvRank  &>(mldest);
        const LSV_Generic &src  = dynamic_cast<const LSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isIMPExpect(mldest) && isIMP(mlsrc) )
    {
              IMP_Expect  &dest = dynamic_cast<      IMP_Expect  &>(mldest);
        const IMP_Generic &src  = dynamic_cast<const IMP_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isIMPParSVM(mldest) && isIMP(mlsrc) )
    {
              IMP_ParSVM  &dest = dynamic_cast<      IMP_ParSVM  &>(mldest);
        const IMP_Generic &src  = dynamic_cast<const IMP_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSSVScalar(mldest) && isSSV(mlsrc) )
    {
              SSV_Scalar  &dest = dynamic_cast<      SSV_Scalar  &>(mldest);
        const SSV_Generic &src  = dynamic_cast<const SSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSSVBinary(mldest) && isSSV(mlsrc) )
    {
              SSV_Binary  &dest = dynamic_cast<      SSV_Binary  &>(mldest);
        const SSV_Generic &src  = dynamic_cast<const SSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isSSVSingle(mldest) && isSSV(mlsrc) )
    {
              SSV_Single  &dest = dynamic_cast<      SSV_Single  &>(mldest);
        const SSV_Generic &src  = dynamic_cast<const SSV_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isMLMScalar(mldest) && isMLM(mlsrc) )
    {
              MLM_Scalar  &dest = dynamic_cast<      MLM_Scalar  &>(mldest);
        const MLM_Generic &src  = dynamic_cast<const MLM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isMLMBinary(mldest) && isMLM(mlsrc) )
    {
              MLM_Binary  &dest = dynamic_cast<      MLM_Binary  &>(mldest);
        const MLM_Generic &src  = dynamic_cast<const MLM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    else if ( isMLMVector(mldest) && isMLM(mlsrc) )
    {
              MLM_Vector  &dest = dynamic_cast<      MLM_Vector  &>(mldest);
        const MLM_Generic &src  = dynamic_cast<const MLM_Generic &>(mlsrc );

        (void) src;
        (void) dest;
    }

    return;
}








int ML_Mutable::egetparam(int ind, Vector<gentype> &val, const Vector<gentype> &xa, int ia, const Vector<gentype> &xb, int ib) const
{
    return getMLconst().egetparam(ind,val,xa,ia,xb,ib);
}


int ML_Mutable::getparam(int ind, gentype &val, const gentype &xa, int ia, const gentype &xb, int ib) const
{
    return getMLconst().getparam(ind,val,xa,ia,xb,ib);
}

