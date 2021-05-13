
//
// Generic type
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <string>
#include <math.h>
#include <ctype.h>
#include "gentype.h"
#include "opttest.h"
#include "paretotest.h"

// MAXINTFACT is the largest number n for which n! fits in 32 bits, signed.
//            Tested by noting that 12!/11! = 12, but 13!/12! != 13.  I'm
//            being lazy here and assuming 32 bit integers.  Not terribly
//            important for >32 bits (bit of rounding is the only problem)
//            but for <32 bits this will lead to difficulties.
// NUMFNDEF number of defined functions

#define MAXINTFACT 12
#define NUMFNDEF   299

// Needed because you can't have commas in macro arguments

//typedef Dgraph<gentype,double> xDgraph;


void evalgenFunc(int i, int j, const gentype &xa, int ia, const gentype &xb, int ib, gentype &res);

/* - commented out here, but KEEP FOR DOCUMENTATION
// Function information block.  This class contains all relevant information
// for a given function.

class fninfoblock
{
    public:
    // fnname:      function name
    // numargs:     number of arguments taken by function
    //
    // dirchkargs:  binary, bit set if functionality test in evaluation 
    //              requires isValEqnDir true. eg: 6 = 110b means apply
    //              isValEqnDir to arguments 2 and 3 but not argument 1
    // widechkargs: like dirchkargs, but using isValEqn.
    //
    // Basically if an argument is elementwise you need to set dirchkargs,
    // and otherwise you need to set widechkargs.  For example sin is
    // elementwise and norm2 is not.
    //
    // preEvalArgs: binary, bit set if evaluate should be applied to this
    //              argument prior to evaluating the function itself.
    // derivDeffed: set if derivative is defined for this function.
    // isInDetermin:0 if function is deterministic (eg sin)
    //              1 global function indeterminant
    //              2 random indeterminant
    //
    // fn0arg: pointer to 0 argument fn
    // fn1arg: pointer to 1 argument fn
    // fn2arg: pointer to 2 argument fn
    // fn3arg: pointer to 3 argument fn
    // fn4arg: pointer to 4 argument fn
    // fn5arg: pointer to 5 argument fn
    // fn6arg: pointer to 6 argument fn
    //
    // fn0arg: pointer to 0 argument fn, operator type (if defined)
    // fn1arg: pointer to 1 argument fn, operator type (if defined)
    // fn2arg: pointer to 2 argument fn, operator type (if defined)
    // fn3arg: pointer to 3 argument fn, operator type (if defined)
    // fn4arg: pointer to 4 argument fn, operator type (if defined)
    // fn5arg: pointer to 5 argument fn, operator type (if defined)
    // fn6arg: pointer to 6 argument fn, operator type (if defined)
    //
    // conjargmod: if negative, reverse order post conjugation
    //             bit 0: conjugate arg 0 if set
    //             bit 1: conjugate arg 1 if set
    //                ...
    //
    // conjfnname: conjugated function name (~ means unchanged)
    //
    // fnconjind: index of conjugated function
    //
    // realargcopy: bit 0: argument 0 used if set
    //              bit 1: argument 1 used if set
    //                 ...
    // realdrvcopy: bit 0: derivative 0 used if set
    //              bit 1: derivative 1 used if set
    //                 ...
    // realderiv:   real derivative, parsed, if constructed.
    // realderivfn: real derivative string.
    //              var(0,i) is the argument i
    //              var(1,i) is the derivative of argument i

    const char *fnname;
    int numargs;
    int dirchkargs;
    int widechkargs;
    int preEvalArgs;
    int derivDeffed;
    int isInDetermin;

    gentype (*fn0arg)();
    gentype (*fn1arg)(const gentype &);
    gentype (*fn2arg)(const gentype &, const gentype &);
    gentype (*fn3arg)(const gentype &, const gentype &, const gentype &);
    gentype (*fn4arg)(const gentype &, const gentype &, const gentype &, const gentype &);
    gentype (*fn5arg)(const gentype &, const gentype &, const gentype &, const gentype &, const gentype &);
    gentype (*fn6arg)(const gentype &, const gentype &, const gentype &, const gentype &, const gentype &, const gentype &);

    gentype &(*OP_fn0arg)();
    gentype &(*OP_fn1arg)(gentype &);
    gentype &(*OP_fn2arg)(gentype &, const gentype &);
    gentype &(*OP_fn3arg)(gentype &, const gentype &, const gentype &);
    gentype &(*OP_fn4arg)(gentype &, const gentype &, const gentype &, const gentype &);
    gentype &(*OP_fn5arg)(gentype &, const gentype &, const gentype &, const gentype &, const gentype &);
    gentype &(*OP_fn6arg)(gentype &, const gentype &, const gentype &, const gentype &, const gentype &, const gentype &);

    int conjargmod;
    const char *conjfnname;
    int fnconjind; // index of conjugate function

    int realargcopy;
    int realdrvcopy;
    gentype *realderiv;
    const char *realderivfn;

    ~fninfoblock()
    {
        // NB: we *cannot* rely on deleting realderiv here, as that may require access to
        // parts of fninfo that have already been deleted (at fninfoblock is only used by
        // the global fninfo).  Instead we *must* use the atext function or let the memory
        // leak stand and hope the OS does appropriate cleanup).
        //
        //if ( realderiv )
        //{
        //    MEMDEL(realderiv);
        //    realderiv = NULL;
        //}

        // All other pointers point to statics, so don't delete them

        return;
    }
};
*/


// getfninfo:    Return array containing all the function information
// getfnind:     Get the index for the function string given
// getfnindConj: Get the index for the conjugate fn of the fn string given
// getfninfo:    Get function information for given function

//const char *getfnname(int fnnameind);
//int getfnind(const std::string &fnname);
//int getfnindConj(int fnInd);
//const fninfoblock *getfninfo(int fnIndex);







class eqninfoblock
{
public:

    // text:   relevant text for this block
    // type:   0 == opcode
    //         1 == number
    //         2 == expression
    // res:    for numbers, 1 = int,
    //         2 = real,
    //         3 = anion,
    //         4 = vector,
    //         5 = matrix,
    //         6 = string,
    //         7 = error,
    //         8 = set,
    //         9 = dgraph
    //         for expressions this is the number of arguments in the expression
    // fnname: for expressions this is the name of the function
    // commas: for expressions this is a vector containing the positions of all commas
    // isstr:  for expressions this indicates if the expression is a string (no following brackets)

    std::string text;
    int type;
    int res;
    std::string fnname;
    Vector<int> commas;
    int isstr;
};

std::istream &operator>>(std::istream &input, eqninfoblock &dest);
std::istream &streamItIn(std::istream &input, eqninfoblock &dest, int processxyzvw = 1);

std::istream &operator>>(std::istream &input, eqninfoblock &dest) { (void) dest; throw("Blind"); return input; }
std::istream &streamItIn(std::istream &input, eqninfoblock &dest, int processxyzvw) { (void) dest; (void) processxyzvw; throw("Blind"); return input; }

STREAMINDUMMY(const eqninfoblock *); STREAMINDUMMY(eqninfoblock *);



inline eqninfoblock &setident (eqninfoblock &a);
inline eqninfoblock &setzero  (eqninfoblock &a);
inline eqninfoblock &setposate(eqninfoblock &a);
inline eqninfoblock &setnegate(eqninfoblock &a);
inline eqninfoblock &setconj  (eqninfoblock &a);
inline eqninfoblock &setrand  (eqninfoblock &a); // random -1 or 1
inline eqninfoblock &leftmult (eqninfoblock &a, eqninfoblock  b);
inline eqninfoblock &rightmult(eqninfoblock  a, eqninfoblock &b);
inline eqninfoblock &postProInnerProd(eqninfoblock &a);

inline eqninfoblock &setident (eqninfoblock &a) { throw("Huh?"); return a; }
inline eqninfoblock &setzero  (eqninfoblock &a) { eqninfoblock b; return a = b; }
inline eqninfoblock &setposate(eqninfoblock &a) { return a; }
inline eqninfoblock &setnegate(eqninfoblock &a) { throw("Computer says no."); return a; }
inline eqninfoblock &setconj  (eqninfoblock &a) { throw("Ummm... yeah!"); return a; }
inline eqninfoblock &setrand  (eqninfoblock &a) { throw("What say you?"); return a; }
inline eqninfoblock &leftmult (eqninfoblock &a, eqninfoblock  b) { (void) b; throw("No, no and no!"); return a; }
inline eqninfoblock &rightmult(eqninfoblock  a, eqninfoblock &b) { (void) a; throw("Banana."); return b; }
inline eqninfoblock &postProInnerProd(eqninfoblock &a) { return a; }



int pairBrackets(int start, int &end, const std::string &src, int LRorRL);
int processNumLtoR(int start, int &end, const std::string &src);
int processExprLtoR(int start, int &end, int &isitastring, const std::string &src, std::string &exprname, Vector<int> &commapos);
std::ostream &operator<<(std::ostream &output, const eqninfoblock &src );
void qswap(eqninfoblock &a, eqninfoblock &b);
int operatorToFunction(int LtoRRtoL, int UnaryBinary, const Vector<std::string> &opSymb, const Vector<std::string> &opFuncEquiv, Vector<eqninfoblock> &srcxblock);



double gentypeToMatrixRep(const gentype &temp, int dim, int iq, int jq)
{
    double res;

    if ( ( iq >= dim ) || ( jq >= dim ) )
    {
        throw("Order error in matrix representation computation (mercer).");
    }

    if ( temp.isCastableToRealWithoutLoss() )
    {
        if ( iq == jq )
        {
            res = temp.cast_double(0);
        }

        else
        {
            res = 0.0;
        }
    }

    else if ( temp.isCastableToAnionWithoutLoss() )
    {
        int order = ceilintlog2(dim);

        d_anion restemp(temp.cast_anion(0));
        restemp.setorder(order);

        // Re(conj(x).Kij.y)
        // = xq.Kq.y0 + xq.K0.yq - x0.Kq.yq + xq.epsilon_qrs.Kr.ys
        //
        // Complex = [ x0 xi ] [ K0 -Ki ] [ y0 ]
        //                     [ Ki  K0 ] [ yi ]
        //
        // Quaternion = [ x0 xi xj xk ] [ K0 -Ki -Kj -Kk ] [ y0 ]
        //                              [ Ki  K0 -Kk  Kj ] [ yi ]
        //                              [ Kj  Kk  K0 -Ki ] [ yj ]
        //                              [ Kk -Kj  Ki  K0 ] [ yk ]

        if ( order == 0 )
        {
            // real case

            res = restemp(0);
        }

        else if ( order == 1 )
        {
            // complex case

            if (      ( iq == 0 ) && ( jq == 0 ) ) { res =  restemp(0); }
            else if ( ( iq == 0 ) && ( jq == 1 ) ) { res = -restemp(1); }
            else if ( ( iq == 1 ) && ( jq == 0 ) ) { res =  restemp(1); }
            else if ( ( iq == 1 ) && ( jq == 1 ) ) { res =  restemp(0); }

            else
            {
                throw("Order error in matrix representation computation (mercer).");
            }
        }

        else if ( order == 2 )
        {
            // quaternion case

            if (      ( iq == 0 ) && ( jq == 0 ) ) { res =  restemp(0); }
            else if ( ( iq == 0 ) && ( jq == 1 ) ) { res = -restemp(1); }
            else if ( ( iq == 0 ) && ( jq == 2 ) ) { res = -restemp(2); }
            else if ( ( iq == 0 ) && ( jq == 3 ) ) { res = -restemp(3); }
            else if ( ( iq == 1 ) && ( jq == 0 ) ) { res =  restemp(1); }
            else if ( ( iq == 1 ) && ( jq == 1 ) ) { res =  restemp(0); }
            else if ( ( iq == 1 ) && ( jq == 2 ) ) { res = -restemp(3); }
            else if ( ( iq == 1 ) && ( jq == 3 ) ) { res =  restemp(2); }
            else if ( ( iq == 2 ) && ( jq == 0 ) ) { res =  restemp(2); }
            else if ( ( iq == 2 ) && ( jq == 1 ) ) { res =  restemp(3); }
            else if ( ( iq == 2 ) && ( jq == 2 ) ) { res =  restemp(0); }
            else if ( ( iq == 2 ) && ( jq == 3 ) ) { res = -restemp(1); }
            else if ( ( iq == 3 ) && ( jq == 0 ) ) { res =  restemp(3); }
            else if ( ( iq == 3 ) && ( jq == 1 ) ) { res = -restemp(2); }
            else if ( ( iq == 3 ) && ( jq == 2 ) ) { res =  restemp(1); }
            else if ( ( iq == 3 ) && ( jq == 3 ) ) { res =  restemp(0); }

            else
            {
                throw("Order error in matrix representation computation (mercer).");
            }
        }

        else
        {
            // general case

            int r;

            if ( iq == jq )
            {
                res = restemp(0);
            }

            else if ( !iq )
            {
                res = -restemp(jq);
            }

            else if ( !jq )
            {
                res = restemp(iq);
            }

            else
            {
                r = 1;

                while ( !epsilon(order,iq,r,jq) && ( r < 1<<order ) )
                {
                    r++;
                }

                if ( r < 1<<order )
                {
                    res = epsilon(order,iq,r,jq)*restemp(r);
                }

                else
                {
                    res = 0.0;
                }
            }
        }
    }

    else if ( temp.isCastableToMatrixWithoutLoss() )
    {
        res = (temp.cast_matrix(0))(iq,jq);
    }

    else
    {
        throw("Kernel error: non-matrix-castable result for matrix kernel");
    }

    return res;
}

Matrix<double> &gentypeToMatrixRep(Matrix<double> &res, const gentype &src, int spaceDim)
{
    res.resize(spaceDim,spaceDim);
    res = 0.0;

    if ( src.isCastableToRealWithoutLoss() )
    {
        double tempres = src.cast_double(0);

        int i;

        for ( i = 0 ; i < spaceDim ; i++ )
        {
            res("&",i,i) = tempres;
        }
    }

    else if ( src.isCastableToAnionWithoutLoss() )
    {
        int order = ceilintlog2(spaceDim);

        NiceAssert( spaceDim == 1<<order );

        d_anion tempres = src.cast_anion(0);

        int i;

        for ( i = 0 ; i < spaceDim ; i++ )
        {
            res("&",i,i) = tempres(0);
        }

        tempres.setorder(order);

        // Re(conj(x).Kij.y)
        // = xq.Kq.y0 + xq.K0.yq - x0.Kq.yq + xq.epsilon_qrs.Kr.ys
        //
        // Complex = [ x0 xi ] [ K0 -Ki ] [ y0 ]
        //                     [ Ki  K0 ] [ yi ]
        //
        // Quaternion = [ x0 xi xj xk ] [ K0 -Ki -Kj -Kk ] [ y0 ]
        //                              [ Ki  K0 -Kk  Kj ] [ yi ]
        //                              [ Kj  Kk  K0 -Ki ] [ yj ]
        //                              [ Kk -Kj  Ki  K0 ] [ yk ]

        if ( order == 0 )
        {
            // real case

            res("&",0,0) = tempres(0);
        }

        else if ( order == 1 )
        {
            // complex case

            res("&",0,0) =  tempres(0);
            res("&",0,1) = -tempres(1);
            res("&",1,0) =  tempres(1);
            res("&",1,1) =  tempres(0);
        }

        else if ( order == 2 )
        {
            // quaternion case

            res("&",0,0) =  tempres(0);
            res("&",0,1) = -tempres(1);
            res("&",0,2) = -tempres(2);
            res("&",0,3) = -tempres(3);
            res("&",1,0) =  tempres(1);
            res("&",1,1) =  tempres(0);
            res("&",1,2) = -tempres(3);
            res("&",1,3) =  tempres(2);
            res("&",2,0) =  tempres(2);
            res("&",2,1) =  tempres(3);
            res("&",2,2) =  tempres(0);
            res("&",2,3) = -tempres(1);
            res("&",3,0) =  tempres(3);
            res("&",3,1) = -tempres(2);
            res("&",3,2) =  tempres(1);
            res("&",3,3) =  tempres(0);
        }

        else
        {
            // general case

            int q,r,s;

            for ( q = 0 ; q < tempres.size() ; q++ )
            {
                for ( s = 0 ; s < tempres.size() ; s++ )
                {
                    if ( q == s )
                    {
                        res("&",q,s) = tempres(0);
                    }

                    else if ( !q )
                    {
                        res("&",q,s) = -tempres(s);
                    }

                    else if ( !s )
                    {
                        res("&",q,s) = tempres(q);
                    }

                    else
                    {
                        r = 1;

                        while ( !epsilon(order,q,r,s) && ( r < tempres.size() ) )
                        {
                            r++;
                        }

                        if ( r < tempres.size() )
                        {
                            res("&",q,s) = epsilon(order,q,r,s)*tempres(r);
                        }

                        else
                        {
                            res("&",q,s) = 0;
                        }
                    }
                }
            }
        }
    }

    else if ( src.isCastableToMatrixWithoutLoss() )
    {
        int i,j;
        Matrix<gentype> tempres;

        tempres = src.cast_matrix(0);

        NiceAssert( spaceDim == tempres.numRows() );
        NiceAssert( tempres.isSquare() );

        if ( tempres.numRows() && tempres.numCols() )
        {
            for ( i = 0 ; i < tempres.numRows() ; i++ )
            {
                for ( j = 0 ; j < tempres.numCols() ; j++ )
                {
                    res("&",i,j) = tempres(i,j).cast_double(0);
                }
            }
        }
    }

    else
    {
        throw("Kernel error: non-matrix-castable result for matrix kernel");
    }

    return res;
}













std::istream &operator>>(std::istream &input, gentype &dest)
{
    return streamItIn(input,dest);
}

std::ostream &operator<<(std::ostream &output, const gentype &src )
{
    // Aside: we don't use isCastableToIntegerWithoutLoss here, as this may
    // return true for an anion, and we want to preserve the *order* of the
    // anion (complex, quaternion, whatever).  Likewise we avoid demoting
    // anions to real even if it can be done, and we avoid demoting reals to
    // integers by optionally appending .0 to the end of the result.

    if ( src.scalarfn_isscalarfn() )
    {
        output << "@(";
        output << src.scalarfn_i() << ",";
        output << src.scalarfn_j() << ",";
        output << src.scalarfn_numpts() << "):";
    }

         if ( src.isValNull()    ) { output         << "null";             }
    else if ( src.isValInteger() ) { output         << src.cast_int(0);    }
    else if ( src.isValAnion()   ) { output         << src.cast_anion(0);  }
    else if ( src.isValVector()  ) { output         << src.cast_vector(0); }
    else if ( src.isValMatrix()  ) { output << "M:" << src.cast_matrix(0); }
    else if ( src.isValSet()     ) { output         << src.cast_set(0);    }
    else if ( src.isValDgraph()  ) { output << "G:" << src.cast_dgraph(0); }

    else if ( src.isValReal() )
    {
        double srcdval = src.cast_double(0);

        if ( testisvnan(srcdval) )
        {
            output << "vnan()";
        }

        else if ( testispinf(srcdval) )
        {
            output << "pinf()";
        }

        else if ( testisninf(srcdval) )
        {
            output << "ninf()";
        }

	else
	{
            // Small niggle.  If number is real then we want to retain the
            // "realness" - that is, if read back it should parse as real.
            // To ensure this we need to add .0 to the end if neither e nor
            // . is present in the printed result.

            streamItOut(output,srcdval,1);

//            char tempres[100];
//            sprintf(tempres,"%.17g",srcdval);
//            std::string tempresb(tempres);
//            output << tempresb;
//
//            if ( !tempresb.find(".") && !tempresb.find("e") && !tempresb.find("E") )
//            {
//                output << ".0";
//            }
	}
    }

    else
    {
        std::string resstring(src.cast_string(0));

        if ( src.isValString() )
        {
            output << '\"';
        }

	if ( src.size() )
	{
            unsigned int i;

            // Looping to maintain escape characters

            for ( i = 0 ; i < resstring.size() ; i++ )
            {           
                if ( ( i == 2 ) && src.isValError() )
                {
                    output << '\"' << resstring[i];
                }

                //if ( resstring[i] == '\"' ) { output << "\\\""; }
                //else if ( resstring[i] == '\'' ) { output << "\\\'"; }
                //else if ( resstring[i] == '\?' ) { output << "\\\?"; }
                //else if ( resstring[i] == '\\' ) { output << "\\"; }
                //else if ( resstring[i] == '\a' ) { output << "\\a"; }
                //else if ( resstring[i] == '\b' ) { output << "\\b"; }
                //else if ( resstring[i] == '\f' ) { output << "\\f"; }
                //else if ( resstring[i] == '\n' ) { output << "\\n"; }
                //else if ( resstring[i] == '\r' ) { output << "\\r"; }
                //else if ( resstring[i] == '\t' ) { output << "\\t"; }
                //else if ( resstring[i] == '\v' ) { output << "\\v"; }

                else
                {
                    output << resstring[i];
                }
	    }
	}

        if ( src.isValString() || src.isValError() )
        {
            output << '\"';
        }
    }

    return output;
}

std::istream &streamItIn(std::istream &input, gentype &dest, int processxyzvw)
{
    std::string resstore;
    char tt;

    // Note: this bit is important.  Weirdness, but if you wonder
    // why try inputing
    //
    // [ 1
    // 2 ]
    //
    // (with the newline) directly into std::cin >> an instance of gentype.
    // Without the following "pause" code this will not work.

    while ( isspace(input.peek()) )
    {
	input.get(tt);
    }

    input.peek();

    // Need to look for scalar function indicator

    Vector<int> sfni(1);
    Vector<int> sfnj(1);

    sfni = (int) DEFAULTVARI;
    sfnj = (int) DEFAULTVARJ;

    dest.scalarfn_setisscalarfn(0);
    dest.scalarfn_setnumpts(DEFAULT_INTEGRAL_SLICES);

    dest.scalarfn_seti(sfni);
    dest.scalarfn_setj(sfnj);

    if ( input.peek() == '@' )
    {
        dest.scalarfn_setisscalarfn(1);

        char tt;

        input.get(tt);
        NiceAssert( tt == '@' );
        input.get(tt);
        NiceAssert( tt == '(' );

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        if ( input.peek() == ')' )
        {
            input.get(tt);
            goto donepoint;
        }

        {
            Vector<int> nv;

            streamItIn(input,nv,processxyzvw);

            dest.scalarfn_seti(nv);
        }

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        input.get(tt);

        if ( tt == ')' )
        {
            goto donepoint;
        }

        NiceAssert( tt == ',' );

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        {
            Vector<int> nv;

            streamItIn(input,nv,processxyzvw);

            dest.scalarfn_setj(nv);
        }

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        input.get(tt);

        if ( tt == ')' )
        {
            goto donepoint;
        }

        NiceAssert( tt == ',' );

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        {
            int nv = 0;
            input.get(tt);

            while ( isdigit(tt) )
            {
                nv *= 10;
                nv += tt-'0';

                input.get(tt);
            }

            dest.scalarfn_setnumpts(nv);
        }

        while ( isspace(input.peek()) )
        {
            input.get(tt);
        }

        input.get(tt);
        NiceAssert( tt == ')' );

donepoint:

        input.get(tt);
        NiceAssert( tt == ':' );
    }

    // Treat any number as an equation.  Then, when we call makeEqn,
    // this will simplify if possible and convert number equations
    // into actual numbers.

    if ( readParenString(input,resstore) )
    {
        throw("Unpaired brackets in streamItIn gentype expression.");
    }

    if ( dest.makeEqn(resstore,processxyzvw) )
    {
errstream() << "phantomxyz 0: " << resstore << "\n";
        throw("Syntax error in streamItIn gentype expression.");
    }

    return input;
}
















// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------


const gentype &twoProduct(gentype &res, const gentype &a, const gentype &b)
{
    if ( ( &res == &a ) && ( &res == &b ) )
    {
        gentype aa(a);
        gentype bb(b);

        twoProduct(res,aa,bb);
    }

    else if ( &res == &a )
    {
        gentype aa(a);

        twoProduct(res,aa,b);
    }

    else if ( &res == &b )
    {
        gentype bb(b);

        twoProduct(res,a,bb);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 0 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        res.zero();
        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = ( a.scalarfn_numpts() > b.scalarfn_numpts() ) ? a.scalarfn_numpts() : b.scalarfn_numpts();

        int i;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;
        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);
            xb("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(xa);
            bb = b(xb);

            bb.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        int numvar = a.scalarfn_i().size();
        int numpts = ( a.scalarfn_numpts() > b.scalarfn_numpts() ) ? a.scalarfn_numpts() : b.scalarfn_numpts();
        int numtot = (int) pow(numpts,numvar);

        int i,j;
        Vector<int> k(numvar);

        k = zeroint();

        res.zero();

        SparseVector<SparseVector<gentype> > xa;
        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numtot ; i++ )
        {
            for ( j = 0 ; j < numvar ; j++ )
            {
                xa("&",a.scalarfn_i()(j))("&",a.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
                xb("&",b.scalarfn_i()(j))("&",b.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
            }

            aa = a(xa);
            bb = b(xb);

            bb.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numtot);
            aa *= bb;

            res += aa;

            for ( j = 0 ; j < numvar ; j++ )
            {
                k("&",j)++;

                if ( k(j) >= numpts )
                {
                    k("&",j) = 0;
                }

                else
                {
                    break;
                }
            }
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.isValVector() && b.infsize() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();
        int i;
        gentype aa;
        gentype bb;
        const Vector<gentype> &bvec = b.cast_vector(0);
        SparseVector<SparseVector<gentype> > x;

        res.zero();

        for ( i = 0 ; i < numpts ; i++ )
        {
            x("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(x);
            bb = bvec(x("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())));

            bb.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( b.scalarfn_isscalarfn() && a.isValVector() && a.infsize() && ( b.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( b.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = b.scalarfn_numpts();
        int i;
        gentype aa;
        gentype bb;
        const Vector<gentype> &avec = a.cast_vector(0);
        SparseVector<SparseVector<gentype> > x;

        res.zero();

        for ( i = 0 ; i < numpts ; i++ )
        {
            x("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = avec(x("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())));
            bb = b(x);

            bb.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.isValVector() && b.scalarfn_isscalarfn() && ( b.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( b.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = b.scalarfn_numpts();

        int i,q;

        res.zero();

        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            q = (int) (((((double) i)+0.5)/((double) numpts))*a.size()); // always rounds toward zero
            xb("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(q);
            bb = b(xb);

            bb.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.isValVector() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();

        int i,q;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);
            q = (int) (((((double) i)+0.5)/((double) numpts))*b.size()); // always rounds toward zero

            aa = a(xa);
            bb = b(q);

            bb.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        res = a;
        setconj(res);
        res *= b;
    }

    else if ( a.isValNull() || a.isValInteger() || a.isValReal() || a.isValAnion() )
    {
        res = a;
        setconj(res);
        res *= b;
    }

    else if ( a.isValVector() )
    {
        if ( b.isValVector() )
        {
            twoProduct(res,
            a.cast_vector(0),
            b.cast_vector(0));
        }

        else if ( b.isValMatrix() )
        {
            res.force_vector() = a.cast_vector(0);
            setconj(res.dir_vector());
            res.dir_vector() *= b.cast_matrix(0);
        }

        else
        {
            res.force_vector() = a.cast_vector(0);
            setconj(res.dir_vector());
            res.dir_vector() *= b;
        }
    }

    else if ( a.isValMatrix() )
    {
        if ( b.isValVector() )
        {
            res.force_vector() = b.cast_vector(0);
            setconj(res.dir_vector());
            res.dir_vector() *= a.cast_matrix(0);
            setconj(res.dir_vector());
        }

        else if ( b.isValMatrix() )
        {
            res.force_matrix() = a.cast_matrix(0);
            setconj(res.dir_matrix());
            res.dir_matrix() *= b.cast_matrix(0);
        }

        else
        {
            res.force_matrix() = a.cast_matrix(0);
            setconj(res.dir_matrix());
            res.dir_matrix() *= b;
        }
    }

    else
    {
        if ( b.isValVector() )
        {
            res.force_vector() = b.cast_vector(0);
            setconj(res.dir_vector());
            res.dir_vector() *= a;
            setconj(res.dir_vector());
        }

        else if ( b.isValMatrix() )
        {
            res.force_matrix() = b.cast_matrix(0);
            setconj(res.dir_matrix());
            res.dir_matrix() *= a;
            setconj(res.dir_matrix());
        }

        else
        {
            res = a;
            setconj(res);
            res *= b;
        }
    }

    return res;
}

const gentype &twoProductNoConj(gentype &res, const gentype &a, const gentype &b)
{
    if ( ( &res == &a ) && ( &res == &b ) )
    {
        gentype aa(a);
        gentype bb(b);

        twoProduct(res,aa,bb);
    }

    else if ( &res == &a )
    {
        gentype aa(a);

        twoProduct(res,aa,b);
    }

    else if ( &res == &b )
    {
        gentype bb(b);

        twoProduct(res,a,bb);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 0 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        res.zero();
        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = ( a.scalarfn_numpts() > b.scalarfn_numpts() ) ? a.scalarfn_numpts() : b.scalarfn_numpts();

        int i;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;
        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);
            xb("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(xa);
            bb = b(xb);

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        int numvar = a.scalarfn_i().size();
        int numpts = ( a.scalarfn_numpts() > b.scalarfn_numpts() ) ? a.scalarfn_numpts() : b.scalarfn_numpts();
        int numtot = (int) pow(numpts,numvar);

        int i,j;
        Vector<int> k(numvar);

        k = zeroint();

        res.zero();

        SparseVector<SparseVector<gentype> > xa;
        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numtot ; i++ )
        {
            for ( j = 0 ; j < numvar ; j++ )
            {
                xa("&",a.scalarfn_i()(j))("&",a.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
                xb("&",b.scalarfn_i()(j))("&",b.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
            }

            aa = a(xa);
            bb = b(xb);

            aa.finalise();
            bb.finalise();

            aa /= ((double) numtot);
            aa *= bb;

            res += aa;

            for ( j = 0 ; j < numvar ; j++ )
            {
                k("&",j)++;

                if ( k(j) >= numpts )
                {
                    k("&",j) = 0;
                }

                else
                {
                    break;
                }
            }
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.isValVector() && b.infsize() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();
        int i;
        gentype aa;
        gentype bb;
        const Vector<gentype> &bvec = b.cast_vector(0);
        SparseVector<SparseVector<gentype> > x;

        res.zero();

        for ( i = 0 ; i < numpts ; i++ )
        {
            x("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(x);
            bb = bvec(x("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())));

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( b.scalarfn_isscalarfn() && a.isValVector() && a.infsize() && ( b.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( b.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = b.scalarfn_numpts();
        int i;
        gentype aa;
        gentype bb;
        const Vector<gentype> &avec = a.cast_vector(0);
        SparseVector<SparseVector<gentype> > x;

        res.zero();

        for ( i = 0 ; i < numpts ; i++ )
        {
            x("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = avec(x("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())));
            bb = b(x);

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.isValVector() && b.scalarfn_isscalarfn() && ( b.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( b.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = b.scalarfn_numpts();

        int i,q;

        res.zero();

        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            q = (int) (((((double) i)+0.5)/((double) numpts))*a.size()); // always rounds toward zero
            xb("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(q);
            bb = b(xb);

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.isValVector() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();

        int i,q;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);
            q = (int) (((((double) i)+0.5)/((double) numpts))*b.size()); // always rounds toward zero

            aa = a(xa);
            bb = b(q);

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        res = a;
        res *= b;
    }

    else if ( a.isValNull() || a.isValInteger() || a.isValReal() || a.isValAnion() )
    {
        res = a;
        res *= b;
    }

    else if ( a.isValVector() )
    {
        if ( b.isValVector() )
        {
            twoProductNoConj(res,a.cast_vector(0),b.cast_vector(0));
        }

        else if ( b.isValMatrix() )
        {
            res.force_vector() = a.cast_vector(0);
            res.dir_vector() *= b.cast_matrix(0);
        }

        else
        {
            res.force_vector() = a.cast_vector(0);
            res.dir_vector() *= b;
        }
    }

    else if ( a.isValMatrix() )
    {
        if ( b.isValVector() )
        {
            res.force_vector() = b.cast_vector(0);
            res.dir_vector() *= a.cast_matrix(0);
        }

        else if ( b.isValMatrix() )
        {
            res.force_matrix() = a.cast_matrix(0);
            res.dir_matrix() *= b.cast_matrix(0);
        }

        else
        {
            res.force_matrix() = a.cast_matrix(0);
            res.dir_matrix() *= b;
        }
    }

    else
    {
        if ( b.isValVector() )
        {
            res.force_vector() = b.cast_vector(0);
            res.dir_vector() *= a;
        }

        else if ( b.isValMatrix() )
        {
            res.force_matrix() = b.cast_matrix(0);
            res.dir_matrix() *= a;
        }

        else
        {
            res = a;
            res *= b;
        }
    }

    return res;
}

const gentype &twoProductRevConj(gentype &res, const gentype &a, const gentype &b)
{
    if ( ( &res == &a ) && ( &res == &b ) )
    {
        gentype aa(a);
        gentype bb(b);

        twoProduct(res,aa,bb);
    }

    else if ( &res == &a )
    {
        gentype aa(a);

        twoProduct(res,aa,b);
    }

    else if ( &res == &b )
    {
        gentype bb(b);

        twoProduct(res,a,bb);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 0 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        res.zero();
        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = ( a.scalarfn_numpts() > b.scalarfn_numpts() ) ? a.scalarfn_numpts() : b.scalarfn_numpts();

        int i;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;
        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);
            xb("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(xa);
            bb = b(xb);

            aa.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.scalarfn_isscalarfn() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_i().size() );
        NiceAssert( a.scalarfn_i().size() == b.scalarfn_j().size() );

        int numvar = a.scalarfn_i().size();
        int numpts = ( a.scalarfn_numpts() > b.scalarfn_numpts() ) ? a.scalarfn_numpts() : b.scalarfn_numpts();
        int numtot = (int) pow(numpts,numvar);

        int i,j;
        Vector<int> k(numvar);

        k = zeroint();

        res.zero();

        SparseVector<SparseVector<gentype> > xa;
        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numtot ; i++ )
        {
            for ( j = 0 ; j < numvar ; j++ )
            {
                xa("&",a.scalarfn_i()(j))("&",a.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
                xb("&",b.scalarfn_i()(j))("&",b.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
            }

            aa = a(xa);
            bb = b(xb);

            aa.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numtot);
            aa *= bb;

            res += aa;

            for ( j = 0 ; j < numvar ; j++ )
            {
                k("&",j)++;

                if ( k(j) >= numpts )
                {
                    k("&",j) = 0;
                }

                else
                {
                    break;
                }
            }
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.isValVector() && b.infsize() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();
        int i;
        gentype aa;
        gentype bb;
        const Vector<gentype> &bvec = b.cast_vector(0);
        SparseVector<SparseVector<gentype> > x;

        res.zero();

        for ( i = 0 ; i < numpts ; i++ )
        {
            x("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(x);
            bb = bvec(x("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())));

            aa.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( b.scalarfn_isscalarfn() && a.isValVector() && a.infsize() && ( b.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( b.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = b.scalarfn_numpts();
        int i;
        gentype aa;
        gentype bb;
        const Vector<gentype> &avec = a.cast_vector(0);
        SparseVector<SparseVector<gentype> > x;

        res.zero();

        for ( i = 0 ; i < numpts ; i++ )
        {
            x("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = avec(x("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())));
            bb = b(x);

            aa.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.isValVector() && b.scalarfn_isscalarfn() && ( b.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( b.scalarfn_i().size() == b.scalarfn_j().size() );

        int numpts = b.scalarfn_numpts();

        int i,q;

        res.zero();

        SparseVector<SparseVector<gentype> > xb;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            q = (int) (((((double) i)+0.5)/((double) numpts))*a.size()); // always rounds toward zero
            xb("&",b.scalarfn_i()(zeroint()))("&",b.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(q);
            bb = b(xb);

            aa.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.scalarfn_isscalarfn() && b.isValVector() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();

        int i,q;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa;
        gentype bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);
            q = (int) (((((double) i)+0.5)/((double) numpts))*b.size()); // always rounds toward zero

            aa = a(xa);
            bb = b(q);

            aa.conj();

            aa.finalise();
            bb.finalise();

            aa /= ((double) numpts);
            aa *= bb;

            res += aa;
        }

        res.scalarfn_setisscalarfn(0);
    }

    else if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        res = b;
        setconj(res);
        rightmult(a,res);
    }

    else if ( a.isValNull() || a.isValInteger() || a.isValReal() || a.isValAnion() )
    {
        res = b;
        setconj(res);
        rightmult(a,res);
    }

    else if ( a.isValVector() )
    {
        if ( b.isValVector() )
        {
            twoProductRevConj(res,
            a.cast_vector(0),
            b.cast_vector(0));
        }

        else if ( b.isValMatrix() )
        {
            res.force_vector() = a.cast_vector(0);
            res.dir_vector() *= conj(b);
        }

        else
        {
            res.force_vector() = a.cast_vector(0);
            res.dir_vector() *= conj(b);
        }
    }

    else if ( a.isValMatrix() )
    {
        if ( b.isValVector() )
        {
            res.force_vector() = (a.cast_matrix(0))*conj(b);
        }

        else if ( b.isValMatrix() )
        {
            res.force_matrix() = a.cast_matrix(0);
            res.dir_matrix() *= conj(b);
        }

        else
        {
            res.force_matrix() = a.cast_matrix(0);
            res.dir_matrix() *= conj(b);
        }
    }

    else
    {
        if ( b.isValVector() )
        {
            res.force_vector() = a*conj(b);
        }

        else if ( b.isValMatrix() )
        {
            res.force_matrix() = a*conj(b);
        }

        else
        {
            res = a;
            res *= conj(b);
        }
    }

    return res;
}

const gentype &fourProduct(gentype &res, const gentype &a, const gentype &b, const gentype &c, const gentype &d)
{
    return twoProductNoConj(res,a*b,c*d);
}


// Assignment operators

gentype &gentype::fastcopy(const gentype &src, int areDistinct)
{
    // Very important subtlety here: often in the code we have expressions
    // like *this = eqnargs(i) - that is, the current node is overwritten by
    // one of its children.  Therefore we must preserve the contents of the
    // child nodes, vectors etc before assignment occurs.
    //
    // Relevant reversion branch: 23/6/2015

    if ( src.varid_isscalar )
    {
        varid_isscalar = src.varid_isscalar;
        varid_numpts   = src.varid_numpts;

        varid_xi = src.varid_xi;
        varid_xj = src.varid_xj;
    }

    if ( isfasttype() && src.isfasttype() )
    {
        //deleteVectMatMem(); just leave any temporaries lying around, they
        // will be deleted later anyhow and deleting them wastes time.

        typeis     = src.typeis;
        intval     = src.intval;
        doubleval  = src.doubleval;
    }

    else if ( areDistinct )
    {
        int i,j;

        if ( src.isValAnion() )
        {
            deleteVectMatMem('A');

            *anionval = *(src.anionval);
        }

        else if ( src.isValVector() )
        {
            deleteVectMatMem('V',(*(src.vectorval)).size());

            if ( (*(src.vectorval)).infsize() )
            {
                (*vectorval) = *(src.vectorval);
            }

            else
            {
                (*vectorval).resize((*(src.vectorval)).size());

                for ( i = 0 ; i < (*(src.vectorval)).size() ; i++ )
                {
                    (*vectorval)("&",i).fastcopy((*(src.vectorval))(i),areDistinct);
                }
            }
        }

        else if ( src.isValEqnDir() )
        {
            deleteVectMatMem();

            MEMNEW(eqnargs,Vector<gentype>((*(src.eqnargs)).size()));

            for ( i = 0 ; i < (*(src.eqnargs)).size() ; i++ )
            {
                (*eqnargs)("&",i).fastcopy((*(src.eqnargs))(i),areDistinct);
            }
        }

        else if ( src.isValMatrix() )
        {
            deleteVectMatMem('M',(*(src.matrixval)).numRows(),(*(src.matrixval)).numCols());
            (*matrixval).resize((*(src.matrixval)).numRows(),(*(src.matrixval)).numCols());

            for ( i = 0 ; i < (*(src.matrixval)).numRows() ; i++ )
            {
                for ( j = 0 ; j < (*(src.matrixval)).numCols() ; j++ )
                {
                    (*matrixval)("&",i,j).fastcopy((*(src.matrixval))(i,j),areDistinct);
                }
            }
        }

        else if ( src.isValSet() )
        {
            deleteVectMatMem('X');

            *setval = *(src.setval);
        }

        else if ( src.isValDgraph() )
        {
            deleteVectMatMem('G');

            *dgraphval = *(src.dgraphval);
        }

        else if ( src.isValString() )
        {
            deleteVectMatMem('S');

            *stringval = *(src.stringval);
        }

        else if ( src.isValError() )
        {
            deleteVectMatMem('E');

            *stringval = *(src.stringval);
        }

        else
        {
            deleteVectMatMem();
        }

        typeis     = src.typeis;
        intval     = src.intval;
        doubleval  = src.doubleval;
        fnnameind  = src.fnnameind;
        thisfninfo = src.thisfninfo;
    }

    else
    {
        d_anion                *anistore = NULL;
        Vector<gentype>        *vecstore = NULL;
        Matrix<gentype>        *matstore = NULL;
        Set<gentype>           *setstore = NULL;
        Dgraph<gentype,double> *dgrstore = NULL;
        Vector<gentype>        *eqnstore = NULL;
        std::string            *strstore = NULL;

        int wasValAnion  = src.isValAnion();
        int wasValVector = src.isValVector();
        int wasValMatrix = src.isValMatrix();
        int wasValSet    = src.isValSet();
        int wasValDgraph = src.isValDgraph();
        int wasValEqn    = src.isValEqnDir();
        int wasValStrErr = src.isValStrErr();

        if ( wasValAnion  ) { MEMNEW(anistore,d_anion(*(src.anionval)));                 }
        if ( wasValVector ) { MEMNEW(vecstore,Vector<gentype       >(*(src.vectorval))); }
        if ( wasValMatrix ) { MEMNEW(matstore,Matrix<gentype       >(*(src.matrixval))); }
        if ( wasValSet    ) { MEMNEW(setstore,Set<   gentype       >(*(src.setval)));    }
        if ( wasValDgraph ) { MEMNEW(dgrstore,xDgraph               (*(src.dgraphval))); }
        if ( wasValEqn    ) { MEMNEW(eqnstore,Vector<gentype       >(*(src.eqnargs)));   }
        if ( wasValStrErr ) { MEMNEW(strstore,std::string(*(src.stringval)));            }

        char              srctypeis      = src.typeis;
        int               srcintval      = src.intval;
        double            srcdoubleval   = src.doubleval;
        int               srcfnnameind   = src.fnnameind;
        const fninfoblock *srcthisfninfo = src.thisfninfo;

        // Only now can we safely delete the contents of *this

        if ( wasValEqn )
        {
            deleteVectMatMem();
        }

        typeis     = srctypeis;
        intval     = srcintval;
        doubleval  = srcdoubleval;
        fnnameind  = srcfnnameind;
        thisfninfo = srcthisfninfo;

        if ( wasValAnion  ) { *this = *anistore;     MEMDEL(anistore); }
        if ( wasValVector ) { *this = *vecstore;     MEMDEL(vecstore); }
        if ( wasValMatrix ) { *this = *matstore;     MEMDEL(matstore); }
        if ( wasValSet    ) { *this = *setstore;     MEMDEL(setstore); }
        if ( wasValDgraph ) { *this = *dgrstore;     MEMDEL(dgrstore); }
        if ( wasValStrErr ) { makeString(*strstore); MEMDEL(strstore); }

        if ( wasValEqn )
        {
            MEMNEW(eqnargs,Vector<gentype>);
            *eqnargs = *eqnstore;
            MEMDEL(eqnstore);
        }
    }

    return *this;
}

void gentype::switcheroo(gentype &src)
{
    gentype temp; // will be constructed to zero int, which is really quick
                  // will be deleted on exit like any local variable

    int loc_varid_isscalar = varid_isscalar;
    int loc_varid_numpts   = varid_numpts;

    Vector<int> loc_varid_xi = varid_xi;
    Vector<int> loc_varid_xj = varid_xj;

    if ( src.varid_isscalar )
    {
        loc_varid_isscalar = src.varid_isscalar;
        loc_varid_numpts   = src.varid_numpts;

        loc_varid_xi = src.varid_xi;
        loc_varid_xj = src.varid_xj;
    }

    qswap(src,temp);
    qswap(*this,temp);

    varid_isscalar = loc_varid_isscalar;
    varid_numpts   = loc_varid_numpts;

    varid_xj = loc_varid_xi;
    varid_xi = loc_varid_xj;

    return;
}








// Make equation

int gentype::makeEqn(const std::string &src, int processxyzvw)
{
    deleteVectMatMem();

    // A single character is always interpretted as a string.  This
    // is done to better deal with standard SVM training datasets 
    // that by default use single characters to represent categorical
    // data.  We want to read it as such so that we can the apply
    // symbolic multiplication for inner products - that is:
    //
    // "a"*"a" = 1 (the same)
    // "a"*"b" = 0 (different)
    //
    // Exception: the variables x,y,z,v,w,X,Y,Z,V,W are reserved and
    // will not be interpretted as string if !processxyzvw

    if ( src.length() == 1 )
    {
        if ( ( src[0] == 'a' ) ||                      ( src[0] == 'o' ) || 
             ( src[0] == 'b' ) || ( src[0] == 'i' ) || ( src[0] == 'p' ) || 
             ( src[0] == 'c' ) || ( src[0] == 'j' ) || ( src[0] == 'q' ) || 
             ( src[0] == 'd' ) || ( src[0] == 'k' ) || ( src[0] == 'r' ) || 
             ( src[0] == 'e' ) || ( src[0] == 'l' ) || ( src[0] == 's' ) || 
             ( src[0] == 'f' ) || ( src[0] == 'm' ) || ( src[0] == 't' ) || 
                                  ( src[0] == 'n' ) || ( src[0] == 'u' ) ||
             ( src[0] == 'A' ) ||                      ( src[0] == 'O' ) || 
             ( src[0] == 'B' ) || ( src[0] == 'I' ) || ( src[0] == 'P' ) || 
             ( src[0] == 'C' ) || ( src[0] == 'J' ) || ( src[0] == 'Q' ) || 
             ( src[0] == 'D' ) || ( src[0] == 'K' ) || ( src[0] == 'R' ) || 
             ( src[0] == 'E' ) || ( src[0] == 'L' ) || ( src[0] == 'S' ) || 
             ( src[0] == 'F' ) || ( src[0] == 'M' ) || ( src[0] == 'T' ) || 
                                  ( src[0] == 'N' ) || ( src[0] == 'U' ) ||
             ( !processxyzvw && ( ( src[0] == 'x' ) || ( src[0] == 'X' ) ||
                                  ( src[0] == 'y' ) || ( src[0] == 'Y' ) ||
                                  ( src[0] == 'z' ) || ( src[0] == 'Z' ) ||
                                  ( src[0] == 'v' ) || ( src[0] == 'V' ) ||
                                  ( src[0] == 'w' ) || ( src[0] == 'W' ) ||
                                  ( src[0] == 'g' ) || ( src[0] == 'G' ) ||
                                  ( src[0] == 'h' ) || ( src[0] == 'H' )    ) ) )
        {
            makeString(src);

            return 0;
        }
    }

    int res;
    std::string srca;
    std::string srcb;

    // Make string "nice" by fixing +s and -s so that + is a binary operator
    // and - unary

//errstream() << "phantomx 0: " << src << "\n";
    if ( ( res = makeMathsStringNice(srca,src) ) )
    {
	makeError("Syntax error: ill-formed equation/number.");

        return res;
    }

    // Parse equation to convert operators to purely functional form

//errstream() << "phantomx 1: " << srca << "\n";
    if ( ( res = mathsparse(srcb,srca) ) )
    {
	makeError("Syntax error: unable to parse equation.");

        return res;
    }

    // Construct the equation.

//errstream() << "phantomx 2: " << srcb << "\n";
    if ( ( res = makeEqnInternal(srcb) ) )
    {
        return res;
    }

    // Simplify the equation as much as possible.

//errstream() << "phantomx 3: " << *this << "\n";
    evaluate();

//errstream() << "phantomx 4: " << *this << "\n";
    return 0;
}

int gentype::makeEqn(const char *src, int processxyzvw)
{
    std::string srcx(src);

    return makeEqn(srcx,processxyzvw);
}

























// Casting operators

gentype &gentype::toNull(gentype &res) const
{
    if ( !(res.isValNull()) )
    {
        res.deleteVectMatMem('N');
        res.typeis = 'N';
    }

    std::string errstr;

    if ( loctoNull(errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

gentype &gentype::toInteger(gentype &res) const
{
    if ( !(res.isValInteger()) )
    {
        res.deleteVectMatMem('Z');
        res.typeis = 'Z';
        res.intval = 0;
        res.doubleval = res.intval;
    }

    std::string errstr;

    if ( loctoInteger(res.intval,errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

gentype &gentype::toReal(gentype &res)    const
{
    if ( !(res.isValReal()) )
    {
        res.deleteVectMatMem('R');
        res.typeis    = 'R';
        res.doubleval = 0.0;
    }

    std::string errstr;

    if ( loctoReal(res.doubleval,errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

gentype &gentype::toAnion(gentype &res)   const
{
    if ( !(res.isValAnion()) )
    {
        res.deleteVectMatMem('A');
        res.typeis   = 'A';
        *(res.anionval) = 0.0;
    }

    std::string errstr;

    if ( loctoAnion(*(res.anionval),errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

gentype &gentype::toVector(gentype &res)  const
{
    if ( !(res.isValVector()) )
    {
        res.deleteVectMatMem('V');
        res.typeis    = 'V';
        (*res.vectorval).resize(0);
        //MEMNEW(res.vectorval,Vector<gentype>);
    }

    std::string errstr;

    if ( loctoVector(*(res.vectorval),errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

gentype &gentype::toMatrix(gentype &res)  const
{
    if ( !(res.isValMatrix()) )
    {
        res.deleteVectMatMem('M');
        res.typeis    = 'M';
        (*res.matrixval).resize(0,0);
        //MEMNEW(res.matrixval,Matrix<gentype>);
    }

    std::string errstr;

    if ( loctoMatrix(*(res.matrixval),errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

gentype &gentype::toSet(gentype &res)  const
{
    if ( !(res.isValSet()) )
    {
        res.deleteVectMatMem('X');
        res.typeis = 'X';
        (*res.setval).zero();
        //MEMNEW(res.setval,Set<gentype>);
    }

    std::string errstr;

    if ( loctoSet(*(res.setval),errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

gentype &gentype::toDgraph(gentype &res)  const
{
    if ( !(res.isValDgraph()) )
    {
        res.deleteVectMatMem('G');
        res.typeis    = 'G';
        (*res.dgraphval).zero();
        //MEMNEW(res.dgraphval,Dgraph<gentype>);
    }

    std::string errstr;

    if ( loctoDgraph(*(res.dgraphval),errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

gentype &gentype::toString(gentype &res)  const
{
    if ( !(res.isValString()) )
    {
        res.deleteVectMatMem('S');
        res.typeis    = 'S';
        *(res.stringval) = "";
    }

    std::string errstr;

    if ( loctoString(*(res.stringval),errstr) )
    {
        res.makeError(errstr);
    }

    return res;
}

void gentype::cast_null(int finalise) const
{
    std::string errstr;

    if ( !isValNull() )
    {
        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            // This will throw if finalisation doesn't result in a null
            temp.fastevaluate(tempargs,finalise);
            temp.cast_null(0);
        }

        else if ( loctoNull(errstr) )
        {
            throw errstr;
        }
    }

    return;
}

const int &gentype::cast_int(int finalise) const
{
    std::string errstr;

    if ( !isValInteger() )
    {
        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            temp.fastevaluate(tempargs,finalise);
            *NVintval() = temp.cast_int(0);
        }

        else if ( loctoInteger(*NVintval(),errstr) )
        {
            throw errstr;
        }
    }

    return intval;
}

const double &gentype::cast_double(int finalise) const
{
    std::string errstr;

    if ( !isValReal() )
    {
        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            temp.fastevaluate(tempargs,finalise);
            *NVdoubleval() = temp.cast_double(0);
        }

        else if ( loctoReal(*NVdoubleval(),errstr) )
        {
            throw errstr;
        }
    }

    return doubleval;
}

const d_anion &gentype::cast_anion(int finalise) const
{
    std::string errstr;

    if ( !isValAnion() )
    {
        if ( anionval == NULL )
        {
            MEMNEW(*NVanionval(),d_anion);
        }

        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            temp.fastevaluate(tempargs,finalise);
            **NVanionval() = temp.cast_anion(0);
        }

        else if ( loctoAnion(**NVanionval(),errstr) )
        {
            throw errstr;
        }
    }

    return *anionval;
}

const Vector<gentype> &gentype::cast_vector(int finalise) const
{
    std::string errstr;

    if ( !isValVector() )
    {
        if ( vectorval == NULL )
        {
            MEMNEW(*NVvectorval(),Vector<gentype>);
        }

        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            temp.fastevaluate(tempargs,finalise);
            **NVvectorval() = temp.cast_vector(0);
        }

        else if ( loctoVector(**NVvectorval(),errstr) )
        {
            throw errstr;
        }
    }

    return *vectorval;
}

const Vector<double> &gentype::cast_vector_real(int finalise) const
{
    cast_vector(finalise);

    int vsize = (*vectorval).size();

    if ( vectorvalreal == NULL )
    {
        MEMNEW(*NVvectorvalreal(),Vector<double>(vsize));
    }

    else
    {
        (**NVvectorvalreal()).resize(vsize);
    }

    if ( vsize )
    {
        Vector<gentype> &vg = **NVvectorval();
        Vector<double>  &vr = **NVvectorvalreal();

        int i;

        for ( i = 0 ; i < vsize ; i++ )
        {
            vr("&",i) = (double) vg(i);
        }
    }

    return *vectorvalreal;
}

const Matrix<double> &gentype::cast_matrix_real(int finalise) const
{
    cast_matrix(finalise);

    int vrows = (*matrixval).numRows();
    int vcols = (*matrixval).numCols();

    if ( matrixvalreal == NULL )
    {
        MEMNEW(*NVmatrixvalreal(),Matrix<double>(vrows,vcols));
    }

    else
    {
        (**NVmatrixvalreal()).resize(vrows,vcols);
    }

    if ( vrows && vcols )
    {
        Matrix<gentype> &vg = **NVmatrixval();
        Matrix<double>  &vr = **NVmatrixvalreal();

        int i,j;

        for ( i = 0 ; i < vrows ; i++ )
        {
            for ( j = 0 ; j < vcols ; j++ )
            {
                vr("&",i,j) = (double) vg(i,j);
            }
        }
    }

    return *matrixvalreal;
}

const Matrix<gentype> &gentype::cast_matrix(int finalise) const
{
    std::string errstr;

    if ( !isValMatrix() )
    {
        if ( matrixval == NULL )
        {
            MEMNEW(*NVmatrixval(),Matrix<gentype>);
        }

        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            temp.fastevaluate(tempargs,finalise);
            **NVmatrixval() = temp.cast_matrix(0);
        }

        else if ( loctoMatrix(**NVmatrixval(),errstr) )
        {
            throw errstr;
        }
    }

    return *matrixval;
}

const Set<gentype> &gentype::cast_set(int finalise) const
{
    std::string errstr;

    if ( !isValSet() )
    {
        if ( setval == NULL )
        {
            MEMNEW(*NVsetval(),Set<gentype>);
        }

        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            temp.fastevaluate(tempargs,finalise);
            **NVsetval() = temp.cast_set(0);
        }

        else if ( loctoSet(**NVsetval(),errstr) )
        {
            throw errstr;
        }
    }

    return *setval;
}

const Dgraph<gentype,double> &gentype::cast_dgraph(int finalise) const
{
    std::string errstr;

    if ( !isValDgraph() )
    {
        if ( dgraphval == NULL )
        {
            MEMNEW(*NVdgraphval(),xDgraph);
        }

        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            temp.fastevaluate(tempargs,finalise);
            **NVdgraphval() = temp.cast_dgraph(0);
        }

        else if ( loctoDgraph(**NVdgraphval(),errstr) )
        {
            throw errstr;
        }
    }

    return *dgraphval;
}

const std::string  &gentype::cast_string(int finalise) const
{
    std::string errstr;

    if ( !isValString() )
    {
        if ( stringval == NULL )
        {
            MEMNEW(*NVstringval(),std::string);
        }

        if ( finalise && isValEqn() )
        {
            gentype temp(*this);
            const static SparseVector<SparseVector<gentype> > tempargs;

            temp.fastevaluate(tempargs,finalise);
            **NVstringval() = temp.cast_string(0);
        }

        else if ( loctoString(**NVstringval(),errstr) )
        {
            throw errstr;
        }
    }

    return *stringval;
}

gentype &gentype::morph_null(void)
{
    if ( !isValNull() )
    {
        std::string errstr;

        if ( loctoNull(errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'N'; 

    return *this;
}

gentype &gentype::morph_int(void)
{
    if ( !isValInteger() )
    {
        std::string errstr;

        if ( loctoInteger(intval,errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'Z'; 

    return *this;
}

gentype &gentype::morph_double(void)
{
    if ( !isValReal() )
    {
        std::string errstr;

        if ( loctoReal(doubleval,errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'R'; 

    return *this;
}

gentype &gentype::morph_anion(void)
{
    if ( !isValAnion() )
    {
        if ( anionval == NULL )
        {
            MEMNEW(anionval,d_anion);
        }

        std::string errstr;

        if ( loctoAnion(*anionval,errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'A'; 

    return *this;
}

gentype &gentype::morph_vector(void)
{
    if ( !isValVector() )
    {
        if ( vectorval == NULL )
        {
            MEMNEW(vectorval,Vector<gentype>);
        }

        std::string errstr;

        if ( loctoVector(*vectorval,errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'V'; 

    return *this;
}

gentype &gentype::morph_matrix(void)
{
    if ( !isValMatrix() )
    {
        if ( matrixval == NULL )
        {
            MEMNEW(matrixval,Matrix<gentype>);
        }

        std::string errstr;

        if ( loctoMatrix(*matrixval,errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'M'; 

    return *this;
}

gentype &gentype::morph_set(void)
{
    if ( !isValSet() )
    {
        if ( setval == NULL )
        {
            MEMNEW(setval,Set<gentype>);
        }

        std::string errstr;

        if ( loctoSet(*setval,errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'X'; 

    return *this;
}

gentype &gentype::morph_dgraph(void)
{
    if ( !isValDgraph() )
    {
        if ( dgraphval == NULL )
        {
            MEMNEW(dgraphval,xDgraph);
        }

        std::string errstr;

        if ( loctoDgraph(*dgraphval,errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'G'; 

    return *this;
}

gentype &gentype::morph_string(void)
{
    if ( !isValString() )
    {
        if ( stringval == NULL )
        {
            MEMNEW(stringval,std::string);
        }

        std::string errstr;

        if ( loctoString(*stringval,errstr) )
        {
            makeError(errstr);

            return *this;
        }
    }

    typeis = 'S'; 

    return *this;
}

int gentype::loctoNull(std::string &errstr) const
{
    // Very important: don't overwrite res straight away, as it may actually
    // be a reference to ...val.

    int lociserr = 0;
    errstr = "";

    if ( !isValError() )
    {
        if ( isValNull() )
        {
            ;
        }

        else if ( isValInteger() )
	{
            errstr = "Can't cast integer to null.";
	    lociserr = 1;
	}

	else if ( isValReal() )
	{
            errstr = "Can't cast real to null.";
	    lociserr = 1;
	}

	else if ( isValAnion() )
	{
            errstr = "Can't cast anion to null.";
	    lociserr = 1;
	}

	else if ( isValVector() )
	{
            errstr = "Can't cast vector to null.";
	    lociserr = 1;
	}

	else if ( isValMatrix() )
	{
            errstr = "Can't cast matrix to null.";
	    lociserr = 1;
	}

        else if ( isValSet() )
	{
            errstr = "Can't cast set to null.";
	    lociserr = 1;
	}

        else if ( isValDgraph() )
	{
            errstr = "Can't cast dgraph to null.";
	    lociserr = 1;
	}

	else if ( isValString() )
	{
            errstr = "Can't cast string to null.";
	    lociserr = 1;
	}

	else if ( isValEqn() )
	{
            errstr = "Can't cast equation to null.";
	    lociserr = 1;
	}
    }

    else
    {
        errstr = *stringval;
	errstr += "\n";
        errstr += "Can't cast error to null.";

        lociserr = 1;
    }

    return lociserr;
}

int gentype::loctoInteger(int &res, std::string &errstr) const
{
    // Very important: don't overwrite res straight away, as it may actually
    // be a reference to ...val.

    int lociserr = 0;
    errstr = "";

    if ( !isValError() )
    {
        if ( isValNull() )
	{
            res = 0;
	}

        else if ( isValInteger() )
	{
            res = intval;
	}

	else if ( isValReal() )
	{
	    res = (int) doubleval;
	}

	else if ( isValAnion() )
	{
            if ( (*anionval).isreal() )
	    {
                res = (int) (*anionval).realpart();
	    }

	    else
	    {
		errstr = "Can't cast imaginary number to integer.";
                lociserr = 1;
	    }
	}

	else if ( isValVector() )
	{
            if ( (*vectorval).size() == 1 )
	    {
                res = (*vectorval)(zeroint()).cast_int(0);
	    }

	    else
	    {
                errstr = "Can't cast vector to int.";
                lociserr = 1;
	    }
	}

	else if ( isValMatrix() )
	{
            if ( ( (*matrixval).numRows() == 1 ) && ( (*matrixval).numCols() == 1 ) )
	    {
                res = (*matrixval)(zeroint(),zeroint()).cast_int(0);
	    }

	    else
	    {
                errstr = "Can't cast matrix to int.";
                lociserr = 1;
            }
	}

        else if ( isValSet() )
	{
            errstr = "Can't cast set to integer.";
	    lociserr = 1;
	}

        else if ( isValDgraph() )
	{
            errstr = "Can't cast dgraph to integer.";
	    lociserr = 1;
	}

	else if ( isValString() )
	{
	    errstr = "Can't cast string to integer.";
	    lociserr = 1;
	}

	else if ( isValEqn() )
	{
	    errstr = "Can't cast equation to integer.";
	    lociserr = 1;
	}
    }

    else
    {
        errstr = *stringval;
	errstr += "\n";
        errstr += "Can't cast error to integer.";

        lociserr = 1;
    }

    if ( lociserr )
    {
        res = 0;
    }

    return lociserr;
}

int gentype::loctoReal(double &res, std::string &errstr) const
{
    // Very important: don't overwrite res straight away, as it may actually
    // be a reference to ...val.

    int lociserr = 0;
    errstr = "";

    if ( !isValError() )
    {
        if ( isValNull() )
	{
            res = 0.0;
	}

        else if ( isValInteger() )
	{
	    res = (double) intval;
	}

	else if ( isValReal() )
	{
            res = doubleval;
	}

	else if ( isValAnion() )
	{
            if ( (*anionval).isreal() )
	    {
                res = (*anionval).realpart();
	    }

	    else
	    {
		errstr = "Can't cast imaginary number to real.";
                lociserr = 1;
	    }
	}

	else if ( isValVector() )
	{
            if ( (*vectorval).size() == 1 )
	    {
                res = (*vectorval)(zeroint()).cast_double(0);
	    }

	    else
	    {
                errstr = "Can't cast vector to real.";
                lociserr = 1;
	    }
	}

	else if ( isValMatrix() )
	{
            if ( ( (*matrixval).numRows() == 1 ) && ( (*matrixval).numCols() == 1 ) )
	    {
                res = (*matrixval)(zeroint(),zeroint()).cast_double(0);
	    }

	    else
	    {
                errstr = "Can't cast matrix to real.";
                lociserr = 1;
            }
	}

        else if ( isValSet() )
	{
            errstr = "Can't cast set to real.";
	    lociserr = 1;
	}

        else if ( isValDgraph() )
	{
            errstr = "Can't cast dgraph to real.";
	    lociserr = 1;
	}

	else if ( isValString() )
	{
	    errstr = "Can't cast string to real.";
	    lociserr = 1;
	}

	else if ( isValEqn() )
	{
	    errstr = "Can't cast equation to real.";
	    lociserr = 1;
	}
    }

    else
    {
        errstr = *stringval;
	errstr += "\n";
        errstr += "Can't cast error to integer.";

        lociserr = 1;
    }

    if ( lociserr )
    {
        res = 0.0;
    }

    return lociserr;
}

int gentype::loctoAnion(d_anion &res, std::string &errstr) const
{
    // Very important: don't overwrite res straight away, as it may actually
    // be a reference to ...val.

    int lociserr = 0;
    errstr = "";

    if ( !isValError() )
    {
        if ( isValNull() )
	{
            res = 0.0;
	}

        else if ( isValInteger() )
	{
            res = (double) intval;
	}

	else if ( isValReal() )
	{
            res = doubleval;
	}

	else if ( isValAnion() )
	{
            // Overwriting a with a would not be sensible

            if ( &res != anionval )
            {
                res = *anionval;
            }
	}

	else if ( isValVector() )
	{
            if ( (*vectorval).size() == 1 )
	    {
                res = (*vectorval)(zeroint()).cast_anion(0);
	    }

	    else
	    {
                errstr = "Can't cast vector to anion.";
                lociserr = 1;
	    }
	}

	else if ( isValMatrix() )
	{
            if ( ( (*matrixval).numRows() == 1 ) && ( (*matrixval).numCols() == 1 ) )
	    {
                res = (*matrixval)(zeroint(),zeroint()).cast_anion(0);
	    }

	    else
	    {
                errstr = "Can't cast matrix to anion.";
                lociserr = 1;
            }
	}

        else if ( isValSet() )
	{
            errstr = "Can't cast set to anion.";
	    lociserr = 1;
	}

        else if ( isValDgraph() )
	{
            errstr = "Can't cast dgraph to anion.";
	    lociserr = 1;
	}

	else if ( isValString() )
	{
	    errstr = "Can't cast string to anion.";
	    lociserr = 1;
	}

	else if ( isValEqn() )
	{
	    errstr = "Can't cast equation to anion.";
	    lociserr = 1;
	}
    }

    else
    {
        errstr = *stringval;
	errstr += "\n";
        errstr += "Can't cast error to integer.";

        lociserr = 1;
    }

    if ( lociserr )
    {
        res = 0.0;
    }

    return lociserr;
}

int gentype::loctoVector(Vector<gentype> &res, std::string &errstr) const
{
    // Very important: don't overwrite res straight away, as it may actually
    // be a reference to ...val.

    int lociserr = 0;
    errstr = "";

    if ( !isValError() )
    {
        if ( isValNull() )
	{
            res.resize(0);
	}

        else if ( isValInteger() || isValReal() || isValAnion() )
	{
	    res.resize(1);

	    res("&",zeroint()) = *this;
	}

	else if ( isValVector() )
	{
            // Overwriting a with a would not be sensible

            if ( &res != vectorval )
            {
                res = (*vectorval);
            }
	}

	else if ( isValMatrix() )
	{
            retVector<gentype> tmpva;

	    if ( (*matrixval).numRows() == 1 )
	    {
		res = (*matrixval)(zeroint(),tmpva);
	    }

	    else if ( (*matrixval).numCols() == 1 )
	    {
		res.resize((*matrixval).numRows());

		int i;

		if ( (*matrixval).numRows() )
		{
		    for ( i = 0 ; i < (*matrixval).numRows() ; i++ )
		    {
                        res("&",i) = (*matrixval)(i,zeroint());
		    }
		}
	    }

	    else if ( (*matrixval).numRows() || (*matrixval).numCols() )
	    {
		errstr = "Can't cast non row matrix to vector.";
                lociserr = 1;
	    }
	}

        else if ( isValSet() )
	{
            errstr = "Can't cast set to vector.";
	    lociserr = 1;
	}

        else if ( isValDgraph() )
	{
            errstr = "Can't cast dgraph to vector.";
	    lociserr = 1;
	}

	else if ( isValString() )
	{
	    errstr = "Can't cast string to vector.";
	    lociserr = 1;
	}

	else if ( isValEqn() )
	{
	    errstr = "Can't cast equation to vector.";
	    lociserr = 1;
	}
    }

    else
    {
        errstr = *stringval;
	errstr += "\n";
        errstr += "Can't cast error to integer.";

        lociserr = 1;
    }

    if ( lociserr )
    {
        res.resize(zeroint());
    }

    return lociserr;
}

int gentype::loctoMatrix(Matrix<gentype> &res, std::string &errstr) const
{
    // Very important: don't overwrite res straight away, as it may actually
    // be a reference to ...val.

    int lociserr = 0;
    res.resize(zeroint(),zeroint());

    errstr = "";

    if ( !isValError() )
    {
        if ( isValNull() )
        {
            res.resize(0,0);
        }

        else if ( isValInteger() || isValReal() || isValAnion() )
	{
	    res.resize(1,1);

	    res("&",zeroint(),zeroint()) = *this;
	}

	else if ( isValVector() )
	{
	    res.resize((*vectorval).size(),1);

	    int dessize = (*vectorval).size();

	    if ( dessize )
	    {
		int i;

		for ( i = 0 ; i < dessize ; i++ )
		{
		    res("&",i,0) = (*vectorval)(i);
		}
	    }
	}

	else if ( isValMatrix() )
	{
            // Overwriting a with a would not be sensible

            if ( &res != matrixval )
            {
                res = (*matrixval);
            }
	}

        else if ( isValSet() )
	{
            errstr = "Can't cast set to vector.";
	    lociserr = 1;
	}

        else if ( isValDgraph() )
	{
            errstr = "Can't cast dgraph to vector.";
	    lociserr = 1;
	}

	else if ( isValString() )
	{
	    errstr = "Can't cast string to matrix.";
	    lociserr = 1;
	}

	else if ( isValEqn() )
	{
	    errstr = "Can't cast equation to matrix.";
	    lociserr = 1;
	}
    }

    else
    {
        errstr = *stringval;
	errstr += "\n";
	errstr += "Can't cast error to integer.";

        lociserr = 1;
    }

    if ( lociserr )
    {
        res.resize(zeroint(),zeroint());
    }

    return lociserr;
}

int gentype::loctoSet(Set<gentype> &res, std::string &errstr) const
{
    // Very important: don't overwrite res straight away, as it may actually
    // be a reference to ...val.

    int lociserr = 0;
    errstr = "";

    if ( !isValError() )
    {
        if ( isValNull() )
	{
            res.zero();
	}

        else if ( isValInteger() || isValReal()   || isValAnion()  ||
                  isValVector()  || isValMatrix() || isValString() ||
                  isValEqn()     || isValDgraph()                     )
	{
            res.zero();
            res.add(*this);
	}

        else if ( isValSet() )
	{
            // Overwriting a with a would not be sensible

            if ( &res != setval )
            {
                res = *setval;
            }
	}
    }

    else
    {
        errstr = *stringval;
	errstr += "\n";
        errstr += "Can't cast error to set.";

        lociserr = 1;
    }

    if ( lociserr )
    {
        res.zero();
    }

    return lociserr;
}

int gentype::loctoDgraph(Dgraph<gentype,double> &res, std::string &errstr) const
{
    int lociserr = 0;
    errstr = "";

    if ( !isValError() )
    {
        if ( isValNull() )
	{
            res.zero();
	}

        else if ( isValInteger() )
	{
            errstr = "Can't cast integer to dgraph.";
	    lociserr = 1;
	}

        else if ( isValReal() )
	{
            errstr = "Can't cast real to dgraph.";
	    lociserr = 1;
	}

        else if ( isValAnion() )
	{
            errstr = "Can't cast anion to dgraph.";
	    lociserr = 1;
	}

	else if ( isValVector() )
	{
            errstr = "Can't cast vector to dgraph.";
	    lociserr = 1;
	}

	else if ( isValMatrix() )
	{
            errstr = "Can't cast matrix to dgraph.";
	    lociserr = 1;
	}

        else if ( isValSet() )
	{
            errstr = "Can't cast set to dgraph.";
	    lociserr = 1;
	}

        else if ( isValDgraph() )
	{
            // Overwriting a with a would not be sensible

            if ( &res != dgraphval )
            {
                res = *dgraphval;
            }
	}

	else if ( isValString() )
	{
            errstr = "Can't cast string to dgraph.";
	    lociserr = 1;
	}

	else if ( isValEqn() )
	{
            errstr = "Can't cast equation to dgraph.";
	    lociserr = 1;
	}
    }

    else
    {
        errstr = *stringval;
	errstr += "\n";
        errstr += "Can't cast error to set.";

        lociserr = 1;
    }

    if ( lociserr )
    {
        res.zero();
    }

    return lociserr;
}

int gentype::loctoString(std::string &res, std::string &errstr) const
{
    errstr = "";

    std::stringstream resbuffer;

    if ( isValString() )
    {
        // Overwriting a with a would not be sensible

        if ( &res != stringval )
        {
            res = *stringval;
        }
    }

    else if ( isValError() )
    {
        // Overwriting a with a would not be sensible

        if ( &res != stringval )
        {
            res = "E:";
            res += *stringval;
        }

        else
        {
            std::string temp(res);

            res = "E:";
            res += temp;
        }
    }

    else if ( isValEqn() )
    {
        // NB: static initialisation occurs the first time the code block
        //     is encountered, and is only done once.  Hence each of these
        //     indices will be looked up once and then remain unchanged.

        const static int factInd          = getfnind("fact");
        const static int negInd           = getfnind("neg");
        const static int posInd           = getfnind("pos");
        const static int lnotInd          = getfnind("lnot");
        const static int powInd           = getfnind("pow");
        const static int epowInd          = getfnind("epow");
        const static int mulInd           = getfnind("mul");
        const static int divInd           = getfnind("div");
        const static int rdivInd          = getfnind("rdiv");
        const static int modInd           = getfnind("mod");
        const static int emulInd          = getfnind("emul");
        const static int edivInd          = getfnind("ediv");
        const static int erdivInd         = getfnind("erdiv");
        const static int emodInd          = getfnind("emod");
        const static int addInd           = getfnind("add");
        const static int subInd           = getfnind("sub");
        const static int cayleyDicksonInd = getfnind("cayleyDickson");
        const static int eeqInd           = getfnind("eeq");
        const static int eneInd           = getfnind("ene");
        const static int egtInd           = getfnind("egt");
        const static int egeInd           = getfnind("ege");
        const static int eltInd           = getfnind("elt");
        const static int eleInd           = getfnind("ele");
        const static int eqInd            = getfnind("eq");
        const static int neInd            = getfnind("ne");
        const static int gtInd            = getfnind("gt");
        const static int geInd            = getfnind("ge");
        const static int ltInd            = getfnind("lt");
        const static int leInd            = getfnind("le");
        const static int lorInd           = getfnind("lor");
        const static int landInd          = getfnind("land");

        const static int varInd = getfnind("var");
        const static int VarInd = getfnind("Var");

	std::string openbracketis = "(";
	std::string separis = ",";
	std::string closebracketis = ")";

        int qw = fnnameind;
        int fnnameind = qw; // yep, you read that right.
                            // shadow so we can overwrite sans changing

    // 	            - + ~ left to right                          -a    -> neg(a)
    //                                                           ~a    -> lnot(a)
    //                                                           +a    -> pos(a)       this will never occur and can be ignored as the equation is simplified
    //	            ^ .^ (right to left)                         a^b   -> pow(a,b)
    //                                                           a.^b  -> epow(a,b)
    //	            * / \ % .* ./ .\ .% (left to right)          a*b   -> mul(a,b)
    //                                                           a/b   -> div(a,b)
    //                                                           a\b   -> rdiv(a,b)
    //                                                           a%b   -> mod(a,b)
    //                                                           a.*b  -> emul(a,b)
    //                                                           a./b  -> ediv(a,b)
    //                                                           a.\b  -> erdiv(a,b)
    //                                                           a.%b  -> emod(a,b)
    //	            + - (left to right)                          a+b   -> add(a,b)
    //                                                           a-b   -> sub(a,b)     this will never occur and can be ignored as the equation is simplified
    //	            | (cayley-dickson) left to right             a|b   -> cayleyDickson(a,b)
    //              == ~= > < >= <= .== .~= .> .< .>= .<= left to right a==b  -> eq(a,b)
    //                                                           a~=b  -> ne(a,b)
    //                                                           a>b   -> gt(a,b)
    //                                                           a>=b  -> ge(a,b)
    //                                                           a<=b  -> le(a,b)
    //                                                           a<b   -> lt(a,b)
    //                                                           a.==b -> eeq(a,b)
    //                                                           a.~=b -> ene(a,b)
    //                                                           a.>b  -> egt(a,b)
    //                                                           a.>=b -> ege(a,b)
    //                                                           a.<=b -> ele(a,b)
    //                                                           a.<b  -> elt(a,b)
    //              && !! left to right                          a||b  -> lor(a,b)
    //                                                           a&&b  -> land(a,b)

             if ( fnnameind == factInd          ) { fnnameind = 0; openbracketis = "(";  separis = "";    closebracketis = ")!"; }
        else if ( fnnameind == negInd           ) { fnnameind = 0; openbracketis = "-("; separis = "";    closebracketis = ")";  }
        else if ( fnnameind == posInd           ) { fnnameind = 0; openbracketis = "(";  separis = "";    closebracketis = ")";  }
        else if ( fnnameind == lnotInd          ) { fnnameind = 0; openbracketis = "~("; separis = "";    closebracketis = ")";  }
        else if ( fnnameind == powInd           ) { fnnameind = 0; openbracketis = "(";  separis = "^";   closebracketis = ")";  }
        else if ( fnnameind == epowInd          ) { fnnameind = 0; openbracketis = "(";  separis = ".^";  closebracketis = ")";  }
        else if ( fnnameind == mulInd           ) { fnnameind = 0; openbracketis = "(";  separis = "*";   closebracketis = ")";  }
        else if ( fnnameind == divInd           ) { fnnameind = 0; openbracketis = "(";  separis = "/";   closebracketis = ")";  }
        else if ( fnnameind == rdivInd          ) { fnnameind = 0; openbracketis = "(";  separis = "\\";  closebracketis = ")";  }
        else if ( fnnameind == modInd           ) { fnnameind = 0; openbracketis = "(";  separis = "%";   closebracketis = ")";  }
        else if ( fnnameind == emulInd          ) { fnnameind = 0; openbracketis = "(";  separis = ".*";  closebracketis = ")";  }
        else if ( fnnameind == edivInd          ) { fnnameind = 0; openbracketis = "(";  separis = "./";  closebracketis = ")";  }
        else if ( fnnameind == erdivInd         ) { fnnameind = 0; openbracketis = "(";  separis = ".\\"; closebracketis = ")";  }
        else if ( fnnameind == emodInd          ) { fnnameind = 0; openbracketis = "(";  separis = ".%";  closebracketis = ")";  }
        else if ( fnnameind == addInd           ) { fnnameind = 0; openbracketis = "(";  separis = "+";   closebracketis = ")";  }
        else if ( fnnameind == subInd           ) { fnnameind = 0; openbracketis = "(";  separis = "-";   closebracketis = ")";  }
        else if ( fnnameind == cayleyDicksonInd ) { fnnameind = 0; openbracketis = "(";  separis = "|";   closebracketis = ")";  }
        else if ( fnnameind == eeqInd           ) { fnnameind = 0; openbracketis = "(";  separis = ".=="; closebracketis = ")";  }
        else if ( fnnameind == eneInd           ) { fnnameind = 0; openbracketis = "(";  separis = ".~="; closebracketis = ")";  }
        else if ( fnnameind == egtInd           ) { fnnameind = 0; openbracketis = "(";  separis = ".>";  closebracketis = ")";  }
        else if ( fnnameind == egeInd           ) { fnnameind = 0; openbracketis = "(";  separis = ".>="; closebracketis = ")";  }
        else if ( fnnameind == eltInd           ) { fnnameind = 0; openbracketis = "(";  separis = ".<";  closebracketis = ")";  }
        else if ( fnnameind == eleInd           ) { fnnameind = 0; openbracketis = "(";  separis = ".<="; closebracketis = ")";  }
        else if ( fnnameind == eqInd            ) { fnnameind = 0; openbracketis = "(";  separis = "==";  closebracketis = ")";  }
        else if ( fnnameind == neInd            ) { fnnameind = 0; openbracketis = "(";  separis = "~=";  closebracketis = ")";  }
        else if ( fnnameind == gtInd            ) { fnnameind = 0; openbracketis = "(";  separis = ">";   closebracketis = ")";  }
        else if ( fnnameind == geInd            ) { fnnameind = 0; openbracketis = "(";  separis = ">=";  closebracketis = ")";  }
        else if ( fnnameind == ltInd            ) { fnnameind = 0; openbracketis = "(";  separis = "<";   closebracketis = ")";  }
        else if ( fnnameind == leInd            ) { fnnameind = 0; openbracketis = "(";  separis = "<=";  closebracketis = ")";  }
        else if ( fnnameind == lorInd           ) { fnnameind = 0; openbracketis = "(";  separis = "||";  closebracketis = ")";  }
        else if ( fnnameind == landInd          ) { fnnameind = 0; openbracketis = "(";  separis = "&&";  closebracketis = ")";  }

        if ( ( fnnameind == varInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 0 ) )
        {
            resbuffer << "x";
        }

        else if ( ( fnnameind == varInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 1 ) )
        {
            resbuffer << "y";
        }

        else if ( ( fnnameind == varInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 2 ) )
        {
            resbuffer << "z";
        }

        else if ( ( fnnameind == varInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 3 ) )
        {
            resbuffer << "v";
        }

        else if ( ( fnnameind == varInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 4 ) )
        {
            resbuffer << "w";
        }

        else if ( ( fnnameind == varInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 5 ) )
        {
            resbuffer << "g";
        }

        else if ( ( fnnameind == varInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 42 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 42 ) )
        {
            resbuffer << "h";
        }

        else if ( ( fnnameind == VarInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 0 ) )
        {
            resbuffer << "X";
        }

        else if ( ( fnnameind == VarInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 1 ) )
        {
            resbuffer << "Y";
        }

        else if ( ( fnnameind == VarInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 2 ) )
        {
            resbuffer << "Z";
        }

        else if ( ( fnnameind == VarInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 3 ) )
        {
            resbuffer << "V";
        }

        else if ( ( fnnameind == VarInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 4 ) )
        {
            resbuffer << "W";
        }

        else if ( ( fnnameind == VarInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 0 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 5 ) )
        {
            resbuffer << "G";
        }

        else if ( ( fnnameind == VarInd ) &&
             ( ((*eqnargs)(zeroint())).isValInteger() ) &&
             ( ((*eqnargs)(zeroint())).intval == 42 ) &&
             ( ((*eqnargs)(1)).isValInteger() ) &&
             ( ((*eqnargs)(1)).intval == 42 ) )
        {
            resbuffer << "H";
        }

        else
        {
            resbuffer << getfnname(fnnameind) << openbracketis;

            if ( (*eqnargs).size() )
            {
                int i;

                for ( i = 0 ; i < (*eqnargs).size() ; i++ )
                {
                    resbuffer << (*eqnargs)(i);

                    if ( i < (*eqnargs).size()-1 )
                    {
                        resbuffer << separis;
                    }
                }
            }

            resbuffer << closebracketis;
        }

	res = resbuffer.str();
    }

    else
    {
        resbuffer << *this;
        res = resbuffer.str();
    }

    return 0;
}













SparseVector<SparseVector<int> > &gentype::varsUsed(SparseVector<SparseVector<int> > &res) const
{
    const static int varInd = getfnind("var");
    const static int VarInd = getfnind("Var");
    const static int gvarInd = getfnind("gvar");
    const static int gVarInd = getfnind("gVar");

         if ( isValNull()    ) { ; }
    else if ( isValInteger() ) { ; }
    else if ( isValReal()    ) { ; }
    else if ( isValAnion()   ) { ; }
    else if ( isValString()  ) { ; }
    else if ( isValError()   ) { ; }

    else if ( isValVector()  )
    {
        int i;
        int xsize = size();

        if ( xsize )
        {
            for ( i = 0 ; i < xsize ; i++ )
            {
                ((*vectorval)(i)).varsUsed(res);
            }
        }
    }

    else if ( isValMatrix()  )
    {
        int i,j;
        int xrows = numRows();
        int xcols = numCols();

        if ( xrows && xcols )
        {
            for ( i = 0 ; i < xrows ; i++ )
            {
                for ( j = 0 ; j < xcols ; j++ )
                {
                    ((*matrixval)(i,j)).varsUsed(res);
                }
            }
        }
    }

    else if ( isValSet()  )
    {
        int i;
        int xsize = size();

        if ( xsize )
        {
            for ( i = 0 ; i < xsize ; i++ )
            {
                (((*setval).all())(i)).varsUsed(res);
            }
        }
    }

    else if ( isValDgraph()  )
    {
        int i;
        int xsize = size();

        if ( xsize )
        {
            for ( i = 0 ; i < xsize ; i++ )
            {
                (((*dgraphval).all())(i)).varsUsed(res);
            }
        }
    }

    else if ( ( fnnameind == varInd ) || ( fnnameind == VarInd ) )
    {
        if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() && ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
        {
            (res("&",((*eqnargs)(zeroint())).cast_int(0)))("&",((*eqnargs)(1)).cast_int(0)) = 1;
        }
    }

    else if ( ( fnnameind == gvarInd ) || ( fnnameind == gVarInd ) )
    {
        if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() && ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
        {
            (res("&",((*eqnargs)(zeroint())).cast_int(0)))("&",((*eqnargs)(1)).cast_int(0)) = 1;
        }
    }

    else
    {
        int i;

        for ( i = 0 ; i < (*eqnargs).size() ; i++ )
        {
            (*eqnargs)(i).varsUsed(res);
        }
    }

    return res;
}

SparseVector<int> &gentype::rowsUsed(SparseVector<int> &res) const
{
    const static int varInd = getfnind("var");
    const static int VarInd = getfnind("Var");

         if ( isValNull()    ) { ; }
    else if ( isValInteger() ) { ; }
    else if ( isValReal()    ) { ; }
    else if ( isValAnion()   ) { ; }
    else if ( isValString()  ) { ; }
    else if ( isValError()   ) { ; }

    else if ( isValVector()  )
    {
        int i;
        int xsize = size();

        if ( xsize )
        {
            for ( i = 0 ; i < xsize ; i++ )
            {
                ((*vectorval)(i)).rowsUsed(res);
            }
        }
    }

    else if ( isValSet()  )
    {
        int i;
        int xsize = size();

        if ( xsize )
        {
            for ( i = 0 ; i < xsize ; i++ )
            {
                (((*setval).all())(i)).rowsUsed(res);
            }
        }
    }

    else if ( isValDgraph()  )
    {
        int i;
        int xsize = size();

        if ( xsize )
        {
            for ( i = 0 ; i < xsize ; i++ )
            {
                (((*dgraphval).all())(i)).rowsUsed(res);
            }
        }
    }

    else if ( isValMatrix()  )
    {
        int i,j;
        int xrows = numRows();
        int xcols = numCols();

        if ( xrows && xcols )
        {
            for ( i = 0 ; i < xrows ; i++ )
            {
                for ( j = 0 ; j < xcols ; j++ )
                {
                    ((*matrixval)(i,j)).rowsUsed(res);
                }
            }
        }
    }

    else
    {
        if ( ( fnnameind == varInd ) || ( fnnameind == VarInd ) )
        {
            if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() )
            {
                res("&",((*eqnargs)(zeroint())).cast_int(0)) = 1;
            }
        }
    }

    return res;
}














// Other stuff

int gentype::realDeriv(const gentype &ix, const gentype &jx)
{
    int res = 0;

    const static int varInd = getfnind("var");
    const static int VarInd = getfnind("Var");
    const static int gvarInd = getfnind("gvar");
    const static int gVarInd = getfnind("gVar");
    const static int realDerivInd = getfnind("realDeriv");

    int i,j;

    if ( ix.isValVector() && jx.isValVector() )
    {
	gentype iy,jy;
	Vector<gentype> ii(ix.cast_vector(0));
	Vector<gentype> jj(jx.cast_vector(0));
        Matrix<gentype> tempres(ix.size(),jx.size());

        tempres = *this;

	for ( i = 0 ; i < ix.size() ; i++ )
	{
	    for ( j = 0 ; j < jx.size() ; j++ )
	    {
		iy = ii(i);
		jy = jj(j);
		tempres("&",i,j).realDeriv(iy,jy);
	    }
	}

        *this = tempres;
        res++;
    }

    else if ( ix.isValVector() && ( jx.isCastableToIntegerWithoutLoss() || jx.isValEqnDir() ) )
    {
        gentype iy;
	Vector<gentype> ii(ix.cast_vector(0));
	Vector<gentype> tempres(ix.size());

        tempres = *this;

	for ( i = 0 ; i < ix.size() ; i++ )
	{
	    iy = ii(i);
	    tempres("&",i).realDeriv(iy,jx);
	}

        *this = tempres;
        res++;
    }

    else if ( ( ix.isCastableToIntegerWithoutLoss() || ix.isValEqnDir() ) && jx.isValVector() )
    {
	gentype jy;
	Vector<gentype> jj(jx.cast_vector(0));
        Vector<gentype> tempres(jx.size());

        tempres = *this;

	for ( j = 0 ; j < jx.size() ; j++ )
	{
	    jy = jj(j);
	    tempres("&",j).realDeriv(ix,jy);
	}

        *this = tempres;
        res++;
    }

    else if ( ix.isCastableToIntegerWithoutLoss() && jx.isCastableToIntegerWithoutLoss() )
    {
             if ( isValNull()    ) { *this = zeroint(); }
        else if ( isValInteger() ) { *this = zeroint(); }
	else if ( isValReal()    ) { *this = zeroint(); }
	else if ( isValAnion()   ) { *this = zeroint(); }
	else if ( isValString()  ) { *this = zeroint(); }
	else if ( isValError()   ) { ; }
        else if ( isValDgraph()  ) { *this = zeroint(); }

        else if ( isValSet() )
        {
            if ( (*setval).size() )
	    {
                for ( i = 0 ; i < (*setval).size() ; i++ )
		{
                    (((*setval).ncall())("&",i)).realDeriv(ix,jx);
		}
	    }
        }

	else if ( isValVector()  )
	{
	    if ( (*vectorval).size() )
	    {
		for ( i = 0 ; i < (*vectorval).size() ; i++ )
		{
                    ((*vectorval)("&",i)).realDeriv(ix,jx);
		}
	    }
	}

	else if ( isValMatrix()  )
	{
	    if ( (*matrixval).numRows() && (*matrixval).numCols() )
	    {
		for ( i = 0 ; i < (*matrixval).numRows() ; i++ )
		{
		    for ( j = 0 ; j < (*matrixval).numCols() ; j++ )
		    {
                        ((*matrixval)("&",i,j)).realDeriv(ix,jx);
		    }
		}
	    }
	}

	else if ( isValEqnDir()  )
	{
            if ( ( fnnameind == varInd ) || ( fnnameind == VarInd ) )
	    {
		// This is a thing
                //
                // Note the assumption that we are dealing with real variables
                // here.  For completeness in terms of hypercomplex variables
                // you would need to follow more complicated rules.  It can
                // be done... put it on the to do list.
                //
                // As for derivatives with respect to vectors, matrices, ...
                // I don't know, too many ambiguities.  It will work for
                // vectors in limited cases (norms and such).

                const static gentype tempres("kronDelta(x,y)*kronDelta(z,v)");
                gentype res(tempres);

                res.substitute((*eqnargs)(zeroint()),ix,(*eqnargs)(1),jx);
                fastcopy(res,1);
                //(*this) = res((*eqnargs)(zeroint()),ix,(*eqnargs)(1),jx);
	    }

            else if ( ( ( fnnameind == gvarInd ) || ( fnnameind == gVarInd ) ) &&
                      !( ix.isValEqn() ) && !( jx.isValEqn() ) && ix.iseq((*eqnargs)(zeroint())) && jx.iseq((*eqnargs)(1)) )
	    {
                // This is like above but *only* if ix == eqnargs(0) and jx == eqnargs(1).
                // Otherwise undefined, so leave unchange.
                // (that is the difference between var and gvar)

                fastcopy(oneintgentype(),1);
	    }

            else if ( ( realDerivInd == fnnameind ) || ( fnnameind == gvarInd ) || ( fnnameind == gVarInd ) || ( fnnameind == varInd ) || ( VarInd == fnnameind ) )
            {
                const static gentype tempres("realDeriv(x,y,z)");
                gentype temp(tempres);

                temp.substitute(ix,jx,*this);
                fastcopy(temp,1);
            }

            else if ( realDerivDefinedDir() )
            {
                gentype tempres(*((*thisfninfo).realderiv));
                SparseVector<SparseVector<gentype> > temparg;

                if ( (*thisfninfo).numargs )
                {
                    j = 1;

                    for ( i = 0 ; i < (*thisfninfo).numargs ; i++ )
                    {
                        if ( (*thisfninfo).realargcopy & j )
                        {
                            temparg("&",0)("&",i).fastcopy((*eqnargs)(i),1);
                        }

                        if ( (*thisfninfo).realdrvcopy & j )
                        {
                            temparg("&",1)("&",i).fastcopy((*eqnargs)(i),1);
                            temparg("&",1)("&",i).realDeriv(ix,jx);
                        }

                        temparg("&",2)("&",0).fastcopy(ix,1);
                        temparg("&",2)("&",1).fastcopy(jx,1);

                        j *= 2;
                    }
                }

                tempres.evaluate(temparg);
                fastcopy(tempres,1);
                // *this = tempres(temparg);

            }

            else
            {
                constructError(ix,jx,*this,"Derivative not defined for this function.");

//                const static gentype tempres("realDeriv(x,y,z)");
//                gentype temp(tempres);
//
//                temp.substitute(ix,jx,*this);
//                fastcopy(temp,1);
            }
	}

	else
	{
            constructError(ix,jx,*this,"An impossible error appears to have occured in the realDeriv function.");
	}

        res++;
    }

    else if ( ( ix.isCastableToIntegerWithoutLoss() && jx.isValEqnDir()                    ) ||
              ( ix.isValEqnDir()                    && jx.isCastableToIntegerWithoutLoss() ) ||
              ( ix.isValEqnDir()                    && jx.isValEqnDir()                    )    )
    {
        const static gentype tempres("realDeriv(x,y,z)");
        gentype temp(tempres);
        
        temp.substitute(ix,jx,*this);
        fastcopy(temp,1);
        res++;
    }

    else
    {
        constructError(ix,jx,*this,"realDeriv indices must be real or vector/matrix thereof");
        res++;
    }

    return res;
}













// Internal version of substitute - function evaluation
//
// finalise: 1 means finalise randoms (indetermin 2)
//           2 means finalise globals (indetermin 1)
//           3 means both
//
// isValEqn: 1 contains indeterminant random or global parts
//           2 contains deterministic parts
//           4 contains indeterminant random parts
//           8 contains indeterminant global parts
//
// isIndeterm: 1 global function indeterminant
//             2 random indeterminant

int gentype::fastevaluate(const SparseVector<SparseVector<gentype> > &evalargs, int finalise)
{
    const static int NULLInd      = getfnind("");
    const static int varInd       = getfnind("var");
    const static int VarInd       = getfnind("Var");
    const static int gvarInd      = getfnind("gvar");
    const static int gVarInd      = getfnind("gVar");
    const static int posInd       = getfnind("pos");
    const static int negInd       = getfnind("neg");
    const static int conjInd      = getfnind("conj");
    const static int addInd       = getfnind("add");
    const static int subInd       = getfnind("sub");
    const static int mulInd       = getfnind("mul");
    const static int divInd       = getfnind("div");
    const static int rdivInd      = getfnind("rdiv");
    const static int emulInd      = getfnind("emul");
    const static int edivInd      = getfnind("ediv");
    const static int erdivInd     = getfnind("erdiv");
    const static int landInd      = getfnind("land");
    const static int lorInd       = getfnind("lor");
    const static int powInd       = getfnind("pow");
    const static int PowInd       = getfnind("Pow");
    const static int powlInd      = getfnind("powl");
    const static int PowlInd      = getfnind("Powl");
    const static int powrInd      = getfnind("powr");
    const static int PowrInd      = getfnind("Powr");
    const static int epowInd      = getfnind("epow");
    const static int EpowInd      = getfnind("Epow");
    const static int epowlInd     = getfnind("epowl");
    const static int EpowlInd     = getfnind("Epowl");
    const static int epowrInd     = getfnind("epowr");
    const static int EpowrInd     = getfnind("Epowr");
    const static int sizeInd      = getfnind("size");
    const static int numRowsInd   = getfnind("numRows");
    const static int numColsInd   = getfnind("numCols");
    const static int realDerivInd = getfnind("realDeriv");
    const static int abs1Ind      = getfnind("abs1");
    const static int abs2Ind      = getfnind("abs2");
    const static int absinfInd    = getfnind("absinf");
    const static int abspInd      = getfnind("absp");
    const static int norm1Ind     = getfnind("norm1");
    const static int norm2Ind     = getfnind("norm2");
    const static int normpInd     = getfnind("normp");

    int oldfnnameind = fnnameind;

    int i,j;
    int ccnt = 0; // counts number of changes (in principle)
    int doagain = 0; // set 1 if global function has been evaluated, so repeat required (as global function might itself return a function)
    //const fninfoblock *locthisfninfo = thisfninfo;

//errstream() << "phantomxwtf 0: " << isValEqn() << "\n";
    if ( !isValEqn() )
    {
        // Not an equation, so no need to evaluate

        ;
    }

    else if ( isValVector() )
    {
	if ( size() )
	{
	    for ( i = 0 ; i < size() ; i++ )
	    {
                ccnt += ((*vectorval)("&",i)).fastevaluate(evalargs,finalise);
	    }
	}
    }

    else if ( isValMatrix() )
    {
	if ( numRows() && numCols() )
	{
	    for ( i = 0 ; i < numRows() ; i++ )
	    {
		for ( j = 0 ; j < numCols() ; j++ )
		{
                    ccnt += ((*matrixval)("&",i,j)).fastevaluate(evalargs,finalise);
		}
	    }
	}
    }

    else if ( isValSet() )
    {
	if ( size() )
	{
	    for ( i = 0 ; i < size() ; i++ )
	    {
                ccnt += (((*setval).ncall())("&",i)).fastevaluate(evalargs,finalise);
	    }
	}
    }

    else if ( isValDgraph() )
    {
	if ( size() )
	{
	    for ( i = 0 ; i < size() ; i++ )
	    {
                ccnt += (((*dgraphval).ncall())("&",i)).fastevaluate(evalargs,finalise);
	    }
	}
    }

    else
    {
        int argsize = (*thisfninfo).numargs;
        int isInDetermin = (*thisfninfo).isInDetermin;
	int isresfunction = 0;
        int argbit = 1;
        int isresfullyeval = 1;
        int res_varid_isscalarfn = 0;
        int res_varid_numpts     = 0;
        Vector<int> res_varid_xi(1);
        Vector<int> res_varid_xj(1);

        res_varid_xi = zeroint();
        res_varid_xj = zeroint();

        if ( argsize )
	{
            for ( i = 0 ; i < argsize ; i++ )
            {
                if ( (*thisfninfo).preEvalArgs & argbit )
                {
                    ccnt += ((*eqnargs)("&",i)).fastevaluate(evalargs,finalise);
                }

                else
                {
                    isresfullyeval = 0;
                }

                if ( ((*eqnargs)(i)).scalarfn_isscalarfn() )
                {
                    if ( res_varid_isscalarfn )
                    {
                        NiceAssert( res_varid_xi == ((*eqnargs)(i)).scalarfn_i() );
                        NiceAssert( res_varid_xj == ((*eqnargs)(i)).scalarfn_j() );

                        res_varid_isscalarfn = ((*eqnargs)(i)).scalarfn_isscalarfn();
                        res_varid_numpts     = ( ((*eqnargs)(i)).scalarfn_numpts() > res_varid_numpts ) ? ((*eqnargs)(i)).scalarfn_numpts() : res_varid_numpts;

                        res_varid_xi = ((*eqnargs)(i)).scalarfn_i();
                        res_varid_xj = ((*eqnargs)(i)).scalarfn_j();
                    }

                    else
                    {
                        res_varid_isscalarfn = ((*eqnargs)(i)).scalarfn_isscalarfn();
                        res_varid_numpts     = ((*eqnargs)(i)).scalarfn_numpts();

                        res_varid_xi = ((*eqnargs)(i)).scalarfn_i();
                        res_varid_xj = ((*eqnargs)(i)).scalarfn_j();
                    }

                    if ( !( ( fnnameind == abs1Ind   ) ||
                            ( fnnameind == abs2Ind   ) ||
                            ( fnnameind == abspInd   ) ||
                            ( fnnameind == absinfInd ) ||
                            ( fnnameind == norm1Ind  ) ||
                            ( fnnameind == norm2Ind  ) ||
                            ( fnnameind == normpInd  )    ) )
                    {
                        ((*eqnargs)("&",i)).varid_isscalar = 0; // This ensures that the scalar "bit" is strictly on the outer layer - but we must leave it there for the norms to evaluate correctly!
                    }
                }

                int argtemp = 0;

                if ( ( (*thisfninfo).dirchkargs & argbit ) && ((*eqnargs)(i)).isValEqnDir() )
                {
                    isresfunction += argbit;
                }

                else if ( ( (*thisfninfo).widechkargs & argbit ) && ( argtemp = ((*eqnargs)(i)).isValEqn() ) )
                {
// finalise: 1 means finalise randoms (indetermin 2)
//           2 means finalise globals (indetermin 1)
//           3 means both
//
// isValEqn: 1 contains indeterminant random or global parts
//           2 contains deterministic parts
//           4 contains indeterminant random parts
//           8 contains indeterminant global parts
//
// isIndeterm: 1 global function indeterminant
//             2 random indeterminant

//                    if ( ( argtemp & 2 ) || !finalise || ( ( ( argtemp & 4 ) && !( finalise & 1 ) ) || ( ( argtemp & 8 ) && !( finalise & 2 ) ) ) )
//                    if ( ( argtemp & 0x10 ) || !finalise || ( ( ( argtemp & 0x04 ) && !( finalise & 0x01 ) ) || ( ( argtemp & 0x08 ) && !( finalise & 0x02 ) ) ) )

                    // Function if any of
                    //
                    // - we're not finalising
                    // - contains deterministic parts
                    // - it contains globals and we're not finalising globals
                    // - it contains randoms and we're not finalising randoms

//if ( isInDetermin & 1 )
//{
//errstream() << "phantomxaa 0: " << *this << "\n";
//errstream() << "phantomxaa 1: finalise = " << finalise << "\n";
//errstream() << "phantomxaa 2: isValEqn = " << argtemp << "\n";
//errstream() << "phantomxaa 3: isInDetermin = " << isInDetermin << "\n";
//}
                    if ( !finalise || ( argtemp & 2 ) ||
                         ( ( argtemp & 8 ) && !( finalise & 2 ) ) ||
                         ( ( argtemp & 4 ) && !( finalise & 1 ) )    )
                    {
                        // Exception: IF  the current function *is* global
                        //            AND we're finalising globals (finalise & 2)
                        //            AND the argument is random but not global ( argemp & 4 ) && !( argtemp & 8 )
                        //            then this is not a function

                        if ( ( isInDetermin == 1 ) && ( finalise & 2 ) && ( argtemp & 4 ) && !( argtemp & 8 ) )
                        {
                            ;
                        }

                        else
                        {
                            isresfunction += argbit;
                        }
                    }
                }

                argbit *= 2;
            }
	}

        (void) isresfullyeval;

        // Simplify if possible

        int issimple = 0;

        if ( isresfunction )
        {
            if ( fnnameind == negInd )
            {
                if ( ((*eqnargs)(zeroint())).isValEqnDir() )
                {
                    if ( ((*eqnargs)(zeroint())).fnnameind == negInd )
                    {
                        // *this = (*(((*eqnargs)(zeroint())).eqnargs))(zeroint());
                        switcheroo((*(((*eqnargs)("&",0)).eqnargs))("&",0));

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( fnnameind == addInd )
            {
                if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(zeroint())).cast_int(0) == 0 )
                    {
                        // *this = ((*eqnargs)(1));
                        switcheroo((*eqnargs)("&",1));

                        issimple = 1;
                        ccnt++;
                    }
                }

                else if ( ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(1)).cast_int(0) == 0 )
                    {
                        // *this = ((*eqnargs)(zeroint()));
                        switcheroo(((*eqnargs)("&",0)));

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( fnnameind == subInd )
            {
                if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(zeroint())).cast_int(0) == 0 )
                    {
                        // *this = ((*eqnargs)(1));
                        switcheroo((*eqnargs)("&",1));
                        OP_neg(*this);

                        issimple = 1;
                        ccnt++;
                    }
                }

                else if ( ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(1)).cast_int(0) == 0 )
                    {
                        // *this = ((*eqnargs)(zeroint()));
                        switcheroo(((*eqnargs)("&",0)));

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( ( fnnameind == mulInd ) || ( fnnameind == emulInd ) )
            {
                if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(zeroint())).cast_int(0) == 0 )
                    {
                        *this = zeroint();

                        issimple = 1;
                        ccnt++;
                    }

                    else if ( ((*eqnargs)(zeroint())).cast_int(0) == 1 )
                    {
                        // *this = ((*eqnargs)(1));
                        switcheroo(((*eqnargs)("&",1)));

                        issimple = 1;
                        ccnt++;
                    }
                }

                else if ( ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(1)).cast_int(0) == 0 )
                    {
                        *this = zeroint();
    
                        issimple = 1;
                        ccnt++;
                    }

                    else if ( ((*eqnargs)(1)).cast_int(0) == 1 )
                    {
                        // *this = ((*eqnargs)(zeroint()));
                        switcheroo(((*eqnargs)("&",0)));

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( ( fnnameind == divInd ) || ( fnnameind == edivInd ) )
            {
                if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(zeroint())).cast_int(0) == 0 )
                    {
                        *this = zeroint();

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( ( fnnameind == rdivInd ) || ( fnnameind == erdivInd ) )
            {
                if ( ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(1)).cast_int(0) == 0 )
                    {
                        *this = zeroint();

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( fnnameind == landInd )
            {
                if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(zeroint())).cast_int(0) == 0 )
                    {
                        *this = zeroint();

                        issimple = 1;
                        ccnt++;
                    }
                }

                else if ( ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(1)).cast_int(0) == 0 )
                    {
                        *this = zeroint();

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( fnnameind == lorInd )
            {
                if ( ((*eqnargs)(zeroint())).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(zeroint())).cast_int(0) == 1 )
                    {
                        *this = 1;

                        issimple = 1;
                        ccnt++;
                    }
                }

                else if ( ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(1)).cast_int(0) == 1 )
                    {
                        *this = 1;

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( ( fnnameind == powInd   ) || ( fnnameind == PowInd   ) ||
                      ( fnnameind == powlInd  ) || ( fnnameind == PowlInd  ) ||
                      ( fnnameind == powrInd  ) || ( fnnameind == PowrInd  ) ||
                      ( fnnameind == epowInd  ) || ( fnnameind == EpowInd  ) ||
                      ( fnnameind == epowlInd ) || ( fnnameind == EpowlInd ) ||
                      ( fnnameind == epowrInd ) || ( fnnameind == EpowrInd )    )
            {
                if ( ((*eqnargs)(1)).isCastableToIntegerWithoutLoss() )
                {
                    if ( ((*eqnargs)(1)).cast_int(0) == 0 )
                    {
                        *this = 1;

                        issimple = 1;
                        ccnt++;
                    }

                    else if ( ((*eqnargs)(1)).cast_int(0) == 1 )
                    {
                        // *this = ((*eqnargs)(zeroint()));
                        switcheroo(((*eqnargs)("&",0)));

                        issimple = 1;
                        ccnt++;
                    }
                }
            }

            else if ( ( ( fnnameind == abs2Ind   ) ||
                        ( fnnameind == abs1Ind   ) ||
                        ( fnnameind == absinfInd ) ||
                        ( fnnameind == norm2Ind  ) ||
                        ( fnnameind == norm1Ind  )    ) && (*eqnargs)(zeroint()).scalarfn_isscalarfn() )
            {
                *this = ((*thisfninfo).fn1arg)((*eqnargs)(zeroint()));

                if ( fnnameind != oldfnnameind )
                {
                    return fastevaluate(evalargs,finalise);
                }
            }

            else if ( ( ( fnnameind == abspInd   ) ||
                        ( fnnameind == normpInd  )    ) && (*eqnargs)(zeroint()).scalarfn_isscalarfn() && (*eqnargs)(1).isCastableToRealWithoutLoss() )
            {
                *this = ((*thisfninfo).fn2arg)((*eqnargs)(zeroint()),(*eqnargs)(1));

                if ( fnnameind != oldfnnameind )
                {
                    return fastevaluate(evalargs,finalise);
                }
            }
        }

//        if ( issimple || isresfunction || ( isInDetermin && !finalise ) )
//errstream() << "phantomx 3fgfg: " << isInDetermin << "\n";
//errstream() << "phantomx 4fgfg: " << finalise << "\n";

// finalise: 1 means finalise randoms (indetermin 2)
//           2 means finalise globals (indetermin 1)
//           3 means both
//
// isValEqn: 1 contains indeterminant random or global parts
//           2 contains deterministic parts
//           4 contains indeterminant random parts
//           8 contains indeterminant global parts
//
// isIndeterm: 1 global function indeterminant
//             2 random indeterminant

//        if ( issimple || isresfunction || ( ( ( isInDetermin & 2 ) && !( finalise & 1 ) ) || ( ( isInDetermin & 1 ) && !( finalise & 2 ) ) ) )

//if ( isInDetermin & 1 ) 
//{ 
//errstream() << "phantomxaa 7: uissimple = " << issimple << "\n"; 
//errstream() << "phantomxaa 7: isfunction = " << isresfunction << "\n"; 
//errstream() << "phantomxaa 7: isInDetermin = " << isInDetermin << "\n"; 
//errstream() << "phantomxaa 7: finalise = " << finalise << "\n"; 
//}
        if ( issimple || isresfunction || ( ( isInDetermin & 2 ) && !( finalise & 1 ) ) || ( ( isInDetermin & 1 ) && !( finalise & 2 ) ) )
        {
            ;
        }

        else if ( fnnameind == NULLInd    ) { switcheroo((*eqnargs)("&",0));           ccnt++; }
        else if ( fnnameind == posInd     ) { switcheroo((*eqnargs)("&",0));           ccnt++; }
        else if ( fnnameind == conjInd    ) { switcheroo((*eqnargs)("&",0)); conj();   ccnt++; }
        else if ( fnnameind == sizeInd    ) { *this = (*eqnargs)(zeroint()).size();    ccnt++; }
        else if ( fnnameind == numRowsInd ) { *this = (*eqnargs)(zeroint()).numRows(); ccnt++; }
        else if ( fnnameind == numColsInd ) { *this = (*eqnargs)(zeroint()).numCols(); ccnt++; }

        else if ( fnnameind == varInd     ) { ccnt += OP_var(evalargs);         }
        else if ( fnnameind == VarInd     ) { ccnt += OP_var(evalargs); conj(); }
        else if ( fnnameind == gvarInd    ) { ccnt += OP_var(evalargs);         }
        else if ( fnnameind == gVarInd    ) { ccnt += OP_var(evalargs); conj(); }

        else if ( fnnameind == realDerivInd )
        {
            if ( ((*eqnargs)(2)).realDerivDefinedDir() )
            {
                gentype ti,tj;

                ti.fastcopy((*eqnargs)(zeroint()),1);
                tj.fastcopy((*eqnargs)(1),1);

                switcheroo((*eqnargs)("&",2));
                ccnt += realDeriv(ti,tj);
            }

            else
            {
                ;
            }
//FIXME phantomxy: if evalargs is nonempty then need to subst at this point
//FIXME: need fast copy below
        }

        else if ( argsize == 0 ) { ccnt++; doagain = (*thisfninfo).isInDetermin & 1; oldfnnameind = fnnameind; *this = ((*thisfninfo).fn0arg)();                                                                                            }
        else if ( argsize == 1 ) { ccnt++; doagain = (*thisfninfo).isInDetermin & 1; oldfnnameind = fnnameind; *this = ((*thisfninfo).fn1arg)((*eqnargs)(zeroint()));                                                                       }
        else if ( argsize == 2 ) { ccnt++; doagain = (*thisfninfo).isInDetermin & 1; oldfnnameind = fnnameind; *this = ((*thisfninfo).fn2arg)((*eqnargs)(zeroint()),(*eqnargs)(1));                                                         }
        else if ( argsize == 3 ) { ccnt++; doagain = (*thisfninfo).isInDetermin & 1; oldfnnameind = fnnameind; *this = ((*thisfninfo).fn3arg)((*eqnargs)(zeroint()),(*eqnargs)(1),(*eqnargs)(2));                                           }
        else if ( argsize == 4 ) { ccnt++; doagain = (*thisfninfo).isInDetermin & 1; oldfnnameind = fnnameind; *this = ((*thisfninfo).fn4arg)((*eqnargs)(zeroint()),(*eqnargs)(1),(*eqnargs)(2),(*eqnargs)(3));                             }
        else if ( argsize == 5 ) { ccnt++; doagain = (*thisfninfo).isInDetermin & 1; oldfnnameind = fnnameind; *this = ((*thisfninfo).fn5arg)((*eqnargs)(zeroint()),(*eqnargs)(1),(*eqnargs)(2),(*eqnargs)(3),(*eqnargs)(4));               }
        else if ( argsize == 6 ) { ccnt++; doagain = (*thisfninfo).isInDetermin & 1; oldfnnameind = fnnameind; *this = ((*thisfninfo).fn6arg)((*eqnargs)(zeroint()),(*eqnargs)(1),(*eqnargs)(2),(*eqnargs)(3),(*eqnargs)(4),(*eqnargs)(5)); }

        else
        {
            std::string locfnname(getfnname(fnnameind));

            constructError(*this,locfnname+" is not a recognised function name");
        }

        //else if ( ( argsize == 1 ) && ( (*thisfninfo).OP_fn1arg != NULL ) ) { *this = (*eqnargs)(zeroint()); ((*locthisfninfo).OP_fn1arg)(*this);                                                                       }
        //else if ( ( argsize == 2 ) && ( (*thisfninfo).OP_fn2arg != NULL ) ) { *this = (*eqnargs)(zeroint()); ((*locthisfninfo).OP_fn2arg)(*this,(*eqnargs)(1));                                                         }
        //else if ( ( argsize == 3 ) && ( (*thisfninfo).OP_fn3arg != NULL ) ) { *this = (*eqnargs)(zeroint()); ((*locthisfninfo).OP_fn3arg)(*this,(*eqnargs)(1),(*eqnargs)(2));                                           }
        //else if ( ( argsize == 4 ) && ( (*thisfninfo).OP_fn4arg != NULL ) ) { *this = (*eqnargs)(zeroint()); ((*locthisfninfo).OP_fn4arg)(*this,(*eqnargs)(1),(*eqnargs)(2),(*eqnargs)(3));                             }
        //else if ( ( argsize == 5 ) && ( (*thisfninfo).OP_fn5arg != NULL ) ) { *this = (*eqnargs)(zeroint()); ((*locthisfninfo).OP_fn5arg)(*this,(*eqnargs)(1),(*eqnargs)(2),(*eqnargs)(3),(*eqnargs)(4));               }
        //else if ( ( argsize == 6 ) && ( (*thisfninfo).OP_fn5arg != NULL ) ) { *this = (*eqnargs)(zeroint()); ((*locthisfninfo).OP_fn5arg)(*this,(*eqnargs)(1),(*eqnargs)(2),(*eqnargs)(3),(*eqnargs)(4),(*eqnargs)(5)); }

        if ( res_varid_isscalarfn )
        {
            scalarfn_setisscalarfn(1);
            scalarfn_setnumpts(res_varid_numpts);

            scalarfn_seti(res_varid_xi);
            scalarfn_setj(res_varid_xj);
        }
    }

    if ( doagain && ( finalise & 2 ) && isValEqnDir() && ( ((*thisfninfo).isInDetermin) & 1 ) && ( oldfnnameind != fnnameind ) )
    {
        ccnt += fastevaluate(evalargs,finalise);
    }

    if ( scalarfn_isscalarfn() && !isValEqn() )
    {
        scalarfn_setisscalarfn(0);
    }

    return ccnt;
}




















// Helper functions:

/*
gentype &gentype::ident(void)
{
    *this = 1; return *this;

    //     if ( isValNull()    ) { ;                      }
    //else if ( isValInteger() ) { intval = 1;            }
    //else if ( isValReal()    ) { doubleval = 1;         }
    //else if ( isValAnion()   ) { (*anionval) = 1.0;     }
    //else if ( isValVector()  ) { (*vectorval).ident();  }
    //else if ( isValMatrix()  ) { (*matrixval).ident();  }
    //else if ( isValSet()     ) { (*setval).ident();     }
    //else if ( isValDgraph()  ) { (*dgraphval).ident();  }
    //else if ( isValString()  ) { (*stringval) = "";     }
    //else if ( isValError()   ) { ;                      }
    //else                       { *this = 1;             }
    //
    //return *this;
}

gentype &gentype::zero(void)
{
    *this = 0; return *this;

    //     if ( isValNull()    ) { ;                      }
    //else if ( isValInteger() ) { intval = 0;            }
    //else if ( isValReal()    ) { doubleval = 0;         }
    //else if ( isValAnion()   ) { (*anionval) = 0.0;     }
    //else if ( isValVector()  ) { (*vectorval).zero();   }
    //else if ( isValMatrix()  ) { (*matrixval).zero();   }
    //else if ( isValSet()     ) { (*setval).zero();      }
    //else if ( isValDgraph()  ) { (*dgraphval).zero();   }
    //else if ( isValString()  ) { (*stringval) = "";     }
    //else if ( isValError()   ) { ;                      }
    //else                       { *this = zeroint();     }
    //
    //return *this;
}
*/

void gentype::reversestring(void)
{
    int i;

    NiceAssert( stringval );

    if ( (*stringval).length() )
    {
        std::string temp = *stringval;
        *stringval = "";

	for ( i = (int) temp.length()-1 ; i >= 0 ; i-- )
	{
            *stringval += temp[i];
	}
    }

    return;
}                        

void gentype::invertstringcase(void)
{
    int i;

    NiceAssert( stringval );

    if ( (*stringval).length() )
    {
        for ( i = 0 ; i < (int) (*stringval).length() ; i++ )
        {
            switch ( (*stringval)[i] )
	    {
            case 'a': { (*stringval)[i] = 'A'; break; }
            case 'b': { (*stringval)[i] = 'B'; break; }
            case 'c': { (*stringval)[i] = 'C'; break; }
            case 'd': { (*stringval)[i] = 'D'; break; }
            case 'e': { (*stringval)[i] = 'E'; break; }
            case 'f': { (*stringval)[i] = 'F'; break; }
            case 'g': { (*stringval)[i] = 'G'; break; }
            case 'h': { (*stringval)[i] = 'H'; break; }
            case 'i': { (*stringval)[i] = 'I'; break; }
            case 'j': { (*stringval)[i] = 'J'; break; }
            case 'k': { (*stringval)[i] = 'K'; break; }
            case 'l': { (*stringval)[i] = 'L'; break; }
            case 'm': { (*stringval)[i] = 'M'; break; }
            case 'n': { (*stringval)[i] = 'N'; break; }
            case 'o': { (*stringval)[i] = 'O'; break; }
            case 'p': { (*stringval)[i] = 'P'; break; }
            case 'q': { (*stringval)[i] = 'Q'; break; }
            case 'r': { (*stringval)[i] = 'R'; break; }
            case 's': { (*stringval)[i] = 'S'; break; }
            case 't': { (*stringval)[i] = 'T'; break; }
            case 'u': { (*stringval)[i] = 'U'; break; }
            case 'v': { (*stringval)[i] = 'V'; break; }
            case 'w': { (*stringval)[i] = 'W'; break; }
            case 'x': { (*stringval)[i] = 'X'; break; }
            case 'y': { (*stringval)[i] = 'Y'; break; }
            case 'z': { (*stringval)[i] = 'Z'; break; }
            case 'A': { (*stringval)[i] = 'a'; break; }
            case 'B': { (*stringval)[i] = 'b'; break; }
            case 'C': { (*stringval)[i] = 'c'; break; }
            case 'D': { (*stringval)[i] = 'd'; break; }
            case 'E': { (*stringval)[i] = 'e'; break; }
            case 'F': { (*stringval)[i] = 'f'; break; }
            case 'G': { (*stringval)[i] = 'g'; break; }
            case 'H': { (*stringval)[i] = 'h'; break; }
            case 'I': { (*stringval)[i] = 'i'; break; }
            case 'J': { (*stringval)[i] = 'j'; break; }
            case 'K': { (*stringval)[i] = 'k'; break; }
            case 'L': { (*stringval)[i] = 'l'; break; }
            case 'M': { (*stringval)[i] = 'm'; break; }
            case 'N': { (*stringval)[i] = 'n'; break; }
            case 'O': { (*stringval)[i] = 'o'; break; }
            case 'P': { (*stringval)[i] = 'p'; break; }
            case 'Q': { (*stringval)[i] = 'q'; break; }
            case 'R': { (*stringval)[i] = 'r'; break; }
            case 'S': { (*stringval)[i] = 's'; break; }
            case 'T': { (*stringval)[i] = 't'; break; }
            case 'U': { (*stringval)[i] = 'u'; break; }
            case 'V': { (*stringval)[i] = 'v'; break; }
            case 'W': { (*stringval)[i] = 'w'; break; }
            case 'X': { (*stringval)[i] = 'x'; break; }
            case 'Y': { (*stringval)[i] = 'y'; break; }
            case 'Z': { (*stringval)[i] = 'z'; break; }
	    default: { break; }
	    }
	}
    }

    return;
}






















// Maths parsing

// Go through string left to right
// Break into a vector of chunk strings - each is either a number, an
// expression or an operator group

int gentype::mathsparse(std::string &srcx, const std::string &src)
{
    int res = 0;
    int AShilton = 0;

    srcx = src;

    // Break src into blocks.  Basically takes srcx and
    // divides it into sequential blocks.  Each block i is described
    // by element i in the vectors:
    //
    // srcxblock(i):     relevant part of srcx
    // srcxblocktype(i): type, where 0 means opcode block, 1 means number,
    //                   2 means expression
    // srcxblockres(i):  for numbers 1 = int, 2 = real, 3 = anion, 4 = vector,
    //                   5 = matrix, 6 = string, 7 = error, 8 = set, 9 = dgraph
    //                   for expressions this is the number of arguments in
    //                   the expression
    // srcxblockname(i): for expressions this is the function name of the block

    std::string exprname = "";
    std::string opblock  = "";
    Vector<int> commapos;
    int i,j,k;

    Vector<eqninfoblock> srcxblock;

    i = 0;
    j = 0;
    k = 0;

    while ( i < (int) srcx.length() )
    {
        res = processNumLtoR(i,j,srcx);

	if ( res == -1 )
	{
//errstream() << "phantomxy 0\n";
            return -1;
	}

	else if ( res )
	{
	    if ( opblock.length() )
	    {
		srcxblock.add(k);

		srcxblock("&",k).text = opblock;
		srcxblock("&",k).type = 0;

		k++;
                opblock = "";
	    }

	    srcxblock.add(k);

	    srcxblock("&",k).text = srcx.substr(i,j-i+1);
	    srcxblock("&",k).type = 1;
	    srcxblock("&",k).res  = res;

	    k++;
            i = j+1;
	}

	else
	{
            res = processExprLtoR(i,j,AShilton,srcx,exprname,commapos);

	    if ( res == -1 )
	    {
//errstream() << "phantomxy 1\n";
		return -1;
	    }

	    else if ( res )
	    {
		if ( opblock.length() )
		{
		    srcxblock.add(k);

		    srcxblock("&",k).text = opblock;
		    srcxblock("&",k).type = 0;

		    k++;
		    opblock = "";
		}

                commapos -= i;

		srcxblock.add(k);

		srcxblock("&",k).text   = srcx.substr(i,j-i+1);
		srcxblock("&",k).type   = 2;
		srcxblock("&",k).res    = res-1;
		srcxblock("&",k).fnname = exprname;
		srcxblock("&",k).commas = commapos;
                srcxblock("&",k).isstr  = AShilton;

		k++;
		i = j+1;
	    }

	    else
	    {
		opblock += srcx[i];
                i++;
	    }
	}
    }

    if ( opblock.length() )
    {
	srcxblock.add(k);

	srcxblock("&",k).text = opblock;
	srcxblock("&",k).type = 0;

	k++;
	opblock = "";
    }

    // Quick sanity check - operators and numbers/expressions should alternate

    if ( srcxblock.size() > 1 )
    {
	int prevblocktype = srcxblock(zeroint()).type;

	for ( i = 1 ; i < srcxblock.size() ; i++ )
	{
	    if ( ( prevblocktype && srcxblock(i).type ) || ( !prevblocktype && !srcxblock(i).type ) )
	    {
//errstream() << "phantomxy 2\n";
                return -1;
	    }

            prevblocktype = srcxblock(i).type;
	}
    }

    else if ( srcxblock.size() == 1 )
    {
	if ( !srcxblock(zeroint()).type )
	{
	    // If there is a single block then it must be a number or an expression, not an operator block

//errstream() << "phantomxy 3\n";
	    return -1;
	}
    }

    else
    {
        // Empty is meaningless

//errstream() << "phantomxy 4\n";
        return -1;
    }

    // Run through all strings and convert to expressions
    //
    // null -> null()
    // nan,NaN,... -> vnan()
    // inf -> pinf()
    // pinf -> pinf()
    // pi -> pi()
    // euler -> euler()
    // x -> var(0,0)
    // y -> var(0,1)
    // z -> var(0,2)
    // v -> var(0,3)
    // w -> var(0,4)
    // g -> var(0,5)
    // h -> var(2,0)

    for ( i = 0 ; i < srcxblock.size() ; i++ )
    {
	if ( srcxblock(i).isstr )
	{
            if ( srcxblock(i).fnname == "null" )
	    {
                srcxblock("&",i).text  = "null()";
		srcxblock("&",i).isstr = 0;
	    }

            else if ( ( srcxblock(i).fnname == "nan" ) || ( srcxblock(i).fnname == "NAN" ) ||
                      ( srcxblock(i).fnname == "Nan" ) || ( srcxblock(i).fnname == "nAn" ) || ( srcxblock(i).fnname == "naN" ) ||
                      ( srcxblock(i).fnname == "nAN" ) || ( srcxblock(i).fnname == "Nan" ) || ( srcxblock(i).fnname == "NAn" )    )
	    {
		srcxblock("&",i).text  = "vnan()";
		srcxblock("&",i).isstr = 0;
	    }

            else if ( ( srcxblock(i).fnname == "inf" ) || ( srcxblock(i).fnname == "INF" ) ||
                      ( srcxblock(i).fnname == "Inf" ) || ( srcxblock(i).fnname == "iNf" ) || ( srcxblock(i).fnname == "inF" ) ||
                      ( srcxblock(i).fnname == "iNF" ) || ( srcxblock(i).fnname == "InF" ) || ( srcxblock(i).fnname == "INf" )    )
	    {
		srcxblock("&",i).text  = "pinf()";
		srcxblock("&",i).isstr = 0;
	    }

            else if ( srcxblock(i).fnname == "pinf" )
	    {
		srcxblock("&",i).text  = "pinf()";
		srcxblock("&",i).isstr = 0;
	    }

            else if ( srcxblock(i).fnname == "ninf" )
	    {
		srcxblock("&",i).text  = "pinf()";
		srcxblock("&",i).isstr = 0;
	    }

            else if ( srcxblock(i).fnname == "pi" )
	    {
		srcxblock("&",i).text  = "pi()";
		srcxblock("&",i).isstr = 0;
	    }

	    else if ( srcxblock(i).fnname == "euler" )
	    {
		srcxblock("&",i).text  = "euler()";
                srcxblock("&",i).isstr = 0;
	    }

	    else if ( srcxblock(i).fnname == "x" )
	    {
		srcxblock("&",i).text   = "var(0,0)";
		srcxblock("&",i).fnname = "var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

	    else if ( srcxblock(i).fnname == "y" )
	    {
		srcxblock("&",i).text   = "var(0,1)";
		srcxblock("&",i).fnname = "var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

	    else if ( srcxblock(i).fnname == "z" )
	    {
		srcxblock("&",i).text   = "var(0,2)";
		srcxblock("&",i).fnname = "var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

	    else if ( srcxblock(i).fnname == "v" )
	    {
		srcxblock("&",i).text   = "var(0,3)";
		srcxblock("&",i).fnname = "var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

	    else if ( srcxblock(i).fnname == "w" )
	    {
		srcxblock("&",i).text   = "var(0,4)";
		srcxblock("&",i).fnname = "var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

	    else if ( srcxblock(i).fnname == "g" )
	    {
		srcxblock("&",i).text   = "var(0,5)";
		srcxblock("&",i).fnname = "var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

	    else if ( srcxblock(i).fnname == "h" )
	    {
		srcxblock("&",i).text   = "var(42,42)";
		srcxblock("&",i).fnname = "var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

            else if ( srcxblock(i).fnname == "X" )
	    {
                srcxblock("&",i).text   = "Var(0,0)";
                srcxblock("&",i).fnname = "Var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

            else if ( srcxblock(i).fnname == "Y" )
	    {
                srcxblock("&",i).text   = "Var(0,1)";
                srcxblock("&",i).fnname = "Var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

            else if ( srcxblock(i).fnname == "Z" )
	    {
                srcxblock("&",i).text   = "Var(0,2)";
                srcxblock("&",i).fnname = "Var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

            else if ( srcxblock(i).fnname == "V" )
	    {
                srcxblock("&",i).text   = "Var(0,3)";
                srcxblock("&",i).fnname = "Var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

            else if ( srcxblock(i).fnname == "W" )
	    {
                srcxblock("&",i).text   = "Var(0,4)";
                srcxblock("&",i).fnname = "Var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

            else if ( srcxblock(i).fnname == "G" )
	    {
                srcxblock("&",i).text   = "Var(0,5)";
                srcxblock("&",i).fnname = "Var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }

            else if ( srcxblock(i).fnname == "H" )
	    {
                srcxblock("&",i).text   = "Var(42,42)";
                srcxblock("&",i).fnname = "Var";
		srcxblock("&",i).isstr  = 0;

		(srcxblock("&",i).commas).resize(1);
                (srcxblock("&",i).commas)("&",zeroint()) = 5;
	    }
	}
    }

    // Now that the srcx has been broken down into blocks we can start
    // processing the operators according to the usual rules of precedence.

    // But first, we need to process any expressions

    std::string subblock;
    int m,n;

    for ( i = 0 ; i < (srcxblock.size()) ; i++ )
    {
	if ( ( srcxblock(i).type == 2 ) && ( srcxblock(i).res > 0 ) )
	{
	    for ( j = (srcxblock(i).res)-1 ; j >= 0 ; j-- )
	    {
		if ( j == 0 )
		{
		    if ( srcxblock(i).res == 1 )
		    {
			m = (int) ((srcxblock(i).fnname).length())+1;
			n = (int) ((srcxblock(i).text).length())-2;
		    }

		    else
		    {
			m = (int) ((srcxblock(i).fnname).length())+1;
			n = ((srcxblock(i).commas)(j))-1;
		    }
		}

		else if ( j == (srcxblock(i).res)-1 )
		{
                    m = ((srcxblock(i).commas)(j-1))+1;
                    n = (int) ((srcxblock(i).text).length())-2;
		}

		else
		{
                    m = ((srcxblock(i).commas)(j-1))+1;
                    n = ((srcxblock(i).commas)(j))-1;
		}

		if ( ( res = mathsparse(subblock,(srcxblock(i).text).substr(m,n-m+1)) ) )
		{
//errstream() << "phantomxy 5\n";
                    return res;
		}

		(srcxblock("&",i).text).erase(m,n-m+1);
                (srcxblock("&",i).text).insert(m,subblock);
	    }
	}
    }

    // Finally up to operators.  Order is as follows
    //
    //              ! right to left                              a!    -> fact(a)
    // 	            - + ~ left to right                          -a    -> neg(a)
    //                                                           ~a    -> lnot(a)
    //                                                           +a    -> pos(a)       this will never occur and can be ignored as the equation is simplified
    //	            ^ .^ (right to left)                         a^b   -> pow(a,b)
    //                                                           a.^b  -> epow(a,b)
    //	            * / \ % .* ./ .\ .% (left to right)          a*b   -> mul(a,b)
    //                                                           a/b   -> div(a,b)
    //                                                           a\b   -> rdiv(a,b)
    //                                                           a%b   -> mod(a,b)
    //                                                           a.*b  -> emul(a,b)
    //                                                           a./b  -> ediv(a,b)
    //                                                           a.\b  -> erdiv(a,b)
    //                                                           a.%b  -> emod(a,b)
    //	            + - (left to right)                          a+b   -> add(a,b)
    //                                                           a-b   -> sub(a,b)     this will never occur and can be ignored as the equation is simplified
    //	            | (cayley-dickson) left to right             a|b   -> cayleyDickson(a,b)
    //              .== .~= .> .< .>= .<= == ~= > < >= <= left to right a==b  -> eq(a,b)
    //                                                           a~=b  -> ne(a,b)
    //                                                           a>b   -> gt(a,b)
    //                                                           a>=b  -> ge(a,b)
    //                                                           a<=b  -> le(a,b)
    //                                                           a<b   -> lt(a,b)
    //                                                           a.==b -> eeq(a,b)
    //                                                           a.~=b -> ene(a,b)
    //                                                           a.>b  -> egt(a,b)
    //                                                           a.>=b -> ege(a,b)
    //                                                           a.<=b -> ele(a,b)
    //                                                           a.<b  -> elt(a,b)
    //              && !! left to right                          a||b  -> lor(a,b)
    //                                                           a&&b  -> land(a,b)

    Vector<std::string> opSymb;
    Vector<std::string> opFuncEquiv;

    opSymb.resize(1);        opSymb("&",zeroint())      = "!";
    opFuncEquiv.resize(1);   opFuncEquiv("&",zeroint()) = "fact";

    if ( ( res = operatorToFunction(1,0,opSymb,opFuncEquiv,srcxblock) ) ) { return res; }

    opSymb.resize(2);        opSymb("&",zeroint())      = "~";    opSymb("&",1)      = "-";
    opFuncEquiv.resize(2);   opFuncEquiv("&",zeroint()) = "lnot"; opFuncEquiv("&",1) = "neg";

    if ( ( res = operatorToFunction(0,0,opSymb,opFuncEquiv,srcxblock) ) ) { return res; }

    opSymb.resize(2);        opSymb("&",zeroint())      = ".^";   opSymb("&",1)      = "^";
    opFuncEquiv.resize(2);   opFuncEquiv("&",zeroint()) = "epow"; opFuncEquiv("&",1) = "pow";

    if ( ( res = operatorToFunction(1,1,opSymb,opFuncEquiv,srcxblock) ) ) { return res; }

    opSymb.resize(8);        opSymb("&",zeroint())      = ".*";   opSymb("&",1)      = "./";   opSymb("&",2)      = ".\\";   opSymb("&",3)      = ".%";   opSymb("&",4)      = "*";   opSymb("&",5)      = "/";   opSymb("&",6)      = "\\";   opSymb("&",7)      = "%";
    opFuncEquiv.resize(8);   opFuncEquiv("&",zeroint()) = "emul"; opFuncEquiv("&",1) = "ediv"; opFuncEquiv("&",2) = "erdiv"; opFuncEquiv("&",3) = "emod"; opFuncEquiv("&",4) = "mul"; opFuncEquiv("&",5) = "div"; opFuncEquiv("&",6) = "rdiv"; opFuncEquiv("&",7) = "mod";

    if ( ( res = operatorToFunction(0,1,opSymb,opFuncEquiv,srcxblock) ) ) { return res; }

    opSymb.resize(1);        opSymb("&",zeroint())      = "+";
    opFuncEquiv.resize(1);   opFuncEquiv("&",zeroint()) = "add";

    if ( ( res = operatorToFunction(0,1,opSymb,opFuncEquiv,srcxblock) ) ) { return res; }

    opSymb.resize(1);        opSymb("&",zeroint())      = "|";
    opFuncEquiv.resize(1);   opFuncEquiv("&",zeroint()) = "cayleyDickson";

    if ( ( res = operatorToFunction(0,1,opSymb,opFuncEquiv,srcxblock) ) ) { return res; }

    opSymb.resize(12);       opSymb("&",zeroint())      = ".=="; opSymb("&",1)      = ".~="; opSymb("&",2)      = ".>="; opSymb("&",3)      = ".<="; opSymb("&",4)      = ".>";  opSymb("&",5)      = ".<";  opSymb("&",6)      = "=="; opSymb("&",7)      = "~="; opSymb("&",8)      = ">="; opSymb("&",9)      = "<="; opSymb("&",10)      = ">";  opSymb("&",11)      = "<";
    opFuncEquiv.resize(12);  opFuncEquiv("&",zeroint()) = "eeq"; opFuncEquiv("&",1) = "ene"; opFuncEquiv("&",2) = "ege"; opFuncEquiv("&",3) = "ele"; opFuncEquiv("&",4) = "egt"; opFuncEquiv("&",5) = "elt"; opFuncEquiv("&",6) = "eq"; opFuncEquiv("&",7) = "ne"; opFuncEquiv("&",8) = "ge"; opFuncEquiv("&",9) = "le"; opFuncEquiv("&",10) = "gt"; opFuncEquiv("&",11) = "lt";

    if ( ( res = operatorToFunction(0,1,opSymb,opFuncEquiv,srcxblock) ) ) { return res; }

    opSymb.resize(2);        opSymb("&",zeroint())      = "||";  opSymb("&",1)      = "&&";
    opFuncEquiv.resize(2);   opFuncEquiv("&",zeroint()) = "lor"; opFuncEquiv("&",1) = "land";

    if ( ( res = operatorToFunction(0,1,opSymb,opFuncEquiv,srcxblock) ) ) { return res; }

    // Sanity check - should only have one block left

    if ( srcxblock.size() != 1 )
    {
//errstream() << "phantomxy 6\n";
	return -1;
    }

    // Finally, we need to reconstruct srcx from the blocks

    srcx = srcxblock(zeroint()).text;

//errstream() << "phantomxy 7\n";
    return res;
}

// NB: matrices and vectors (and in fact numbers in general) are
// parsed back to operator>>, which then calls down for individual
// function elements to get back to here.


void subeforE(std::string &src)
{
    if ( src.length() )
    {
        unsigned int i;

        for ( i = 0 ; i < src.length() ; i++ )
        {
            if ( src[i] == 'E' )
            {
                src[i] = 'e';
            }
        }
    }

    return;
}


// This is a very naive function parser that assumes that the
// function is either a number or in the form
//
// fnname(a,b,...)
//
// where fnname is purely alphanumeric starting with alpha, and
// a,b,... are in the same format.  It does check that each function
// gets the correct number of arguments.  It is simple because it
// is purely recursive and can "hang off" the input stream operator
// if it finds a number.  It assumes brackets are correctly paired.
// The only real complexities is that it needs to check if the string
// is simply "i", "I", "J", .. etc for anions, and [ for vectors, and
// M: for matricies

int gentype::makeEqnInternal(const std::string &src)
{
    deleteVectMatMem();

    int res;
    int i,m,n;
    int AShilton = 0;

    typeis = 'F';

    res = processNumLtoR(0,i,src);

    if ( res == -1 )
    {
	makeError("Syntax error: illformed number.");

	return -1;
    }

    else if ( res == 1 )
    {
        int zres;
	std::stringstream resbuffer;

	resbuffer << src;
	resbuffer >> zres;

        *this = zres;
    }

    else if ( res == 2 )
    {
        // Must deal with E to e substitution first

        std::string locsrc(src);

        subeforE(locsrc);

        double rres;
	std::stringstream resbuffer;

        resbuffer << locsrc;
	resbuffer >> rres;

        *this = rres;
    }

    else if ( res == 3 )
    {
        // Must deal with E to e substitution first

        std::string locsrc(src);

        subeforE(locsrc);

        d_anion ares;
	std::stringstream resbuffer;

        resbuffer << locsrc;
	resbuffer >> ares;

        *this = ares;
    }

    else if ( res == 4 )
    {
        Vector<gentype> vres;
	std::stringstream resbuffer;

	resbuffer << src;
	resbuffer >> vres;

        *this = vres;
    }

    else if ( res == 5 )
    {
        Matrix<gentype> mres;
	std::stringstream resbuffer;

	i = 0;

	while ( src[i] != '[' )
	{
	    i++;

            NiceAssert( i < (int) src.length() );
	}

	resbuffer << src.substr(i,(src.length())-i);
	resbuffer >> mres;

        *this = mres;
    }

    else if ( res == 6 )
    {
        makeString(src.substr(1,src.length()-2));
    }

    else if ( res == 7 )
    {
	i = 0;

	while ( src[i] != '\"' )
	{
	    i++;

            NiceAssert( i < (int) src.length() );
	}

	makeError(src.substr(i+1,src.length()-i-2));
    }

    else if ( res == 8 )
    {
        Set<gentype> sres;
	std::stringstream resbuffer;

	resbuffer << src;
        resbuffer >> sres;

        *this = sres;
    }

    else if ( res == 9 )
    {
        Dgraph<gentype,double> dgres;
	std::stringstream resbuffer;

	i = 0;

        while ( src[i] != '{' )
	{
	    i++;

            NiceAssert( i < (int) src.length() );
	}

	resbuffer << src.substr(i,(src.length())-i);
        resbuffer >> dgres;

        *this = dgres;
    }

    else
    {
	Vector<int> commapos;
        std::string locfnname;

        res = processExprLtoR(0,i,AShilton,src,locfnname,commapos);

        fnnameind = getfnind(locfnname);

        if ( ( res == -1 ) || !res || AShilton )
	{
	    makeError("Syntax error: illformed expression.");

	    return -1;
	}

        res--;

        if ( ( thisfninfo = getfninfo(fnnameind) ) == NULL )
        {
            std::string locfnname(getfnname(fnnameind));

            makeError("Syntax error: no function "+locfnname+" found.");

            return -1;
        }

	// Check number of arguments is correct

        if ( res != (*thisfninfo).numargs )
        {
            std::string locfnname(getfnname(fnnameind));

            makeError("Syntax error: wrong number of arguments for "+locfnname+".");

            return -1;
        }

	// Process arguments

	MEMNEW(eqnargs,Vector<gentype>(res));

	if ( res )
	{
	    int resb;

	    for ( i = 0 ; i < res ; i++ )
	    {
		if ( i == 0 )
		{
		    if ( res == 1 )
		    {
                        m = (int) (locfnname.length())+1;
			n = (int) (src.length())-2;
		    }

		    else
		    {
                        m = (int) (locfnname.length())+1;
			n = (commapos(i))-1;
		    }
		}

		else if ( i == res-1 )
		{
		    m = (commapos(i-1))+1;
		    n = (int) (src.length())-2;
		}

		else
		{
		    m = (commapos(i-1))+1;
		    n = (commapos(i))-1;
		}

                resb = ((*eqnargs)("&",i)).makeEqnInternal(src.substr(m,n-m+1));

		if ( resb )
		{
		    std::string errstris = ((*eqnargs)(i)).cast_string(0);

                    makeError(errstris);

                    return resb;
		}
	    }
	}
    }

    return 0;
}






















// Variable evaluation function - replace this with variable, if available

// Global variables.  As a rule var(i,j) is a local variable, but using
// these functions you can turn it into a global variable.  Global 
// variables are detected and evaluated immediately, taking precedence.
//
// setGlobalVar:   turns var(i,j) into global variable with value val
// getGloalVar:    returns value of global variable var(i,j) (NULL if not present)
// testGlobalVar:  return 1 if var(i,j) is global
// unsetGlobalVar: make var(i,j) non-global
//
// mode = 0: set
//        1: get
//        2: test if present
//        3: unset

void getsetunsetGlobalVar(int i, int j, int mode, const gentype &val, gentype &res, int &ires);
void getsetunsetGlobalVar(int i, int j, int mode, const gentype &val, gentype &res, int &ires)
{
    res.force_null();

    ires = 0;

    static SparseVector<SparseVector<gentype> > globvars;

    switch ( mode )
    {
        case 0:
        {
            svmvolatile static svm_mutex eyelock;
            svm_mutex_lock(eyelock);

            globvars("&",i)("&",j) = val;

            svm_mutex_unlock(eyelock);

            break;
        }

        case 1:
        {
            svmvolatile static svm_mutex eyelock;
            svm_mutex_lock(eyelock);

            // Short-circuit logic
            if ( globvars.isindpresent(i) && globvars(i).isindpresent(j) )
            {
                res = globvars(i)(j);
            }

            svm_mutex_unlock(eyelock);

            break;
        }

        case 2:
        {
            //commented for speed svmvolatile static svm_mutex eyelock;
            //commented for speed svm_mutex_lock(eyelock);

            // Short-circuit logic
            ires = ( globvars.isindpresent(i) && globvars(i).isindpresent(j) );

            //commented for speed svm_mutex_unlock(eyelock);

            break;
        }

        default:
        {
            NiceAssert( mode == 3 );

            svmvolatile static svm_mutex eyelock;
            svm_mutex_lock(eyelock);

            // Short-circuit logic
            if ( globvars.isindpresent(i) && globvars(i).isindpresent(j) )
            {
                globvars("&",i).zero(j);
            }

            svm_mutex_unlock(eyelock);

            break;
        }
    }

    return;
}

void setGlobalVar(int i, int j, const gentype &val)
{
    int ires = 0;
    gentype yyy;

    getsetunsetGlobalVar(i,j,0,val,yyy,ires);

    return;
}

void getGlobalVar(int i, int j, gentype &res)
{
    int ires = 0;
    const gentype xxx;

    getsetunsetGlobalVar(i,j,1,xxx,res,ires);

    return;
}

int testGlobalVar(int i, int j)
{
    int ires = 0;
    const gentype xxx;
    gentype yyy;

    getsetunsetGlobalVar(i,j,2,xxx,yyy,ires);

    return ires;
}

void unsetGlobalVar(int i, int j)
{
    int ires = 0;
    const gentype xxx;
    gentype yyy;

    getsetunsetGlobalVar(i,j,3,xxx,yyy,ires);

    return;
}

// Global functions.  This function table is access by fnB(i,arg) in
// gentype and allows gentype to build on top of pretty much anything

void defaultfnaddrE(int i, int j, gentype &res, const gentype &xa, int ia, const gentype &xb, int ib);
void defaultfnaddrE(int i, int j, gentype &res, const gentype &xa, int ia, const gentype &xb, int ib)
{
    (void) i;
    (void) j;
    (void) xa;
    (void) ia;
    (void) xb;
    (void) ib;

    res.force_null();

    return;
}

void edefaultfnaddrE(int i, int j, Vector<gentype> &eres, const Vector<gentype> &exa, int ia, const Vector<gentype> &exb, int ib);
void edefaultfnaddrE(int i, int j, Vector<gentype> &eres, const Vector<gentype> &exa, int ia, const Vector<gentype> &exb, int ib)
{
    (void) i;
    (void) j;
    (void) exa;
    (void) ia;
    (void) exb;
    (void) ib;

    eres.resize(0);

    return;
}

GenFunc &setzero(GenFunc &x);
GenFunc &setzero(GenFunc &x)
{
    x = defaultfnaddrE;

    return x;
}

void qswap(GenFunc &a, GenFunc &b);
void qswap(GenFunc &a, GenFunc &b)
{
    GenFunc c(a);

    a = b;
    b = c;

    return;
}

eGenFunc &setzero(eGenFunc &x);
eGenFunc &setzero(eGenFunc &x)
{
    x = edefaultfnaddrE;

    return x;
}

void qswap(eGenFunc &a, eGenFunc &b);
void qswap(eGenFunc &a, eGenFunc &b)
{
    eGenFunc c(a);

    a = b;
    b = c;

    return;
}

void getsetunsetgenFunc(int i, int j, int mode, const GenFunc fnaddr, const gentype &xa, int ia, const gentype &xb, int ib, gentype &res, int &ires);
void getsetunsetegenFunc(int i, int j, int mode, const eGenFunc fnaddr, const Vector<gentype> &exa, int ia, const Vector<gentype> &exb, int ib, Vector<gentype> &eres, int &ires);

void setGenFunc(const GenFunc fnaddr)
{
    gentype res;
    int ires = 0;

    const gentype xa;
    int ia = 0;
    const gentype xb;
    int ib = 0;

    getsetunsetgenFunc(0,0,0,fnaddr,xa,ia,xb,ib,res,ires);

    return;
}

void evalgenFunc(int i, int j, const gentype &xa, int ia, const gentype &xb, int ib, gentype &res)
{
    int ires = 0;

    getsetunsetgenFunc(i,j,1,defaultfnaddrE,xa,ia,xb,ib,res,ires);

    return;
}

void getsetunsetgenFunc(int i, int j, int mode, const GenFunc fnaddr, const gentype &xa, int ia, const gentype &xb, int ib, gentype &res, int &ires)
{
    res.force_null();

    ires = 0;

    static GenFunc funcTable = defaultfnaddrE;

    switch ( mode )
    {
        case 0:
        {
            funcTable = fnaddr;

            break;
        }

        default:
        {
            NiceAssert( mode == 1 );

            (*funcTable)(i,j,res,xa,ia,xb,ib);

            break;
        }
    }

    return;
}

void seteGenFunc(const eGenFunc fnaddr)
{
    Vector<gentype> res;
    int ires = 0;

    const Vector<gentype> xa;
    int ia = 0;
    const Vector<gentype> xb;
    int ib = 0;

    getsetunsetegenFunc(0,0,0,fnaddr,xa,ia,xb,ib,res,ires);

    return;
}

void evalegenFunc(int i, int j, const Vector<gentype> &exa, int ia, const Vector<gentype> &exb, int ib, Vector<gentype> &eres)
{
    int ires = 0;

    getsetunsetegenFunc(i,j,1,edefaultfnaddrE,exa,ia,exb,ib,eres,ires);

    return;
}

void getsetunsetegenFunc(int i, int j, int mode, const eGenFunc fnaddr, const Vector<gentype> &exa, int ia, const Vector<gentype> &exb, int ib, Vector<gentype> &eres, int &ires)
{
    NiceAssert( exa.size() == exb.size() );

    eres.resize(exa.size());

    int k;

    for ( k = 0 ; k < exa.size() ; k++ )
    {
        eres("&",k).force_null();
    }

    ires = 0;

    static eGenFunc funcTable = edefaultfnaddrE;

    switch ( mode )
    {
        case 0:
        {
            funcTable = fnaddr;

            break;
        }

        default:
        {
            NiceAssert( mode == 1 );

            (*funcTable)(i,j,eres,exa,ia,exb,ib);

            break;
        }
    }

    return;
}










gentype gentype::var(const SparseVector<SparseVector<gentype> > &evalargs, const Vector<gentype> &argres) const
{
    int i,j;
    gentype res;

    if ( argres(zeroint()).isValVector() && argres(1).isValVector() )
    {
	Vector<gentype> locargres(argres);
        Vector<gentype> ii(argres(zeroint()).cast_vector(0));
	Vector<gentype> jj(argres(1).cast_vector(0));
        Matrix<gentype> tempres(argres(zeroint()).size(),argres(1).size());

        for ( i = 0 ; i < argres(zeroint()).size() ; i++ )
	{
	    for ( j = 0 ; j < argres(1).size() ; j++ )
	    {
		locargres("&",0) = ii(i);
		locargres("&",1) = jj(j);
		tempres("&",i,j) = var(evalargs,locargres);
	    }
	}

        res = tempres;
    }

    else if ( argres(zeroint()).isValVector() )
    {
	Vector<gentype> locargres(argres);
        Vector<gentype> ii(argres(zeroint()).cast_vector(0));
        Vector<gentype> tempres(argres(zeroint()).size());

        locargres("&",1) = argres(1);

        for ( i = 0 ; i < argres(zeroint()).size() ; i++ )
	{
	    locargres("&",0) = ii(i);
	    tempres("&",i) = var(evalargs,locargres);
	}

        res = tempres;
    }

    else if ( argres(1).isValVector() )
    {
	Vector<gentype> locargres(argres);
	Vector<gentype> jj(argres(1).cast_vector(0));
        Vector<gentype> tempres(argres(1).size());

        locargres("&",0) = argres(zeroint());

	for ( j = 0 ; j < argres(1).size() ; j++ )
	{
	    locargres("&",1) = jj(j);
            tempres("&",j) = var(evalargs,locargres);
	}

        res = tempres;
    }

    else if ( argres(zeroint()).isCastableToIntegerWithoutLoss() && argres(1).isCastableToIntegerWithoutLoss() )
    {
	i = argres(0).cast_int(0);
	j = argres(1).cast_int(0);

        if ( testGlobalVar(i,j) )
        {
            getGlobalVar(i,j,res);
        }

	else if ( (evalargs.isindpresent(i)) )
	{
            // We don't want to touch (evalargs)(i) unless we know that
            // it is present, as touching this will automatically cause
            // it's creation otherwise, which we do not want.

            if ( ((evalargs)(i).isindpresent(j)) )
	    {
		res = evalargs(i)(j);
	    }

	    else
	    {
                nearcopy(res,argres);
	    }
	}

	else
	{
	    nearcopy(res,argres);
	}
    }

    else if ( argres(0).isValEqn() || argres(1).isValEqn() )
    {
        std::string locfnname(getfnname(fnnameind));

        res = locfnname+"(x,y)"; // could be var or Var
        res = res(argres(0),argres(1));
    }

    else
    {
	constructError(argres(0),argres(1),res,"Variable indices must be integer (or matrix/vector thereof).");
    }

    return res;
}

int gentype::OP_var(const SparseVector<SparseVector<gentype> > &evalargs)
{
    int res = 0;
    int i,j;

    // Assumption here: this function == var

    if ( (*eqnargs)(zeroint()).isValVector() && (*eqnargs)(1).isValVector() )
    {
        Vector<gentype> locargres((*eqnargs));
        Vector<gentype> ii((*eqnargs)(zeroint()).cast_vector(0));
        Vector<gentype> jj((*eqnargs)(1).cast_vector(0));
        Matrix<gentype> tempres((*eqnargs)(zeroint()).size(),(*eqnargs)(1).size());

        for ( i = 0 ; i < (*eqnargs)(zeroint()).size() ; i++ )
	{
            for ( j = 0 ; j < (*eqnargs)(1).size() ; j++ )
	    {
		locargres("&",0) = ii(i);
		locargres("&",1) = jj(j);
		tempres("&",i,j) = var(evalargs,locargres);
	    }
	}

        res = 1;
        *this = tempres;
    }

    else if ( (*eqnargs)(zeroint()).isValVector() )
    {
        Vector<gentype> locargres((*eqnargs));
        Vector<gentype> ii((*eqnargs)(zeroint()).cast_vector(0));
        Vector<gentype> tempres((*eqnargs)(zeroint()).size());

        locargres("&",1) = (*eqnargs)(1);

        for ( i = 0 ; i < (*eqnargs)(zeroint()).size() ; i++ )
	{
	    locargres("&",0) = ii(i);
	    tempres("&",i) = var(evalargs,locargres);
	}

        res = 1;
        *this = tempres;
    }

    else if ( (*eqnargs)(1).isValVector() )
    {
        Vector<gentype> locargres((*eqnargs));
        Vector<gentype> jj((*eqnargs)(1).cast_vector(0));
        Vector<gentype> tempres((*eqnargs)(1).size());

        locargres("&",0) = (*eqnargs)(zeroint());

        for ( j = 0 ; j < (*eqnargs)(1).size() ; j++ )
	{
	    locargres("&",1) = jj(j);
            tempres("&",j) = var(evalargs,locargres);
	}

        res = 1;
        *this = tempres;
    }

    else if ( (*eqnargs)(zeroint()).isCastableToIntegerWithoutLoss() && (*eqnargs)(1).isCastableToIntegerWithoutLoss() )
    {       
        i = (*eqnargs)(zeroint()).cast_int(0);
        j = (*eqnargs)(1).cast_int(0);

        if ( testGlobalVar(i,j) )
        {
            res = 1;
            getGlobalVar(i,j,*this);
        }

	else if ( (evalargs.isindpresent(i)) )
	{
            // We don't want to touch (evalargs)(i) unless we know that
            // it is present, as touching this will automatically cause
            // it's creation otherwise, which we do not want.

            if ( ((evalargs)(i).isindpresent(j)) )
	    {
                res = 1;
                fastcopy(evalargs(i)(j),1);
	    }
	}
    }

    else if ( (*eqnargs)(zeroint()).isValEqn() || (*eqnargs)(1).isValEqn() )
    {
        ;
    }

    else
    {
        constructError((*eqnargs)(zeroint()),(*eqnargs)(1),*this,"Variable indices must be integer (or matrix/vector thereof).");
    }

    return res;
}

// Given a string and a starting brace, find matching end brace
// and set end to relevant position.  LRorRL is 0 for left to right,
// 1 for right to left.  Returns 0 on success, 1 on failure.

int pairBrackets(int start, int &end, const std::string &src, int LRorRL)
{
    NiceAssert( start >= 0 );
    NiceAssert( start < (int) src.length() );

//errstream() << "phantomxyza 0\n";
    if ( !LRorRL && !( ( src[start] == '(' ) || ( src[start] == '[' ) || ( src[start] == '{' ) || ( src[start] == '\"' ) ) )
    {
        end = start;
//errstream() << "phantomxyza 1\n";
        return 1;
    }

    if (  LRorRL && !( ( src[start] == ')' ) || ( src[start] == ']' ) || ( src[start] == '}' ) || ( src[start] == '\"' ) ) )
    {
        end = start;
//errstream() << "phantomxyza 2\n";
        return 1;
    }

    int curvecount = 0;
    int squarecount = 0;
    int curlycount = 0;
    int quotecount = 0;
    int justgo = 1;
    char charlast = ' ';

    end = start;

//errstream() << "phantomxyza 3\n";
    while ( justgo || curvecount || squarecount || curlycount || quotecount )
    {
	justgo = 0;
//errstream() << "phantomxyza 4\n";

	if ( ( end < 0 ) || ( end >= (int) src.length() ) )
	{
	    end = start;
//errstream() << "phantomxyza 4b\n";
            return 1;
	}

	if ( src[end] == '(' && !quotecount ) { curvecount++; }
	if ( src[end] == ')' && !quotecount ) { curvecount--; }

	if ( src[end] == '[' && !quotecount ) { squarecount++; }
	if ( src[end] == ']' && !quotecount ) { squarecount--; }

	if ( src[end] == '{' && !quotecount ) { curlycount++; }
	if ( src[end] == '}' && !quotecount ) { curlycount--; }

//errstream() << "phantomxyza 5: " << src[end] << "," << charlast << "\n";
	if ( ( src[end] == '\"' ) && ( charlast != '\\' ) ) { quotecount = quotecount ? 0 : 1; }

	if ( !LRorRL && ( ( curvecount < 0 ) || ( squarecount < 0 ) || ( curlycount < 0 ) ) )
	{
	    end = start;
//errstream() << "phantomxyza 6\n";
            return 1;
	}

	if (  LRorRL && ( ( curvecount > 0 ) || ( squarecount > 0 ) || ( curlycount > 0 ) ) )
	{
	    end = start;
//errstream() << "phantomxyza 7\n";
            return 1;
	}

	charlast = src[end];
//errstream() << "phantomxyza 8: " << src[end] << "," << charlast << "\n";

	if ( !LRorRL ) { end++; }
	else           { end--; }
    }

    if ( !LRorRL ) { end--; }
    else           { end++; }

//errstream() << "phantomxyza 9\n";
    return 0;
}

// Given a string src and a starting point start, working from left
// to right, return NZ if a number is located at this point and 0
// otherwise (-1 for error).  A number can be any gentype that isn't
// an equation - that is: an integer, real, anion, vector, matrix of
// string:
//
// Integers: a                    a is a sequence of {0123456789}
// Reals:    a{.b}{[e or E]{-}c}  a is a sequence of {0123456789}
//                                b is a sequence of {0123456789}
//                                c is a sequence of {0123456789}
//           .b{[e or E]{-}c}     b is a sequence of {0123456789}
//                                c is a sequence of {0123456789}
// Anions:   {r}s                 r is a real number
//                                s is one of {iIJKlmnopqr}
// Vectors:  [ ... ]              the relevant end ] can be detected by pairing of brackets
// Matrices: M:[ ... ]            the relevant end ] can be detected by pairing of brackets
// Sets:     { ... }              the relevant end } can be detected by pairing of brackets
// Dgraph:   G:{ ... }            the relevant end } can be detected by pairing of brackets
// String:   "..."                the second quote not preceeded by \ is the end quote
//
// return:
//
// 0 if not a number
// 1 if integer
// 2 if real
// 3 if anion
// 4 if vector
// 5 if matrix
// 6 if string
// 7 if an error
// 8 if set
// 9 if dgraph
// -1 if a syntax error occurs.
//
// Note that we don't include any preceeding -, as this unary operator
// must be dealt with in correct order, for example to ensure that -3! = -6

int processNumLtoR(int start, int &end, const std::string &src)
{
    int res = 0;

    NiceAssert( start >= 0 );
    NiceAssert( start < (int) src.length() );

    end = start;

    // Intentionally not using else statements here to allow fall-through

//errstream() << "phantomxyz 0\n";
    if ( !res && ( ( src[start] == 'i' ) || ( src[start] == 'I' ) || ( src[start] == 'J' ) ||
                   ( src[start] == 'K' ) || ( src[start] == 'l' ) || ( src[start] == 'm' ) ||
		   ( src[start] == 'n' ) || ( src[start] == 'o' ) || ( src[start] == 'p' ) ||
		   ( src[start] == 'q' ) || ( src[start] == 'r' )                             ) )
    {
	if ( end == ((int) src.length())-1 )
	{
            res = 3;
	}

	else
	{
            if ( !(  ( src[end+1] == 'a' ) || ( src[end+1] == 'b' ) || ( src[end+1] == 'c' ) || ( src[end+1] == 'd' ) || ( src[end+1] == 'e' ) || ( src[end+1] == 'f' ) || ( src[end+1] == 'g' ) ||
                     ( src[end+1] == 'h' ) || ( src[end+1] == 'i' ) || ( src[end+1] == 'j' ) || ( src[end+1] == 'k' ) || ( src[end+1] == 'l' ) || ( src[end+1] == 'm' ) || ( src[end+1] == 'n' ) ||
                     ( src[end+1] == 'o' ) || ( src[end+1] == 'p' ) || ( src[end+1] == 'q' ) || ( src[end+1] == 'r' ) || ( src[end+1] == 's' ) || ( src[end+1] == 't' ) || ( src[end+1] == 'u' ) ||
                     ( src[end+1] == 'v' ) || ( src[end+1] == 'w' ) || ( src[end+1] == 'x' ) || ( src[end+1] == 'y' ) || ( src[end+1] == 'z' ) || ( src[end+1] == '_' ) ||
                     ( src[end+1] == 'A' ) || ( src[end+1] == 'B' ) || ( src[end+1] == 'C' ) || ( src[end+1] == 'D' ) || ( src[end+1] == 'E' ) || ( src[end+1] == 'F' ) || ( src[end+1] == 'G' ) ||
                     ( src[end+1] == 'H' ) || ( src[end+1] == 'I' ) || ( src[end+1] == 'J' ) || ( src[end+1] == 'K' ) || ( src[end+1] == 'L' ) || ( src[end+1] == 'M' ) || ( src[end+1] == 'N' ) ||
                     ( src[end+1] == 'O' ) || ( src[end+1] == 'P' ) || ( src[end+1] == 'Q' ) || ( src[end+1] == 'R' ) || ( src[end+1] == 'S' ) || ( src[end+1] == 'T' ) || ( src[end+1] == 'U' ) ||
	             ( src[end+1] == 'V' ) || ( src[end+1] == 'W' ) || ( src[end+1] == 'X' ) || ( src[end+1] == 'Y' ) || ( src[end+1] == 'Z' ) || ( src[end+1] == '(' ) ||
	              ( ( end != start ) && ( ( src[end+1] == '0' ) || ( src[end+1] == '1' ) || ( src[end+1] == '2' ) || ( src[end+1] == '3' ) || ( src[end+1] == '4' ) || ( src[end+1] == '5' ) ||
		           		      ( src[end+1] == '6' ) || ( src[end+1] == '7' ) || ( src[end+1] == '8' ) || ( src[end+1] == '9' ) ) ) ) )
	    {
		res = 3;
	    }
	}
    }

    if ( !res && ( ( src[start] == '0' ) || ( src[start] == '1' ) || ( src[start] == '2' ) ||
                   ( src[start] == '3' ) || ( src[start] == '4' ) || ( src[start] == '5' ) ||
		   ( src[start] == '6' ) || ( src[start] == '7' ) || ( src[start] == '8' ) ||
		   ( src[start] == '9' )                                                      ) )
    {
	// This might be an integer.  Begin by finding end of numeric block

	res = 1;

	while ( ( src[end] == '0' ) || ( src[end] == '1' ) || ( src[end] == '2' ) ||
	      	( src[end] == '3' ) || ( src[end] == '4' ) || ( src[end] == '5' ) ||
		( src[end] == '6' ) || ( src[end] == '7' ) || ( src[end] == '8' ) ||
		( src[end] == '9' )                                                  )
	{
	    end++;

	    if ( end >= (int) src.length() )
	    {
                break;
	    }
	}

	end--;

        // If numeric block is followed by . e E i I J K l m n o p q r then
        // this is not a numeric block

	if ( end+1 < (int) src.length() )
	{
	    if ( src[end+1] == '.' )
	    {
                // Need to know if this is an operator (eg .*) or part of a
                // real (eg .0)
		end++;
                end++;

		if ( end >= (int) src.length() )
		{
		    // . should never occur at the end of an equation

		    end = start-1;
		    return -1;
		}

		if ( ( src[end] == '0' ) || ( src[end] == '1' ) || ( src[end] == '2' ) ||
		     ( src[end] == '3' ) || ( src[end] == '4' ) || ( src[end] == '5' ) ||
		     ( src[end] == '6' ) || ( src[end] == '7' ) || ( src[end] == '8' ) ||
		     ( src[end] == '9' )                                                  )
		{
		    // Not an integer, might be a real

		    res = 0;
		    end = start;
		}

		else
		{
		    // . operator after an integer

		    end--;
		    end--;
		}
	    }

            else if ( ( src[end+1] == 'e' ) || ( src[end+1] == 'E' ) || ( src[end+1] == 'i' ) || 
                      ( src[end+1] == 'I' ) || ( src[end+1] == 'J' ) || ( src[end+1] == 'K' ) ||
                      ( src[end+1] == 'l' ) || ( src[end+1] == 'm' ) || ( src[end+1] == 'n' ) ||
                      ( src[end+1] == 'o' ) || ( src[end+1] == 'p' ) || ( src[end+1] == 'q' ) ||
                      ( src[end+1] == 'r' )    )
	    {
                res = 0;
		end = start;
	    }
	}
    }

    if ( !res && ( ( src[start] == '0' ) || ( src[start] == '1' ) || ( src[start] == '2' ) ||
                   ( src[start] == '3' ) || ( src[start] == '4' ) || ( src[start] == '5' ) ||
		   ( src[start] == '6' ) || ( src[start] == '7' ) || ( src[start] == '8' ) ||
		   ( src[start] == '9' ) || ( src[start] == '.' )                             ) )
    {
	// This might be a real or an anion.  Begin by finding end of the first numeric block

	res = 2;

	if ( src[end] != '.' )
	{
	    while ( ( src[end] == '0' ) || ( src[end] == '1' ) || ( src[end] == '2' ) ||
		    ( src[end] == '3' ) || ( src[end] == '4' ) || ( src[end] == '5' ) ||
		    ( src[end] == '6' ) || ( src[end] == '7' ) || ( src[end] == '8' ) ||
		    ( src[end] == '9' )                                                  )
	    {
		end++;

		if ( end >= (int) src.length() )
		{
		    break;
		}
	    }
	}

        // Must be outside of loop to correctly catch reals starting with a .

	end--;

	// Next we might get a decimal point, which must be followed by a numeric block.
	// Note that we can't have an operator loke .* here as this would have been caught
        // by the integer case.

	if ( end+1 < (int) src.length() )
	{
	    if ( src[end+1] == '.' )
	    {
		end++;
		end++;

		if ( end >= (int) src.length() )
		{
		    // Can't have a . at the end of a string

		    end = start-1;
		    return -1;
		}

		int tempend = end-1;

		while ( ( src[end] == '0' ) || ( src[end] == '1' ) || ( src[end] == '2' ) ||
		        ( src[end] == '3' ) || ( src[end] == '4' ) || ( src[end] == '5' ) ||
		        ( src[end] == '6' ) || ( src[end] == '7' ) || ( src[end] == '8' ) ||
		        ( src[end] == '9' )                                                  )
		{
		    end++;

		    if ( end >= (int) src.length() )
		    {
			break;
		    }
		}

		end--;

		if ( end <= tempend )
		{
		    if ( end <= start )
		    {
			// Must be an operator starting with .

			end = start-1;
			return 0;
		    }

		    else
		    {
			// Case can't happen as it would have been caught by the integer code above

			end = start-1;
			return -1;
		    }
		}
	    }
	}

	// After that we might get an e, possibly followed by -, then must be followed by a numeric block

	if ( end+1 < (int) src.length() )
	{
            if ( ( src[end+1] == 'e' ) || ( src[end+1] == 'E' ) )
	    {
		end++;
                end++;

		if ( end >= (int) src.length() )
		{
		    end = start-1;
		    return -1;
		}

		if ( src[end] == '-' )
		{
		    end++;
		}

		if ( end >= (int) src.length() )
		{
		    end = start-1;
		    return -1;
		}

		int tempend = end-1;

		while ( ( src[end] == '0' ) || ( src[end] == '1' ) || ( src[end] == '2' ) ||
		        ( src[end] == '3' ) || ( src[end] == '4' ) || ( src[end] == '5' ) ||
		        ( src[end] == '6' ) || ( src[end] == '7' ) || ( src[end] == '8' ) ||
		        ( src[end] == '9' )                                                  )
		{
		    end++;

		    if ( end >= (int) src.length() )
		    {
			break;
		    }
		}

		end--;

		if ( end <= tempend )
		{
		    end = start-1;
		    return -1;
		}
	    }
	}

        // Finally, if this is a quaternion, there may be an i I J K l m n o p q r at the end

	if ( end+1 < (int) src.length() )
	{
	    if ( ( src[end+1] == 'i' ) || ( src[end+1] == 'I' ) || ( src[end+1] == 'J' ) ||
		 ( src[end+1] == 'K' ) || ( src[end+1] == 'l' ) || ( src[end+1] == 'm' ) ||
		 ( src[end+1] == 'n' ) || ( src[end+1] == 'o' ) || ( src[end+1] == 'p' ) ||
		 ( src[end+1] == 'q' ) || ( src[end+1] == 'r' )                             )
	    {
		res = 3;
                end++;
	    }
	}
    }

    if ( !res && ( src[start] == '[' ) )
    {
	res = 4;

	if ( pairBrackets(start,end,src,0) )
	{
	    end = start-1;
            return -1;
	}
    }

    if ( !res && ( src[start] == 'M' ) )
    {
	end++;

	if ( end >= (int) src.length() )
	{
	    end = start-1;
	    return -1;
	}

	if ( src[end] == ':' )
	{
	    res = 5;

	    end++;

	    if ( end >= (int) src.length() )
	    {
		end = start-1;
		return -1;
	    }

	    while ( isspace(src[end]) )
	    {
		end++;

		if ( end >= (int) src.length() )
		{
		    end = start-1;
		    return -1;
		}
	    }

	    if ( src[end] != '[' )
	    {
		end = start-1;
		return -1;
	    }

	    // Note: first instance of start passed as value, second instance by reference

	    if ( pairBrackets(end,end,src,0) )
	    {
		end = start-1;
		return -1;
	    }
	}
    }

    if ( !res && ( src[start] == '{' ) )
    {
        res = 8;

	if ( pairBrackets(start,end,src,0) )
	{
	    end = start-1;
            return -1;
	}
    }

    if ( !res && ( src[start] == 'G' ) )
    {
	end++;

	if ( end >= (int) src.length() )
	{
	    end = start-1;
	    return -1;
	}

	if ( src[end] == ':' )
	{
            res = 9;

	    end++;

	    if ( end >= (int) src.length() )
	    {
		end = start-1;
		return -1;
	    }

	    while ( isspace(src[end]) )
	    {
		end++;

		if ( end >= (int) src.length() )
		{
		    end = start-1;
		    return -1;
		}
	    }

            if ( src[end] != '{' )
	    {
		end = start-1;
		return -1;
	    }

	    // Note: first instance of start passed as value, second instance by reference

	    if ( pairBrackets(end,end,src,0) )
	    {
		end = start-1;
		return -1;
	    }
	}
    }

    if ( !res && ( src[start] == '\"' ) )
    {
//errstream() << "phantomxyz 1\n";
	res = 6;

	if ( pairBrackets(start,end,src,0) )
	{
	    end = start-1;
            return -1;
	}
    }

    if ( !res && ( ( src[start] == 'E' ) && ( start < (int) src.length()-1 ) ) )
    {
	end++;

	if ( end >= (int) src.length() )
	{
	    end = start-1;
	    return -1;
	}

	if ( src[end] == ':' )
	{
	    res = 7;

	    end++;

	    if ( end >= (int) src.length() )
	    {
		end = start-1;
		return -1;
	    }

	    while ( isspace(src[end]) )
	    {
		end++;

		if ( end >= (int) src.length() )
		{
		    end = start-1;
		    return -1;
		}
	    }

	    if ( src[end] != '\"' )
	    {
		end = start-1;
		return -1;
	    }

	    // Note: first instance of start passed as value, second instance by reference

	    if ( pairBrackets(end,end,src,0) )
	    {
		end = start-1;
		return -1;
	    }
	}
    }

    return res;
}

// Given a string src and a starting point start, working from left
// to right, return numargs+1 if an expression number is located at
// this point and 0 otherwise (-1 for error).  exprname is a below.
// An expression is:
//
// {a}(b1,b2,...,bn)              a is an alphanumeric sequence starting with an alpha and possibly including _
//                                b. is something with paired brackets
//                                n is numargs
//
// Exceptions:
//
// - i,I,J,K,l,m,n,o,p,q,r, when not followed by a (, are numbers
//   and not syntax errors

int processExprLtoR(int start, int &end, int &isitastring, const std::string &src, std::string &exprname, Vector<int> &commapos)
{
    int res = 0;
    int isexpr = 0;

    NiceAssert( start >= 0 );
    NiceAssert( start < (int) src.length() );

    commapos.resize(zeroint());

    end = start;
    exprname = "";
    isitastring = 0;

    while (  ( src[end] == 'a' ) || ( src[end] == 'b' ) || ( src[end] == 'c' ) || ( src[end] == 'd' ) || ( src[end] == 'e' ) || ( src[end] == 'f' ) || ( src[end] == 'g' ) ||
             ( src[end] == 'h' ) || ( src[end] == 'i' ) || ( src[end] == 'j' ) || ( src[end] == 'k' ) || ( src[end] == 'l' ) || ( src[end] == 'm' ) || ( src[end] == 'n' ) ||
             ( src[end] == 'o' ) || ( src[end] == 'p' ) || ( src[end] == 'q' ) || ( src[end] == 'r' ) || ( src[end] == 's' ) || ( src[end] == 't' ) || ( src[end] == 'u' ) ||
             ( src[end] == 'v' ) || ( src[end] == 'w' ) || ( src[end] == 'x' ) || ( src[end] == 'y' ) || ( src[end] == 'z' ) || ( src[end] == '_' ) ||
             ( src[end] == 'A' ) || ( src[end] == 'B' ) || ( src[end] == 'C' ) || ( src[end] == 'D' ) || ( src[end] == 'E' ) || ( src[end] == 'F' ) || ( src[end] == 'G' ) ||
             ( src[end] == 'H' ) || ( src[end] == 'I' ) || ( src[end] == 'J' ) || ( src[end] == 'K' ) || ( src[end] == 'L' ) || ( src[end] == 'M' ) || ( src[end] == 'N' ) ||
             ( src[end] == 'O' ) || ( src[end] == 'P' ) || ( src[end] == 'Q' ) || ( src[end] == 'R' ) || ( src[end] == 'S' ) || ( src[end] == 'T' ) || ( src[end] == 'U' ) ||
	     ( src[end] == 'V' ) || ( src[end] == 'W' ) || ( src[end] == 'X' ) || ( src[end] == 'Y' ) || ( src[end] == 'Z' ) ||
	    ( ( end != start ) && ( ( src[end] == '0' ) || ( src[end] == '1' ) || ( src[end] == '2' ) || ( src[end] == '3' ) || ( src[end] == '4' ) || ( src[end] == '5' ) ||
				    ( src[end] == '6' ) || ( src[end] == '7' ) || ( src[end] == '8' ) || ( src[end] == '9' ) ) ) )
    {
	exprname += src[end];
	end++;

	if ( end >= (int) src.length() )
	{
            break;
	}
    }

    if ( ( end == start ) && ( src[end] != '(' ) )
    {
	// Definitely not an expression

	end = start-1;
        return 0;
    }

    if ( end < (int) src.length() )
    {
	// An expression is a string (possibly empty) followed by an expression
	// in brackets, so something(blah)

	if ( src[end] == '(' )
	{
	    isexpr = 1;

	    if ( end == ((int) src.length())-1 )
	    {
		// Unpaired brackets error

		exprname = "";
		end = start-1;
		return -1;
	    }
	}
    }

    // Allow for the possible strings that aren't expressions

    if ( !isexpr && ( ( exprname == "i" ) || ( exprname == "I" ) || ( exprname == "J" ) || ( exprname == "K" ) ||
		      ( exprname == "l" ) || ( exprname == "m" ) || ( exprname == "n" ) || ( exprname == "o" ) ||
		      ( exprname == "p" ) || ( exprname == "q" ) || ( exprname == "r" )                           ) )
    {
	exprname = "";
	end = start-1;
	return 0;
    }

    if ( isexpr )
    {
	int tempend = end;

	if ( pairBrackets(end,end,src,0) )
	{
	    // Unpaired brackets error

	    end = start-1;
            return -1;
	}

	if ( end == tempend+1 )
	{
	    // Brackets are (), so no arguments

            res = 1;
	}

	else
	{
	    // Brackets contain arguments - need to count the arguments and record where the commas are

	    res = 2;

            int i;
	    int curvecount = 0;
	    int squarecount = 0;
	    int curlycount = 0;
	    int quotecount = 0;
	    char charlast = ' ';

	    for ( i = tempend+1 ; i <= end-1 ; i++ )
	    {
		if ( src[i] == '(' ) { curvecount++; }
		if ( src[i] == ')' ) { curvecount--; }

		if ( src[i] == '[' ) { squarecount++; }
		if ( src[i] == ']' ) { squarecount--; }

		if ( src[i] == '{' ) { curlycount++; }
		if ( src[i] == '}' ) { curlycount--; }

		if ( ( src[i] == '\"' ) && ( charlast != '\\' ) ) { quotecount = quotecount ? 0 : 1; }

		if ( ( src[i] == ',' ) && !curvecount && !squarecount && !curlycount && !quotecount ) { res++; commapos.add(res-3); commapos("&",res-3) = i; }

		charlast = src[i];
	    }
	}
    }

    else
    {
	// This isn't an expression, but it is a string

	end--;
	isitastring = 1;
        res = 1;
    }

    return res;
}


std::ostream &operator<<(std::ostream &output, const eqninfoblock &src )
{
    output << "------------------------------------------\n";
    output << src.text   << "\n";
    output << src.type   << "\n";
    output << src.res    << "\n";
    output << src.fnname << "\n";
    output << src.commas << "\n";
    output << src.isstr  << "\n";
    output << "------------------------------------------\n";

    return output;
}

void qswap(eqninfoblock &a, eqninfoblock &b)
{
    qswap(a.text  ,b.text  );
    qswap(a.type  ,b.type  );
    qswap(a.res   ,b.res   );
    qswap(a.commas,b.commas);
    qswap(a.fnname,b.fnname);
    qswap(a.isstr ,b.isstr );

    return;
}

// Function to remove operators and replace them with equivalent functions
//
// Assumptions: unary operators right to left cleave to the right (eg a!+b is fine, but a+!b is not)
//              unary operators left to right cleave to the left (eg a||~b is fine, but a~||b is not)

int operatorToFunction(int LtoRRtoL, int UnaryBinary, const Vector<std::string> &opSymb, const Vector<std::string> &opFuncEquiv, Vector<eqninfoblock> &srcxblock)
{
    // Big assumption enforced by the caller: the blocks alternate .../opblock/other/opblock/other/...

    NiceAssert( opSymb.size() == opFuncEquiv.size() );

    int i,j;

    if ( opSymb.size() && ( srcxblock.size() > 1 ) )
    {
	for ( i = ( LtoRRtoL ? srcxblock.size()-1 : 0 ) ; ( i >= 0 ) && ( i < srcxblock.size() ) ; i += ( LtoRRtoL ? -1 : +1 ) )
	{
	    if ( !(srcxblock(i).type) )
	    {
		// This is a block of operators

		if ( i == srcxblock.size()-1 )
		{
		    // Rightmost operator block

		    if ( LtoRRtoL && !UnaryBinary )
		    {
			for ( j = 0 ; j < opSymb.size() ; j++ )
			{
			    if ( opSymb(j) == (srcxblock(i).text).substr(0,opSymb(j).length()) )
			    {
				// Remove operator from operator block

				(srcxblock("&",i).text).erase(0,opSymb(j).length());

				// Insert functional equivalent around previous operator

				(srcxblock("&",i-1).text) = opFuncEquiv(j) + "(" + (srcxblock(i-1).text) + ")";

				// if block empty then remove and break

				if ( !((srcxblock(i).text).length()) )
				{
				    srcxblock.remove(i);
                                    break;
				}
			    }
			}
		    }
		}

		else if ( i == 0 )
		{
		    if ( !LtoRRtoL && !UnaryBinary )
		    {
			for ( j = 0 ; j < opSymb.size() ; j++ )
			{
			    if ( opSymb(j) == (srcxblock(i).text).substr(((srcxblock(i).text).length())-(opSymb(j).length()),opSymb(j).length()) )
			    {
				// Remove operator from operator block

				(srcxblock("&",i).text).erase(((srcxblock(i).text).length())-(opSymb(j).length()),opSymb(j).length());

				// Insert functional equivalent around next operator

				(srcxblock("&",i+1).text) = opFuncEquiv(j) + "(" + (srcxblock(i+1).text) + ")";

				// if block empty then remove and break

				if ( !((srcxblock(i).text).length()) )
				{
				    srcxblock.remove(i);
                                    i--;
                                    break;
				}
			    }
			}
		    }
		}

		else
		{
		    if ( LtoRRtoL && !UnaryBinary )
		    {
			for ( j = 0 ; j < opSymb.size() ; j++ )
			{
			    if ( opSymb(j) == (srcxblock(i).text).substr(0,opSymb(j).length()) )
			    {
				// Remove operator from operator block

				(srcxblock("&",i).text).erase(0,opSymb(j).length());

				// Insert functional equivalent around previous operator

				(srcxblock("&",i-1).text) = opFuncEquiv(j) + "(" + (srcxblock(i-1).text) + ")";

				// if block empty then remove and break

				if ( !((srcxblock(i).text).length()) )
				{
                                    return -1;
				}
			    }
			}
		    }

		    else if ( !LtoRRtoL && !UnaryBinary )
		    {
			for ( j = 0 ; j < opSymb.size() ; j++ )
			{
			    if ( opSymb(j) == (srcxblock(i).text).substr(((srcxblock(i).text).length())-(opSymb(j).length()),opSymb(j).length()) )
			    {
				// Remove operator from operator block

				(srcxblock("&",i).text).erase(((srcxblock(i).text).length())-(opSymb(j).length()),opSymb(j).length());

				// Insert functional equivalent around next operator

				(srcxblock("&",i+1).text) = opFuncEquiv(j) + "(" + (srcxblock(i+1).text) + ")";

				// if block empty then remove and break

				if ( !((srcxblock(i).text).length()) )
				{
                                    return -1;
				}
			    }
			}
		    }

		    else if ( LtoRRtoL && UnaryBinary )
		    {
			for ( j = 0 ; j < opSymb.size() ; j++ )
			{
			    if ( opSymb(j) == srcxblock(i).text )
			    {
				// Insert functional equivalent around previous operator

				(srcxblock("&",i-1).text) = opFuncEquiv(j) + "(" + (srcxblock(i-1).text) + "," + (srcxblock(i+1).text) + ")";

				// Remove operator from operator block and right expression/number

				srcxblock.remove(i+1);
				srcxblock.remove(i);
				i--;
                                break;
			    }
			}
		    }

		    else if ( !LtoRRtoL && UnaryBinary )
		    {
			for ( j = 0 ; j < opSymb.size() ; j++ )
			{
			    if ( opSymb(j) == srcxblock(i).text )
			    {
				// Insert functional equivalent around previous operator

				(srcxblock("&",i+1).text) = opFuncEquiv(j) + "(" + (srcxblock(i-1).text) + "," + (srcxblock(i+1).text) + ")";

				// Remove operator from operator block and right expression/number

				srcxblock.remove(i);
				srcxblock.remove(i-1);
				i--;
				i--;
                                break;
			    }
			}
		    }
		}
	    }
	}
    }

    return 0;
}



// NOTE: the following function should be:
//
// svmvolatile fninfoblock *getfninfo(void);
// svmvolatile fninfoblock *getfninfo(void)
// {
//     svmvolatile static fninfoblock fninfo[NUMFNDEF] = {
//
//     ...
//
// However when compiled using djgpp with -O1, -O2 or -O3 that throws up
// the following compiler error.
//
// gentype.cc: In function 'int getfnind(const std::string&)':
// gentype.cc:6486: internal compiler error: in gimple_rhs_has_side_effects, at gimple.c:2343
// Please submit a full bug report,
// with preprocessed source if appropriate.
// See <http://gcc.gnu.org/bugs.html> for instructions.
//
// Now, for multi-threaded operation it is mandatory to call the init function
// before you start using gentype, so this should not cause an issue (I think),
// but ultimately it would be nice to find a work-around

// NB: this must be a global (non-static, which means something different in this
// context).  This is to ensure that the destructor ~fninfoblock gets called 
// after atexit cleanup has been done to delete realderiv.  If you make this a 
// static member of the function getfninfo (which it used to be) and compile
// using make opt then the destructor is called *before* the atexit functions,
// which results in attempts to access already deleted qqqfninfo elements and
// causes crashes.

////svmvolatile fninfoblock *getfninfo(void);
////svmvolatile fninfoblock *getfninfo(void)
//fninfoblock *getfninfo(void);
//fninfoblock *getfninfo(void)
//{
////    svmvolatile static fninfoblock fninfo[NUMFNDEF] = {
////    static fninfoblock fninfo[NUMFNDEF] = {
    static fninfoblock qqqfninfo[NUMFNDEF] = {
        { ""                ,1,0 ,0,1 ,1,0,NULL ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,1 ,NULL ,"var(1,0)" },
        { "var"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"Var"           ,-1,0 ,0 ,NULL ,"0" },
        { "Var"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"var"           ,-1,0 ,0 ,NULL ,"0" },
        { "gvar"            ,2,3 ,0,3 ,0,0,NULL ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"gVar"          ,-1,0 ,0 ,NULL ,"0" },
        { "gVar"            ,2,3 ,0,3 ,0,0,NULL ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"gvar"          ,-1,0 ,0 ,NULL ,"0" },
        { "pos"             ,1,1 ,0,1 ,1,0,NULL ,pos             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_pos      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,1 ,NULL ,"var(1,0)" },
        { "neg"             ,1,1 ,0,1 ,1,0,NULL ,neg             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_neg      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,1 ,NULL ,"-var(1,0)" },
        { "add"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,add          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,OP_add ,NULL,NULL,NULL,NULL,3 ,"~"             ,-1,0 ,3 ,NULL ,"var(1,0)+var(1,1)" },
        { "sub"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,sub          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,OP_sub ,NULL,NULL,NULL,NULL,3 ,"~"             ,-1,0 ,3 ,NULL ,"var(1,0)-var(1,1)" },
        { "mul"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,mul          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,OP_mul ,NULL,NULL,NULL,NULL,-3,"~"             ,-1,3 ,3 ,NULL ,"(var(1,0)*y)+(x*var(1,1))" },
        { "div"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,div          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,OP_div ,NULL,NULL,NULL,NULL,-3,"rdiv"          ,-1,3 ,3 ,NULL ,"(var(1,0)/y)-((x/(y^2))*var(1,1))" },
        { "idiv"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,idiv         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,OP_idiv,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "rdiv"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,rdiv         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,OP_rdiv,NULL,NULL,NULL,NULL,-3,"div"           ,-1,3 ,3 ,NULL ,"-(var(1,0)*((x^2)\\y))+(x\\var(1,1))" },
        { "mod"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,mod          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,OP_mod ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "pow"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,pow          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Pow"           ,-1,3 ,3 ,NULL ,"(y*pow(x,y-1)*var(1,0))+(log(x)*pow(x,y)*var(1,1))" },
        { "Pow"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Pow          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"pow"           ,-1,3 ,3 ,NULL ,"(y*Pow(x,y-1)*var(1,0))+(Log(x)*Pow(x,y)*var(1,1))" },
        { "powl"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,powl         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Powr"          ,-1,3 ,3 ,NULL ,"(y*powl(x,y-1)*var(1,0))+(log(x)*powl(x,y)*var(1,1))" },
        { "Powl"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Powl         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"powr"          ,-1,3 ,3 ,NULL ,"(y*Powl(x,y-1)*var(1,0))+(Log(x)*Powl(x,y)*var(1,1))" },
        { "powr"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,powr         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Powl"          ,-1,3 ,3 ,NULL ,"(y*powr(x,y-1)*var(1,0))+(log(x)*powr(x,y)*var(1,1))" },
        { "Powr"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Powr         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"powl"          ,-1,3 ,3 ,NULL ,"(y*Powr(x,y-1)*var(1,0))+(Log(x)*Powr(x,y)*var(1,1))" },
        { "emul"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,emul         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,-3,"~"             ,-1,3 ,3 ,NULL ,"(var(1,0).*y)+(x.*var(1,1))" },
        { "ediv"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,ediv         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,-3,"erdiv"         ,-1,3 ,3 ,NULL ,"(var(1,0)./y)-((x./(y.^2)).*var(1,1))" },
        { "eidiv"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,eidiv        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "erdiv"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,erdiv        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,-3,"ediv"          ,-1,3 ,3 ,NULL ,"-(var(1,0).*((x.^2).\\y))+(x.\\var(1,1))" },
        { "emod"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,emod         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "epow"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,epow         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Epow"          ,-1,3 ,3 ,NULL ,"((y.*epow(x,y-1)).*var(1,0))+((log(x).*epow(x,y)).*var(1,1))" },
        { "Epow"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Epow         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"epow"          ,-1,3 ,3 ,NULL ,"((y.*Epow(x,y-1)).*var(1,0))+((Log(x).*Epow(x,y)).*var(1,1))" },
        { "epowl"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,epowl        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Epowr"         ,-1,3 ,3 ,NULL ,"((y.*epowl(x,y-1)).*var(1,0))+((log(x).*epowl(x,y)).*var(1,1))" },
        { "Epowl"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Epowl        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"epowr"         ,-1,3 ,3 ,NULL ,"((y.*Epowl(x,y-1)).*var(1,0))+((Log(x).*Epowl(x,y)).*var(1,1))" },
        { "epowr"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,epowr        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Epowl"         ,-1,3 ,3 ,NULL ,"((y.*epowr(x,y-1)).*var(1,0))+((log(x).*epowr(x,y)).*var(1,1))" },
        { "Epowr"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Epowr        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"epowl"         ,-1,3 ,3 ,NULL ,"((y.*Epowr(x,y-1)).*var(1,0))+((Log(x).*Epowr(x,y)).*var(1,1))" },
        { "eq"              ,2,3 ,0,3 ,1,0,NULL ,NULL            ,eq           ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ne"              ,2,3 ,0,3 ,1,0,NULL ,NULL            ,ne           ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "gt"              ,2,3 ,0,3 ,1,0,NULL ,NULL            ,gt           ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ge"              ,2,3 ,0,3 ,1,0,NULL ,NULL            ,ge           ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "le"              ,2,3 ,0,3 ,1,0,NULL ,NULL            ,le           ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "lt"              ,2,3 ,0,3 ,1,0,NULL ,NULL            ,lt           ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "eeq"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,eeq          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ene"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,ene          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "egt"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,egt          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ege"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,ege          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ele"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,ele          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "elt"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,elt          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "lnot"            ,1,1 ,0,1 ,1,0,NULL ,lnot            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_lnot     ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "lor"             ,2,3 ,0,3 ,1,0,NULL ,NULL            ,lor          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "land"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,land         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ifthenelse"      ,3,1 ,0,7 ,1,0,NULL ,NULL            ,NULL         ,ifthenelse   ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,6 ,"~"             ,-1,6 ,1 ,NULL ,"ifthenelse(var(1,0),y,z)" },
        { "isnull"          ,1,1 ,0,1 ,1,0,NULL ,isnull          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isnull   ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isint"           ,1,1 ,0,1 ,1,0,NULL ,isint           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isint    ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isreal"          ,1,1 ,0,1 ,1,0,NULL ,isreal          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isreal   ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isanion"         ,1,1 ,0,1 ,1,0,NULL ,isanion         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isanion  ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isvector"        ,1,1 ,0,1 ,1,0,NULL ,isvector        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isvector ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ismatrix"        ,1,1 ,0,1 ,1,0,NULL ,ismatrix        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_ismatrix ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isset"           ,1,1 ,0,1 ,1,0,NULL ,isset           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isset    ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isdgraph"        ,1,1 ,0,1 ,1,0,NULL ,isdgraph        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isdgraph ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isstring"        ,1,1 ,0,1 ,1,0,NULL ,isstring        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isstring ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "iserror"         ,1,1 ,0,1 ,1,0,NULL ,iserror         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_iserror  ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "size"            ,1,1 ,0,1 ,1,0,NULL ,size            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_size     ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "numRows"         ,1,1 ,0,1 ,1,0,NULL ,numRows         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_numRows  ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "numCols"         ,1,1 ,0,1 ,1,0,NULL ,numCols         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_numCols  ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "null"            ,0,0 ,0,0 ,1,0,null ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "pi"              ,0,0 ,0,0 ,1,0,pi   ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "euler"           ,0,0 ,0,0 ,1,0,euler,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "pinf"            ,0,0 ,0,0 ,1,0,pinf ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ninf"            ,0,0 ,0,0 ,1,0,ninf ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "vnan"            ,0,0 ,0,0 ,1,0,vnan ,NULL            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "eye"             ,2,0 ,3,3 ,1,0,NULL ,NULL            ,eye          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" }, 
        { "conj"            ,1,0 ,0,1 ,1,0,NULL ,conj            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,""              ,-1,0 ,1 ,NULL ,"conj(var(1,0))" },
        { "realDeriv"       ,3,3 ,0,7 ,0,0,NULL ,NULL            ,NULL         ,realDeriv    ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,4 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "eps_comm"        ,4,15,0,15,1,0,NULL ,NULL            ,NULL         ,NULL         ,eps_comm,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "eps_assoc"       ,5,31,0,31,1,0,NULL ,NULL            ,NULL         ,NULL         ,NULL    ,eps_assoc ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "im_complex"      ,1,1 ,0,1 ,1,0,NULL ,im_complex      ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"Im_complex"    ,-1,0 ,0 ,NULL ,"0" },
        { "im_quat"         ,1,1 ,0,1 ,1,0,NULL ,im_quat         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"Im_quat"       ,-1,0 ,0 ,NULL ,"0" },
        { "im_octo"         ,1,1 ,0,1 ,1,0,NULL ,im_octo         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"Im_octo"       ,-1,0 ,0 ,NULL ,"0" },
        { "im_anion"        ,2,3 ,0,3 ,1,0,NULL ,NULL            ,im_anion     ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"Im_anion"      ,-1,0 ,0 ,NULL ,"0" },
        { "Im_complex"      ,1,1 ,0,1 ,1,0,NULL ,Im_complex      ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"im_complex"    ,-1,0 ,0 ,NULL ,"0" },
        { "Im_quat"         ,1,1 ,0,1 ,1,0,NULL ,Im_quat         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"im_quat"       ,-1,0 ,0 ,NULL ,"0" },
        { "Im_octo"         ,1,1 ,0,1 ,1,0,NULL ,Im_octo         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"im_octo"       ,-1,0 ,0 ,NULL ,"0" },
        { "Im_anion"        ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Im_anion     ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"im_anion"      ,-1,0 ,0 ,NULL ,"0" },
        { "vect_const"      ,2,3 ,0,3 ,1,0,NULL ,NULL            ,vect_const   ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,2 ,"~"             ,-1,1 ,2 ,NULL ,"vect_const(x,var(1,1))" },
        { "vect_unit"       ,2,3 ,0,3 ,1,0,NULL ,NULL            ,vect_unit    ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ivect"           ,3,7 ,0,3 ,1,0,NULL ,NULL            ,NULL         ,ivect        ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,7 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "commutate"       ,2,3 ,0,3 ,1,0,NULL ,NULL            ,commutate    ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,-3,"~"             ,-1,3 ,3 ,NULL ,"commutate(var(1,0),y)+commutate(x,var(1,1))" },
        { "associate"       ,3,7 ,0,7 ,1,0,NULL ,NULL            ,NULL         ,associate    ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,-7,"~"             ,-1,7 ,7 ,NULL ,"associate(var(1,0),y,z)+associate(x,var(1,1),z)+associate(x,y,var(1,2))" },
        { "anticommutate"   ,2,3 ,0,3 ,1,0,NULL ,NULL            ,anticommutate,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"~"             ,-1,3 ,3 ,NULL ,"anticommutate(var(1,0),y)+anticommutate(x,var(1,1))" },
        { "antiassociate"   ,3,7 ,0,7 ,1,0,NULL ,NULL            ,NULL         ,antiassociate,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,7 ,"~"             ,-1,7 ,7 ,NULL ,"antiassociate(var(1,0),y,z)+antiassociate(x,var(1,1),z)+antiassociate(x,y,var(1,2))" },
        { "cayleyDickson"   ,2,3 ,0,3 ,1,0,NULL ,NULL            ,cayleyDickson,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"CayleyDickson" ,-1,0 ,3 ,NULL ,"cayleyDickson(var(1,0),var(1,1))" },
        { "CayleyDickson"   ,2,3 ,0,3 ,1,0,NULL ,NULL            ,CayleyDickson,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"cayleyDickson" ,-1,0 ,3 ,NULL ,"CayleyDickson(var(1,0),var(1,1))" },
        { "kronDelta"       ,2,0 ,3,3 ,1,0,NULL ,NULL            ,kronDelta    ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "diracDelta"      ,1,0 ,1,1 ,1,0,NULL ,diracDelta      ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ekronDelta"      ,2,3 ,0,3 ,1,0,NULL ,NULL            ,ekronDelta   ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ediracDelta"     ,1,1 ,0,1 ,1,0,NULL ,ediracDelta     ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "perm"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,perm         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "comb"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,comb         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "fact"            ,1,1 ,0,1 ,1,0,NULL ,fact            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,gamma(x+1))*var(1,0)" },
        { "abs2"            ,1,1 ,0,1 ,1,0,NULL ,abs2            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"angle(x)*var(1,0)" },
        { "abs1"            ,1,1 ,0,1 ,1,0,NULL ,abs1            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"sgn(x)*var(1,0)" },
        { "absp"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,absp         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,3 ,3 ,NULL ,"realDeriv(0,0,sum(eabs2(x).^y)^inv(y))*var(1,0)+realDeriv(0,1,sum(eabs2(x).^y)^inv(y))*var(1,1)" },
        { "absinf"          ,1,1 ,0,1 ,1,0,NULL ,absinf          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argmaxabs(x))" },
        { "norm1"           ,1,1 ,0,1 ,1,0,NULL ,norm1           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"sgn(x)*var(1,0)" },
        { "norm2"           ,1,1 ,0,1 ,1,0,NULL ,norm2           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"2*x*var(1,0)" },
        { "normp"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,normp        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,3 ,3 ,NULL ,"realDeriv(0,0,sum(eabs2(x).^y))*var(1,0)+realDeriv(0,1,sum(eabs2(x).^y))*var(1,1)" },
        { "angle"           ,1,1 ,0,1 ,1,0,NULL ,angle           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"inv(abs2(x))*(var(1,0)-angle(x)*(angle(x)*var(1,0)))" },
        { "inv"             ,1,1 ,0,1 ,1,0,NULL ,inv             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-(x^-2)*var(1,0)" },
        { "eabs2"           ,1,1 ,0,1 ,1,0,NULL ,eabs2           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_eabs2    ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"sgn(x).*var(1,0)" },
        { "eabs1"           ,1,1 ,0,1 ,1,0,NULL ,eabs1           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_eabs1    ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"sgn(x).*var(1,0)" },
        { "eabsp"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,eabsp        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,3 ,3 ,NULL ,"realDeriv(0,0,enormp(x,y).^inv(y)).*var(1,0)+realDeriv(0,1,enormp(x,y).^inv(y)).*var(1,1)" },
        { "eabsinf"         ,1,1 ,0,1 ,1,0,NULL ,eabsinf         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_eabsinf  ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"sgn(x).*var(1,0)" },
        { "enorm1"          ,1,1 ,0,1 ,1,0,NULL ,enorm1          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_enorm1   ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"sgn(x).*var(1,0)" },
        { "enorm2"          ,1,1 ,0,1 ,1,0,NULL ,enorm2          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_enorm2   ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"x.*var(1,0)" },
        { "enormp"          ,2,3 ,0,3 ,1,0,NULL ,NULL            ,enormp       ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,3 ,3 ,NULL ,"((y.*(eabs2(x).^(y-1)).*sgn(x)).*var(1,0))+((log(x).*(eabs2(x).^y)).*var(1,0))" },
        { "eangle"          ,1,1 ,0,1 ,1,0,NULL ,eangle          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_eangle   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"einv(eabs2(x)).*(var(1,0)-eangle(x).*eangle(x).*var(1,0))" },
        { "einv"            ,1,1 ,0,1 ,1,0,NULL ,einv            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_einv     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-(x.^-2).*var(1,0)" },
        { "real"            ,1,1 ,0,1 ,1,0,NULL ,real            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_real     ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,1 ,NULL ,"real(var(1,0))" },
        { "imag"            ,1,1 ,0,1 ,1,0,NULL ,imag            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_imag     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,eabs2(imagx(x)).*sgn(derefa(x,1))).*var(1,0)" },
        { "imagd"           ,1,1 ,0,1 ,1,0,NULL ,imagd           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_imagd    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Imagd"         ,-1,1 ,1 ,NULL ,"realDeriv(0,0,imagx(x)./imag(x)).*var(1,0)" },
        { "Imagd"           ,1,1 ,0,1 ,1,0,NULL ,Imagd           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Imagd    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"imagd"         ,-1,1 ,1 ,NULL ,"realDeriv(0,0,imagx(x)./imag(x)).*var(1,0)" },
        { "imagx"           ,1,1 ,0,1 ,1,0,NULL ,imagx           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_imagx    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"imagx"         ,-1,0 ,1 ,NULL ,"imagx(var(1,0))" },
        { "arg"             ,1,1 ,0,1 ,1,0,NULL ,arg             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_arg      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,eabs2(imagx(log(x))).*sgn(derefa(x,1))).*var(1,0)" },
        { "argd"            ,1,1 ,0,1 ,1,0,NULL ,argd            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_argd     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Argd"          ,-1,1 ,1 ,NULL ,"realDeriv(0,0,imagx(log(x))./arg(x)).*var(1,0)" },
        { "Argd"            ,1,1 ,0,1 ,1,0,NULL ,Argd            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Argd     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"argd"          ,-1,1 ,1 ,NULL ,"realDeriv(0,0,imagx(Log(x))./arg(x)).*var(1,0)" },
        { "argx"            ,1,1 ,0,1 ,1,0,NULL ,argx            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_argx     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Argx"          ,-1,1 ,1 ,NULL ,"realDeriv(0,0,imagx(log(x))).*var(1,0)" },
        { "Argx"            ,1,1 ,0,1 ,1,0,NULL ,Argx            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Argx     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"argx"          ,-1,1 ,1 ,NULL ,"realDeriv(0,0,imagx(Log(x))).*var(1,0)" },
        { "polar"           ,3,7 ,0,7 ,1,0,NULL ,NULL            ,NULL         ,polar        ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,7 ,"~"             ,-1,7 ,7 ,NULL ,"(realDeriv(0,0,(x.*exp(y.*z)+exp(z.*y).*x)/2).*var(1,0))+(realDeriv(0,1,(x.*exp(y.*z)+exp(z.*y).*x)/2).*var(1,1))+(realDeriv(0,2,(x.*exp(y.*z)+exp(z.*y).*x)/2).*var(1,2))" },
        { "polard"          ,3,7 ,0,7 ,1,0,NULL ,NULL            ,NULL         ,polard       ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,7 ,"~"             ,-1,7 ,7 ,NULL ,"(realDeriv(0,0,(x.*exp(y.*z)+exp(z.*y).*x)/2).*var(1,0))+(realDeriv(0,1,(x.*exp(y.*z)+exp(z.*y).*x)/2).*var(1,1))+(realDeriv(0,2,(x.*exp(y.*z)+exp(z.*y).*x)/2).*var(1,2))" },
        { "polarx"          ,2,3 ,0,3 ,1,0,NULL ,NULL            ,polarx       ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"~"             ,-1,3 ,3 ,NULL ,"realDeriv(0,0,(x.*exp(y)+exp(y).*x)/2).*var(1,0)+realDeriv(0,1,(x.*exp(y)+exp(y).*x)/2).*var(1,1)" },
        { "sgn"             ,1,1 ,0,1 ,1,0,NULL ,sgn             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sgn      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"ediracDelta(x).*var(1,0)" },
        { "sqrt"            ,1,1 ,0,1 ,1,0,NULL ,sqrt            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sqrt     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Sqrt"          ,-1,1 ,1 ,NULL ,"einv(2*sqrt(x)).*var(1,0)" },
        { "Sqrt"            ,1,1 ,0,1 ,1,0,NULL ,Sqrt            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Sqrt     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"sqrt"          ,-1,1 ,1 ,NULL ,"einv(2*Sqrt(x)).*var(1,0)" },
        { "exp"             ,1,1 ,0,1 ,1,0,NULL ,exp             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_exp      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"exp"           ,-1,1 ,1 ,NULL ,"exp(x).*var(1,0)" },
        { "tenup"           ,1,1 ,0,1 ,1,0,NULL ,tenup           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_tenup    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,exp(log(10)*x)).*var(1,0)" },
        { "log"             ,1,1 ,0,1 ,1,0,NULL ,log             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_log      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Log"           ,-1,1 ,1 ,NULL ,"einv(x).*var(1,0)" },
        { "Log"             ,1,1 ,0,1 ,1,0,NULL ,Log             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Log      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"log"           ,-1,1 ,1 ,NULL ,"einv(x).*var(1,0)" },
        { "log10"           ,1,1 ,0,1 ,1,0,NULL ,log10           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_log10    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Log10"         ,-1,1 ,1 ,NULL ,"einv(log(10)*x).*var(1,0)" },
        { "Log10"           ,1,1 ,0,1 ,1,0,NULL ,Log10           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Log10    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"log10"         ,-1,1 ,1 ,NULL ,"einv(log(10)*x).*var(1,0)" },
        { "logb"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,logb         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Logb"          ,-1,3 ,3 ,NULL ,"realDeriv(0,0,(logbl(x,y)+logbr(x,y))/2).*var(1,0)+realDeriv(0,1,(logbl(x,y)+logbr(x,y))/2).*var(1,1)" },
        { "Logb"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Logb         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"logb"          ,-1,3 ,3 ,NULL ,"realDeriv(0,0,(Logbl(x,y)+Logbr(x,y))/2).*var(1,0)+realDeriv(0,1,(Logbl(x,y)+Logbr(x,y))/2).*var(1,1)" },
        { "logbl"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,logbl        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Logbr"         ,-1,3 ,3 ,NULL ,"realDeriv(0,0,log(x)./log(y)).*var(1,0)+realDeriv(0,1,log(x)./log(y)).*var(1,1)" },
        { "Logbl"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Logbl        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"logbr"         ,-1,3 ,3 ,NULL ,"realDeriv(0,0,Log(x)./Log(y)).*var(1,0)+realDeriv(0,1,Log(x)./Log(y)).*var(1,1)" },
        { "logbr"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,logbr        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"Logbl"         ,-1,3 ,3 ,NULL ,"realDeriv(0,0,log(y).\\log(x)).*var(1,0)+realDeriv(0,1,log(y).\\log(x)).*var(1,1)" },
        { "Logbr"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,Logbr        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"logbl"         ,-1,3 ,3 ,NULL ,"realDeriv(0,0,Log(y).\\Log(x)).*var(1,0)+realDeriv(0,1,Log(y).\\Log(x)).*var(1,1)" },
        { "sin"             ,1,1 ,0,1 ,1,0,NULL ,sin             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sin      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"cos(x).*var(1,0)" },
        { "cos"             ,1,1 ,0,1 ,1,0,NULL ,cos             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_cos      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-sin(x).*var(1,0)" },
        { "tan"             ,1,1 ,0,1 ,1,0,NULL ,tan             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_tan      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"(sec(x).^2).*var(1,0)" },
        { "cosec"           ,1,1 ,0,1 ,1,0,NULL ,cosec           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_cosec    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-cosec(x).*cot(x).*var(1,0)" },
        { "sec"             ,1,1 ,0,1 ,1,0,NULL ,sec             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sec      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"sec(x).*tan(x).*var(1,0)" },
        { "cot"             ,1,1 ,0,1 ,1,0,NULL ,cot             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_cot      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-(cosec(x).^2).*var(1,0)" },
        { "asin"            ,1,1 ,0,1 ,1,0,NULL ,asin            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_asin     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Asin"          ,-1,1 ,1 ,NULL ,"einv(sqrt(1-x.^2)).*var(1,0)" },
        { "Asin"            ,1,1 ,0,1 ,1,0,NULL ,Asin            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Asin     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"asin"          ,-1,1 ,1 ,NULL ,"einv(Sqrt(1-x.^2)).*var(1,0)" },
        { "acos"            ,1,1 ,0,1 ,1,0,NULL ,acos            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_acos     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Acos"          ,-1,1 ,1 ,NULL ,"-einv(sqrt(1-x.^2)).*var(1,0)" },
        { "Acos"            ,1,1 ,0,1 ,1,0,NULL ,Acos            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Acos     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"acos"          ,-1,1 ,1 ,NULL ,"-einv(Sqrt(1-x.^2)).*var(1,0)" },
        { "atan"            ,1,1 ,0,1 ,1,0,NULL ,atan            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_atan     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"einv(1+x.^2).*var(1,0)" },
        { "acosec"          ,1,1 ,0,1 ,1,0,NULL ,acosec          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_acosec   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Acosec"        ,-1,1 ,1 ,NULL ,"-einv(x.*sqrt(x.^2-1)).*var(1,0)" },
        { "Acosec"          ,1,1 ,0,1 ,1,0,NULL ,Acosec          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Acosec   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"acosec"        ,-1,1 ,1 ,NULL ,"-einv(x.*Sqrt(x.^2-1)).*var(1,0)" },
        { "asec"            ,1,1 ,0,1 ,1,0,NULL ,asec            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_asec     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Asec"          ,-1,1 ,1 ,NULL ,"einv(x.*sqrt(x.^2-1)).*var(1,0)" },
        { "Asec"            ,1,1 ,0,1 ,1,0,NULL ,Asec            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Asec     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"asec"          ,-1,1 ,1 ,NULL ,"einv(x.*Sqrt(x.^2-1)).*var(1,0)" },
        { "acot"            ,1,1 ,0,1 ,1,0,NULL ,acot            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_acot     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-einv(1+x.^2).*var(1,0)" },
        { "sinc"            ,1,1 ,0,1 ,1,0,NULL ,sinc            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sinc     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"((cos(x)-sinc(x))./x).*var(1,0)" },
        { "cosc"            ,1,1 ,0,1 ,1,0,NULL ,cosc            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_cosc     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"((-sin(x)-cosc(x))./x).*var(1,0)" },
        { "tanc"            ,1,1 ,0,1 ,1,0,NULL ,tanc            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_tanc     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"((sec(x).^2-tanc(x))./x).*var(1,0)" },
        { "vers"            ,1,1 ,0,1 ,1,0,NULL ,vers            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_vers     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,1-cos(x)).*var(1,0)" },
        { "covers"          ,1,1 ,0,1 ,1,0,NULL ,covers          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_covers   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,1-sin(x)).*var(1,0)" },
        { "hav"             ,1,1 ,0,1 ,1,0,NULL ,hav             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_hav      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,vers(x)/2).*var(1,0)" },
        { "excosec"         ,1,1 ,0,1 ,1,0,NULL ,excosec         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_excosec  ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,cosec(x)-1).*var(1,0)" },
        { "exsec"           ,1,1 ,0,1 ,1,0,NULL ,exsec           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_exsec    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,sec(x)-1).*var(1,0)" },
        { "avers"           ,1,1 ,0,1 ,1,0,NULL ,avers           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_avers    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Avers"         ,-1,1 ,1 ,NULL ,"realDeriv(0,0,acos(x+1)).*var(1,0)" },
        { "Avers"           ,1,1 ,0,1 ,1,0,NULL ,Avers           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Avers    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"avers"         ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Acos(x+1)).*var(1,0)" },
        { "acovers"         ,1,1 ,0,1 ,1,0,NULL ,acovers         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_acovers  ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Acovers"       ,-1,1 ,1 ,NULL ,"realDeriv(0,0,asin(x+1)).*var(1,0)" },
        { "Acovers"         ,1,1 ,0,1 ,1,0,NULL ,Acovers         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Acovers  ,NULL   ,NULL,NULL,NULL,NULL,1 ,"acovers"       ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Asin(x+1)).*var(1,0)" },
        { "ahav"            ,1,1 ,0,1 ,1,0,NULL ,ahav            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_ahav     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Ahav"          ,-1,1 ,1 ,NULL ,"realDeriv(0,0,avers(2*x)).*var(1,0)" },
        { "Ahav"            ,1,1 ,0,1 ,1,0,NULL ,Ahav            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Ahav     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"ahav"          ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Avers(2*x)).*var(1,0)" },
        { "aexcosec"        ,1,1 ,0,1 ,1,0,NULL ,aexcosec        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_aexcosec ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Aexcosec"      ,-1,1 ,1 ,NULL ,"realDeriv(0,0,acosec(x+1)).*var(1,0)" },
        { "Aexcosec"        ,1,1 ,0,1 ,1,0,NULL ,Aexcosec        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Aexcosec ,NULL   ,NULL,NULL,NULL,NULL,1 ,"aexcosec"      ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Acosec(x+1)).*var(1,0)" },
        { "aexsec"          ,1,1 ,0,1 ,1,0,NULL ,aexsec          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_aexsec   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Aexsec"        ,-1,1 ,1 ,NULL ,"realDeriv(0,0,asec(x+1)).*var(1,0)" },
        { "Aexsec"          ,1,1 ,0,1 ,1,0,NULL ,Aexsec          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Aexsec   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"aexsec"        ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Asec(x+1)).*var(1,0)" },
        { "sinh"            ,1,1 ,0,1 ,1,0,NULL ,sinh            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sinh     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"cosh(x).*var(1,0)" },
        { "cosh"            ,1,1 ,0,1 ,1,0,NULL ,cosh            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_cosh     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"sinh(x).*var(1,0)" },
        { "tanh"            ,1,1 ,0,1 ,1,0,NULL ,tanh            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_tanh     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"(sech(x).^2).*var(1,0)" },
        { "cosech"          ,1,1 ,0,1 ,1,0,NULL ,cosech          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_cosech   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-cosech(x).*coth(x).*var(1,0)" },
        { "sech"            ,1,1 ,0,1 ,1,0,NULL ,sech            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sech     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-sech(x).*tanh(x).*var(1,0)" },
        { "coth"            ,1,1 ,0,1 ,1,0,NULL ,coth            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_coth     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-(cosech(x).^2).*var(1,0)" },
        { "asinh"           ,1,1 ,0,1 ,1,0,NULL ,asinh           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_asinh    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"einv(sqrt(x.^2+1)).*var(1,0)" },
        { "acosh"           ,1,1 ,0,1 ,1,0,NULL ,acosh           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_acosh    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Acosh"         ,-1,1 ,1 ,NULL ,"einv(sqrt(x.^2-1)).*var(1,0)" },
        { "Acosh"           ,1,1 ,0,1 ,1,0,NULL ,Acosh           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Acosh    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"acosh"         ,-1,1 ,1 ,NULL ,"einv(Sqrt(x.^2-1)).*var(1,0)" },
        { "atanh"           ,1,1 ,0,1 ,1,0,NULL ,atanh           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_atanh    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Atanh"         ,-1,1 ,1 ,NULL ,"einv(1-x.^2).*var(1,0)" },
        { "Atanh"           ,1,1 ,0,1 ,1,0,NULL ,Atanh           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Atanh    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"atanh"         ,-1,1 ,1 ,NULL ,"einv(1-x.^2).*var(1,0)" },
        { "acosech"         ,1,1 ,0,1 ,1,0,NULL ,acosech         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_acosech  ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"-sgn(real(x)).*einv(x.*sqrt(1+x.^2)).*var(1,0)" },
        { "asech"           ,1,1 ,0,1 ,1,0,NULL ,asech           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_asech    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Asech"         ,-1,1 ,1 ,NULL ,"-sgn(real(x)).*einv(x.*sqrt(1-x.^2)).*var(1,0)" },
        { "Asech"           ,1,1 ,0,1 ,1,0,NULL ,Asech           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Asech    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"asech"         ,-1,1 ,1 ,NULL ,"-sgn(real(x)).*einv(x.*Sqrt(1-x.^2)).*var(1,0)" },
        { "acoth"           ,1,1 ,0,1 ,1,0,NULL ,acoth           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_acoth    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Acoth"         ,-1,1 ,1 ,NULL ,"einv(1-x.^2).*var(1,0)" },
        { "Acoth"           ,1,1 ,0,1 ,1,0,NULL ,Acoth           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Acoth    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"acoth"         ,-1,1 ,1 ,NULL ,"einv(1-x.^2).*var(1,0)" },
        { "sinhc"           ,1,1 ,0,1 ,1,0,NULL ,sinhc           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sinhc    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"((cosh(x)-sinhc(x))./x).*var(1,0)" },
        { "coshc"           ,1,1 ,0,1 ,1,0,NULL ,coshc           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_coshc    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"((sinh(x)-coshc(x))./x).*var(1,0)" },
        { "tanhc"           ,1,1 ,0,1 ,1,0,NULL ,tanhc           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_tanhc    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"((sech(x).^2-tanhc(x))./x).*var(1,0)" },
        { "versh"           ,1,1 ,0,1 ,1,0,NULL ,versh           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_versh    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,1-cosh(x)).*var(1,0)" },
        { "coversh"         ,1,1 ,0,1 ,1,0,NULL ,coversh         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_coversh  ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,1-sinh(x)).*var(1,0)" },
        { "havh"            ,1,1 ,0,1 ,1,0,NULL ,havh            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_havh     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,versh(x)/2).*var(1,0)" },
        { "excosech"        ,1,1 ,0,1 ,1,0,NULL ,excosech        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_excosech ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,cosech(x)-1).*var(1,0)" },
        { "exsech"          ,1,1 ,0,1 ,1,0,NULL ,exsech          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_exsech   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,sech(x)-1).*var(1,0)" },
        { "aversh"          ,1,1 ,0,1 ,1,0,NULL ,aversh          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_aversh   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Aversh"        ,-1,1 ,1 ,NULL ,"realDeriv(0,0,acosh(x+1)).*var(1,0)" },
        { "Aversh"          ,1,1 ,0,1 ,1,0,NULL ,Aversh          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Aversh   ,NULL   ,NULL,NULL,NULL,NULL,1 ,"aversh"        ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Acosh(x+1)).*var(1,0)" },
        { "acovrsh"         ,1,1 ,0,1 ,1,0,NULL ,acovrsh         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_acovrsh  ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,asinh(x+1)).*var(1,0)" },
        { "ahavh"           ,1,1 ,0,1 ,1,0,NULL ,ahavh           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_ahavh    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Ahavh"         ,-1,1 ,1 ,NULL ,"realDeriv(0,0,aversh(2*x)).*var(1,0)" },
        { "Ahavh"           ,1,1 ,0,1 ,1,0,NULL ,Ahavh           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Ahavh    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"ahavh"         ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Aversh(2*x)).*var(1,0)" },
        { "aexcosech"       ,1,1 ,0,1 ,1,0,NULL ,aexcosech       ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_aexcosech,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,acosech(x+1)).*var(1,0)" },
        { "aexsech"         ,1,1 ,0,1 ,1,0,NULL ,aexsech         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_aexsech  ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Aexsech"       ,-1,1 ,1 ,NULL ,"realDeriv(0,0,asech(x+1)).*var(1,0)" },
        { "Aexsech"         ,1,1 ,0,1 ,1,0,NULL ,Aexsech         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Aexsech  ,NULL   ,NULL,NULL,NULL,NULL,1 ,"aexsech"       ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Asech(x+1)).*var(1,0)" },
        { "sigm"            ,1,1 ,0,1 ,1,0,NULL ,sigm            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_sigm     ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,einv(1+exp(x))).*var(1,0)" },
        { "gd"              ,1,1 ,0,1 ,1,0,NULL ,gd              ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_gd       ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,2*atan(tanh(x/2))).*var(1,0)" },
        { "asigm"           ,1,1 ,0,1 ,1,0,NULL ,asigm           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_asigm    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Asigm"         ,-1,1 ,1 ,NULL ,"realDeriv(0,0,log(einv(x)-1)).*var(1,0)" },
        { "Asigm"           ,1,1 ,0,1 ,1,0,NULL ,Asigm           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Asigm    ,NULL   ,NULL,NULL,NULL,NULL,1 ,"asigm"         ,-1,1 ,1 ,NULL ,"realDeriv(0,0,Log(einv(x)-1)).*var(1,0)" },
        { "agd"             ,1,1 ,0,1 ,1,0,NULL ,agd             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_agd      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Agd"           ,-1,1 ,1 ,NULL ,"realDeriv(0,0,2*atanh(tan(x/2))).*var(1,0)" },
        { "Agd"             ,1,1 ,0,1 ,1,0,NULL ,Agd             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_Agd      ,NULL   ,NULL,NULL,NULL,NULL,1 ,"agd"           ,-1,1 ,1 ,NULL ,"realDeriv(0,0,2*Atanh(tan(x/2))).*var(1,0)" },
        { "bern"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,bern         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,3 ,3 ,NULL ,"((size(x)-1)*bern(derefv(x,ivect(1,1,size(x)-1))-derefv(x,ivect(0,1,size(x)-2)),y)).*var(1,1)" },
        { "funcv"           ,1,0 ,0,0 ,1,0,NULL ,bernv           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "rkhsv"           ,3,7 ,0,7 ,1,0,NULL ,bernv           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "bernv"           ,1,1 ,0,1 ,1,0,NULL ,bernv           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "normDistr"       ,1,1 ,0,1 ,1,0,NULL ,normDistr       ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"realDeriv(0,0,0.398942280401*exp((-x^2)/2)).*var(1,0)" },
        { "polyDistr"       ,2,3 ,0,3 ,1,0,NULL ,NULL            ,polyDistr    ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"PolyDistr"     ,-1,3 ,3 ,NULL ,"realDeriv(0,0,(x*sqrt(gamma(3/x)/gamma(1/x))/(2*gamma(1/x)))*exp(-(sqrt(gamma(3/x)/gamma(1/x))^x)*(y^x)))*var(1,0)+realDeriv(0,1,(x*sqrt(gamma(3/x)/gamma(1/x))/(2*gamma(1/x)))*exp(-(sqrt(gamma(3/x)/gamma(1/x))^x)*(y^x)))*var(1,1)" },
        { "PolyDistr"       ,2,3 ,0,3 ,1,0,NULL ,NULL            ,PolyDistr    ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,3 ,"polyDistr"     ,-1,3 ,3 ,NULL ,"realDeriv(0,0,(x*Sqrt(gamma(3/x)/gamma(1/x))/(2*gamma(1/x)))*exp(-(Sqrt(gamma(3/x)/gamma(1/x))^x)*(y^x)))*var(1,0)+realDeriv(0,1,(x*Sqrt(gamma(3/x)/gamma(1/x))/(2*gamma(1/x)))*exp(-(Sqrt(gamma(3/x)/gamma(1/x))^x)*(y^x)))*var(1,1)" },
        { "gamma"           ,1,1 ,0,1 ,1,0,NULL ,gamma           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"(gamma(x).*psi(x)).*var(1,0)" },
        { "lngamma"         ,1,1 ,0,1 ,1,0,NULL ,lngamma         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"psi(x).*var(1,0)" },
        { "psi"             ,1,1 ,0,1 ,1,0,NULL ,psi             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"psi_n(1,x).*var(1,0)" },
        { "psi_n"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,psi_n        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,2 ,"~"             ,-1,3 ,2 ,NULL ,"psi_n(x-1,y).*var(1,1)" },
        { "gami"            ,2,3 ,0,3 ,1,0,NULL ,NULL            ,gami         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"E: \"Gradient of gami not defined\"" }, 
        { "gamic"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,gamic        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"E: \"Gradient of gamic not defined\"" },
        { "erf"             ,1,1 ,0,1 ,1,0,NULL ,erf             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"((2/sqrt(pi()))*exp(-x.^2)).*var(1,0)" },
        { "erfc"            ,1,1 ,0,1 ,1,0,NULL ,erfc            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"(-(2/sqrt(pi()))*exp(-x.^2)).*var(1,0)" },
        { "dawson"          ,1,1 ,0,1 ,1,0,NULL ,dawson          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"(1-(2*(x.*dawson(x)))).*var(1,0)" },
        { "rint"            ,1,1 ,0,1 ,1,0,NULL ,rint            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ceil"            ,1,1 ,0,1 ,1,0,NULL ,ceil            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "floor"           ,1,1 ,0,1 ,1,0,NULL ,floor           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "outerProd"       ,2,3 ,0,3 ,1,0,NULL ,NULL            ,outerProd    ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,-3,"OuterProd"     ,-1,3 ,3 ,NULL ,"outerProd(var(1,0),y)+outerProd(x,var(1,1))" },
        { "OuterProd"       ,2,3 ,0,3 ,1,0,NULL ,NULL            ,OuterProd    ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,-3,"outerprod"     ,-1,3 ,3 ,NULL ,"OuterProd(var(1,0),y)+OuterProd(x,var(1,1))" },
        { "fourProd"        ,4,15,0,15,1,0,NULL ,NULL            ,NULL         ,NULL         ,fourProd,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,15,"~"             ,-1,15,15,NULL ,"fourProd(var(1,0),y,z,v)+fourProd(x,var(1,1),z,v)+fourProd(x,y,var(1,2),v)+fourProd(x,y,z,var(1,3))" },
        { "trans"           ,1,1 ,0,1 ,1,0,NULL ,trans           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,1 ,NULL ,"trans(var(1,0))" },
        { "det"             ,1,1 ,0,1 ,1,0,NULL ,det             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,0 ,NULL ,"E: \"Gradient of det not defined\"" },
        { "trace"           ,1,1 ,0,1 ,1,0,NULL ,trace           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,1 ,NULL ,"trace(var(1,0))" },
        { "miner"           ,3,7 ,0,7 ,1,0,NULL ,NULL            ,NULL         ,miner        ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"E: \"Gradient of miner not defined\""  },
        { "cofactor"        ,3,7 ,0,7 ,1,0,NULL ,NULL            ,NULL         ,cofactor     ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"E: \"Gradient of cofactor not defined\""  },
        { "adj"             ,1,1 ,0,1 ,1,0,NULL ,adj             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,0 ,NULL ,"E: \"Gradient of adj not defined\"" },
        { "max"             ,1,0 ,1,1 ,1,0,NULL ,max             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argmax(x))" },
        { "min"             ,1,0 ,1,1 ,1,0,NULL ,min             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argmin(x))" },
        { "maxdiag"         ,1,0 ,1,1 ,1,0,NULL ,maxdiag         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argmaxdiag(x))" },
        { "mindiag"         ,1,0 ,1,1 ,1,0,NULL ,mindiag         ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argmindiag(x))" },
        { "argmax"          ,1,0 ,1,1 ,1,0,NULL ,argmax          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "argmin"          ,1,0 ,1,1 ,1,0,NULL ,argmin          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "argmaxdiag"      ,1,0 ,1,1 ,1,0,NULL ,argmaxdiag      ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "argmindiag"      ,1,0 ,1,1 ,1,0,NULL ,argmindiag      ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "allargmax"       ,1,0 ,1,1 ,1,0,NULL ,allargmax       ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "allargmin"       ,1,0 ,1,1 ,1,0,NULL ,allargmin       ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "allargmaxdiag"   ,1,0 ,1,1 ,1,0,NULL ,allargmaxdiag   ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "allargmindiag"   ,1,0 ,1,1 ,1,0,NULL ,allargmindiag   ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "maxabs"          ,1,0 ,1,1 ,1,0,NULL ,maxabs          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argmaxabs(x))" },
        { "minabs"          ,1,0 ,1,1 ,1,0,NULL ,minabs          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argminabs(x))" },
        { "maxabsdiag"      ,1,0 ,1,1 ,1,0,NULL ,maxabsdiag      ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argmaxabsdiag(x))" },
        { "minabsdiag"      ,1,0 ,1,1 ,1,0,NULL ,minabsdiag      ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argminabsdiag(x))" },
        { "argmaxabs"       ,1,0 ,1,1 ,1,0,NULL ,argmaxabs       ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "argminabs"       ,1,0 ,1,1 ,1,0,NULL ,argminabs       ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "argmaxabsdiag"   ,1,0 ,1,1 ,1,0,NULL ,argmaxabsdiag   ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "argminabsdiag"   ,1,0 ,1,1 ,1,0,NULL ,argminabsdiag   ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "allargmaxabs"    ,1,0 ,1,1 ,1,0,NULL ,allargmaxabs    ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "allargminabs"    ,1,0 ,1,1 ,1,0,NULL ,allargminabs    ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "allargmaxabsdiag",1,0 ,1,1 ,1,0,NULL ,allargmaxabsdiag,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "allargminabsdiag",1,0 ,1,1 ,1,0,NULL ,allargminabsdiag,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "sum"             ,1,1 ,0,1 ,1,0,NULL ,sum             ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,1 ,NULL ,"sum(var(1,0))" },
        { "prod"            ,1,1 ,0,1 ,1,0,NULL ,prod            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"Prod"          ,-1,1 ,0 ,NULL ,"realDeriv(0,0,prod(sgn(var(0,0)))*exp(sum(log(eabs2(var(0,0))))))*var(1,0)" },
        { "Prod"            ,1,1 ,0,1 ,1,0,NULL ,Prod            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"prod"          ,-1,1 ,0 ,NULL ,"realDeriv(0,0,prod(sgn(var(0,0)))*exp(sum(log(eabs2(conj(var(0,0)))))))*var(1,0)" },
        { "mean"            ,1,1 ,0,1 ,1,0,NULL ,mean            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,1 ,NULL ,"mean(var(1,0))" },
        { "median"          ,1,0 ,1,1 ,1,0,NULL ,median          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"deref(var(1,0),argmedian(x))" },
        { "argmedian"       ,1,0 ,1,1 ,1,0,NULL ,argmedian       ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "deref"           ,2,3 ,0,3 ,1,0,NULL ,NULL            ,deref        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,2 ,1 ,NULL ,"deref(var(1,0),y)" },
        { "derefv"          ,2,3 ,0,3 ,1,0,NULL ,NULL            ,derefv       ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,2 ,1 ,NULL ,"derefv(var(1,0),y)" },
        { "derefm"          ,3,7 ,0,7 ,1,0,NULL ,NULL            ,NULL         ,derefm       ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,2 ,1 ,NULL ,"derefm(var(1,0),y,z)" },
        { "derefa"          ,2,3 ,0,3 ,1,0,NULL ,NULL            ,derefa       ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,2 ,1 ,NULL ,"derefa(var(1,0),y)"  },
        { "collapse"        ,1,1 ,0,1 ,1,0,NULL ,collapse        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,0 ,1 ,NULL ,"collapse(var(1,0))" },
        { "zeta"            ,1,1 ,0,1 ,1,0,NULL ,zeta            ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"E: \"Gradient of zeta not defined\"" },
        { "lambertW"        ,1,1 ,0,1 ,1,0,NULL ,lambertW        ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"(lambertW(x)./(x.*(1+lambertW(x)))).*var(1,0)" },
        { "lambertWx"       ,1,1 ,0,1 ,1,0,NULL ,lambertWx       ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,1 ,"~"             ,-1,1 ,1 ,NULL ,"(lambertWx(x)./(x.*(1+lambertWx(x)))).*var(1,0)" },
        { "fnA"             ,2,3 ,0,3 ,1,1,NULL ,NULL            ,fnA          ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "fnB"             ,3,3 ,4,7 ,1,1,NULL ,NULL            ,NULL         ,fnB          ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,7 ,4 ,NULL ,"dfnB(x,y,z,1).*var(1,2)" },
        { "fnC"             ,4,3 ,12,15,1,1,NULL ,NULL           ,NULL         ,NULL         ,fnC     ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,15,12,NULL ,"(dfnC(x,y,z,1,var(0,3),0).*var(1,2))+(dfnC(x,y,z,0,var(0,3),1).*var(1,3))" },
        { "dfnB"            ,4,11,4,15,1,1,NULL ,NULL            ,NULL         ,NULL         ,dfnB    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,15,4 ,NULL ,"dfnB(x,y,z,var(0,3)+1).*var(1,2)" },
        { "dfnC"            ,6,43,20,63,1,1,NULL ,NULL           ,NULL         ,NULL         ,NULL    ,NULL      ,dfnC,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,63,20,NULL ,"(dfnC(x,y,z,var(0,3)+1,var(0,4),var(0,5)).*var(1,2))+(dfnC(x,y,z,var(0,3),var(0,4),var(0,5)+1).*var(1,4))" },
        { "efnB"            ,3,3 ,4,7 ,1,1,NULL ,NULL            ,NULL         ,efnB         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,7 ,4 ,NULL ,"edfnB(x,y,z,1).*var(1,2)" },
        { "efnC"            ,4,3 ,12,15,1,1,NULL ,NULL           ,NULL         ,NULL         ,efnC    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,15,12,NULL ,"(edfnC(x,y,z,1,var(0,3),0).*var(1,2))+(edfnC(x,y,z,0,var(0,3),1).*var(1,3))" },
        { "edfnB"           ,4,11,4,15,1,1,NULL ,NULL            ,NULL         ,NULL         ,edfnB   ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,15,4 ,NULL ,"edfnB(x,y,z,var(0,3)+1).*var(1,2)" },
        { "edfnC"           ,6,43,20,63,1,1,NULL ,NULL           ,NULL         ,NULL         ,NULL    ,NULL     ,edfnC,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,63,20,NULL ,"(edfnC(x,y,z,var(0,3)+1,var(0,4),var(0,5)).*var(1,2))+(edfnC(x,y,z,var(0,3),var(0,4),var(0,5)+1).*var(1,4))" },
        { "irand"           ,1,1 ,0,1 ,1,2,NULL ,irand           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "urand"           ,2,3 ,0,3 ,1,2,NULL ,NULL            ,urand        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"urand(var(1,0),var(1,1))" },
        { "grand"           ,2,0 ,3,3 ,1,2,NULL ,NULL            ,grand        ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"grand(var(1,0),var(1,1))" }, 
        { "testfn"          ,2,2 ,3,3 ,1,0,NULL ,NULL            ,testfn       ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "testfnA"         ,3,2 ,7,7 ,1,0,NULL ,NULL            ,NULL         ,testfnA      ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "partestfn"       ,3,4 ,7,7 ,1,0,NULL ,NULL            ,NULL         ,partestfn    ,NULL    ,NULL      ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "partestfnA"      ,4,4,15,15,1,0,NULL ,NULL            ,NULL         ,NULL         ,partestfnA,NULL    ,NULL,NULL    ,NULL        ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isvnan"          ,1,1 ,0,1 ,1,0,NULL ,isvnan          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isvnan   ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isinf"           ,1,1 ,0,1 ,1,0,NULL ,isinf           ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isinf    ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "ispinf"          ,1,1 ,0,1 ,1,0,NULL ,ispinf          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_ispinf   ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" },
        { "isninf"          ,1,1 ,0,1 ,1,0,NULL ,isninf          ,NULL         ,NULL         ,NULL    ,NULL      ,NULL,NULL    ,OP_isninf   ,NULL   ,NULL,NULL,NULL,NULL,0 ,"~"             ,-1,0 ,0 ,NULL ,"0" } };


// prod(x) = prod(sgn(x))*prod(eabs2(x))
//         = prod(sgn(x))*exp(sum(log(eabs2(x))))
// Prod(x) = prod(sgn(x))*prod(eabs2(conj(x)))
//         = prod(sgn(x))*exp(sum(log(eabs2(conj(x)))))
//
// Derivatives:
//
// realDeriv(prod(...)) = realDeriv(prod(sgn(var(0,0)))*exp(sum(log(eabs2(var(0,0)))))
// realDeriv(Prod(...)) = realDeriv(prod(sgn(var(0,0)))*exp(sum(log(eabs2(conj(var(0,0))))))

fninfoblock *getfninfo(void);
fninfoblock *getfninfo(void)
{
    return qqqfninfo;
}

const char *getfnname(int fnnameind)
{
    return ((const_cast<fninfoblock *>(getfninfo()))[fnnameind]).fnname;
}

int getfnind(const std::string &fnname)
{
    int i,ires = -1;

    for ( i = 0 ; i < NUMFNDEF ; i++ )
    {
        if ( fnname == getfnname(i) )
        {
            ires = i;
            break;
        }
    }

    return ires;
}


int getfnindConj(int fnInd)
{
    int res = (const_cast<fninfoblock *>(getfninfo()))[fnInd].fnconjind;

    if ( res == -1 )
    {
        (const_cast<fninfoblock *>(getfninfo()))[fnInd].fnconjind = getfnind((const_cast<fninfoblock *>(getfninfo()))[fnInd].conjfnname);
        res = (const_cast<fninfoblock *>(getfninfo()))[fnInd].fnconjind;
    }

    return res;
}

const fninfoblock *getfninfo(int ires)
{
    // Can't mutex-lock this!  The problem is that this is recursive, and
    // not in an easy way, so mutex will be self-blocking.  The work-around
    // is that in *all* multi-threaded operations initgentype() *must* be
    // called before any threaded operations begin.

    //svmvolatile static svm_mutex eyelock;
    //svm_mutex_lock(&eyelock);

    static gentype blind(0); // never modified, so no need for this to be volatile
    fninfoblock *res = NULL;

    if ( ires != -1 )
    {
        if ( getfninfo()[ires].realderiv == NULL )
        {
            getfninfo()[ires].realderiv = &blind;
            // The above line is needed to stop getfninfo entering an
            // infinite loop.  For example, if the derivative of add is
            // being generated here then the derivative itself includes an
            // instance of add, which will result in a call back to
            // getfninfo, and if realderiv is still NULL at this point the
            // process will repeat in an infinite loop. Including the above
            // line means that this can only be called once for each
            // function.
            //
            // ASIDE: it is important to only include basic functions (ie.
            // those which have a derivative defined that doesn't include
            // the realDeriv function) in non-basic function derivative
            // strings (is within the realDeriv function).

            gentype *temp;

            MEMNEW(temp,gentype(getfninfo()[ires].realderivfn));

            getfninfo()[ires].realderiv = temp;
        }

        res = const_cast<fninfoblock *>(&(getfninfo()[ires]));
    }

    //svm_mutex_unlock(&eyelock);

    return res;
}

void exitgentype(void);
void exitgentype(void)
{
    int i;

//    errstream() << "Cleaning up after gentype exit\n";
    static gentype blind(0); // never modified, so no need for this to be volatile

    for ( i = 0 ; i < NUMFNDEF ; i++ )
    {
//        errstream() << i << " (" << getfninfo()[i].fnname << "): ";
        if ( getfninfo()[i].realderiv )
        {
//            errstream() << "Deleting " << *(getfninfo()[i].realderiv) << "... ";
            MEMDEL(getfninfo()[i].realderiv);
            getfninfo()[i].realderiv = &blind; // Not null or it just gets created again!
//            errstream() << "done\n";
        }
    }

    for ( i = 0 ; i < NUMFNDEF ; i++ )
    {
        getfninfo()[i].realderiv = NULL; // NULL is safe here.
    }

    return;
}

void initgentype(void)
{
    svm_atexit(exitgentype,"gentype");

//    errstream() << "Constructing gentype derivatives\n";

    int i;

    for ( i = 0 ; i < NUMFNDEF ; i++ )
    {
        getfninfo(i);
    }

//    for ( i = 0 ; i < NUMFNDEF ; i++ )
//    {
//        errstream() << i << " (" << getfninfo()[i].fnname << "): " << *(getfninfo()[i].realderiv) << " done.\n";
//    }

    return;
}












































// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
//
// Helper functions for operators start here
//
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================




// Check if arguments are compatible for addition/subtraction.  Returns:
//
// 0 = no
// 1 = yes, result integer
// 2 = yes, result real
// 3 = yes, result anion
// 4 = yes, result vector
// 5 = yes, result matrix
// 6 = yes, result string
// 8 = yes, result set
// 9 = yes, result dgraph
//10 = yes, result null
//
// Notes: adding a string to anything gives a string
//        7 means leave as gentype
//        isstrict: set for logical operations

int checkAddCompat(const gentype &a, const gentype &b, int &acast, int &bcast, int isstrict = 0);
int checkAddCompat(const gentype &a, const gentype &b, int &acast, int &bcast, int isstrict)
{
    if ( a.isValNull() )
    {
             if ( b.isValNull()                ) { acast = 10; bcast = 10; return 10; }
        else if ( b.isValInteger()             ) { acast = 1;  bcast = 1;  return 1;  }
        else if ( b.isValReal()                ) { acast = 2;  bcast = 2;  return 2;  }
        else if ( b.isValAnion()               ) { acast = 3;  bcast = 3;  return 3;  }
        else if ( b.isValVector() && !isstrict ) { acast = 7;  bcast = 4;  return 4;  }
        else if ( b.isValMatrix() && !isstrict ) { acast = 7;  bcast = 5;  return 5;  }
        else if ( b.isValSet()    && !isstrict ) { acast = 7;  bcast = 8;  return 8;  }
        else if ( b.isValString()              ) { acast = 6;  bcast = 6;  return 6;  }
        else                                     { acast = 0;  bcast = 0;  return 0;  }
    }

    else if ( a.isValInteger() )
    {
             if ( b.isValNull()                ) { acast = 1; bcast = 1; return 1; }
        else if ( b.isValInteger()             ) { acast = 1; bcast = 1; return 1; }
        else if ( b.isValReal()                ) { acast = 2; bcast = 2; return 2; }
        else if ( b.isValAnion()               ) { acast = 3; bcast = 3; return 3; }
        else if ( b.isValVector() && !isstrict ) { acast = 7; bcast = 4; return 4; }
        else if ( b.isValMatrix() && !isstrict ) { acast = 7; bcast = 5; return 5; }
        else if ( b.isValSet()    && !isstrict ) { acast = 7; bcast = 8; return 8; }
        else if ( b.isValString()              ) { acast = 6; bcast = 6; return 6; }
        else                                     { acast = 0; bcast = 0; return 0; }
    }

    else if ( a.isValReal() )
    {
             if ( b.isValNull()                ) { acast = 2; bcast = 2; return 2; }
        else if ( b.isValInteger()             ) { acast = 2; bcast = 2; return 2; }
        else if ( b.isValReal()                ) { acast = 2; bcast = 2; return 2; }
        else if ( b.isValAnion()               ) { acast = 3; bcast = 3; return 3; }
        else if ( b.isValVector() && !isstrict ) { acast = 7; bcast = 4; return 4; }
        else if ( b.isValMatrix() && !isstrict ) { acast = 7; bcast = 5; return 5; }
        else if ( b.isValSet()    && !isstrict ) { acast = 7; bcast = 8; return 8; }
        else if ( b.isValString()              ) { acast = 6; bcast = 6; return 6; }
        else                                     { acast = 0; bcast = 0; return 0; }
    }

    else if ( a.isValAnion() )
    {
             if ( b.isValNull()                ) { acast = 3; bcast = 3; return 3; }
        else if ( b.isValInteger()             ) { acast = 3; bcast = 3; return 3; }
        else if ( b.isValReal()                ) { acast = 3; bcast = 3; return 3; }
        else if ( b.isValAnion()               ) { acast = 3; bcast = 3; return 3; }
        else if ( b.isValVector() && !isstrict ) { acast = 7; bcast = 4; return 4; }
        else if ( b.isValMatrix() && !isstrict ) { acast = 7; bcast = 5; return 5; }
        else if ( b.isValSet()    && !isstrict ) { acast = 7; bcast = 8; return 8; }
        else if ( b.isValString()              ) { acast = 6; bcast = 6; return 6; }
        else                                     { acast = 0; bcast = 0; return 0; }
    }

    else if ( a.isValVector() )
    {
             if ( b.isValNull()    && !isstrict ) { acast = 4; bcast = 7; return 4; }
        else if ( b.isValInteger() && !isstrict ) { acast = 4; bcast = 7; return 4; }
        else if ( b.isValReal()    && !isstrict ) { acast = 4; bcast = 7; return 4; }
        else if ( b.isValAnion()   && !isstrict ) { acast = 4; bcast = 7; return 4; }

        else if ( b.isValVector() )
        {
            if ( a.size() == b.size() ) { acast = 4; bcast = 4; return 4; }
            else                        { acast = 0; bcast = 0; return 0; }
	}

	else if ( b.isValMatrix() )
	{
            if ( ( b.numRows() == a.size() ) && ( b.numCols() == 1 ) ) { acast = 5; bcast = 5; return 5; }
            else                                                       { acast = 0; bcast = 0; return 0; }
	}

        else if ( b.isValSet() && !isstrict ) { acast = 7; bcast = 8; return 8; }
        else if ( b.isValString()           ) { acast = 6; bcast = 6; return 6; }
        else                                  { acast = 0; bcast = 0; return 0; }
    }

    else if ( a.isValMatrix() )
    {
             if ( b.isValNull()    && !isstrict ) { acast = 5; bcast = 7; return 5; }
        else if ( b.isValInteger() && !isstrict ) { acast = 5; bcast = 7; return 5; }
        else if ( b.isValReal()    && !isstrict ) { acast = 5; bcast = 7; return 5; }
        else if ( b.isValAnion()   && !isstrict ) { acast = 5; bcast = 7; return 5; }

        else if ( b.isValVector() )
        {
            if ( ( a.numRows() == b.size() ) && ( a.numCols() == 1 ) ) { acast = 5; bcast = 5; return 5; }
            else                                                       { acast = 0; bcast = 0; return 0; }
	}

	else if ( b.isValMatrix() )
	{
            if ( ( a.numRows() == b.numRows() ) && ( a.numCols() == b.numCols() ) ) { acast = 5; bcast = 5; return 5; }
            else                                                                    { acast = 0; bcast = 0; return 0; }
	}

        else if ( b.isValSet() && !isstrict ) { acast = 7; bcast = 8; return 8; }
        else if ( b.isValString()           ) { acast = 6; bcast = 6; return 6; }
        else                                  { acast = 0; bcast = 0; return 0; }
    }

    else if ( a.isValSet() )
    {
             if ( b.isValNull()    && !isstrict ) { acast = 8; bcast = 7; return 8; }
        else if ( b.isValInteger() && !isstrict ) { acast = 8; bcast = 7; return 8; }
        else if ( b.isValReal()    && !isstrict ) { acast = 8; bcast = 7; return 8; }
        else if ( b.isValAnion()   && !isstrict ) { acast = 8; bcast = 7; return 8; }
        else if ( b.isValVector()  && !isstrict ) { acast = 8; bcast = 7; return 8; }
        else if ( b.isValMatrix()  && !isstrict ) { acast = 8; bcast = 7; return 8; }
        else if ( b.isValSet()                  ) { acast = 8; bcast = 8; return 8; }
        else if ( b.isValString()  && !isstrict ) { acast = 8; bcast = 7; return 8; }
        else                                      { acast = 0; bcast = 0; return 0; }
    }

    else if ( a.isValDgraph() )
    {
        // In this case, we can only "add" in the logical comparison sense,
        // not in the mathematical sense.  Hence isstrict, not !isstrict.

        if ( b.isValDgraph() && isstrict ) { acast = 9; bcast = 9; return 9; }
        else                               { acast = 0; bcast = 0; return 0; }
    }

    else if ( a.isValString() )
    {
             if ( b.isValNull()    ) { acast = 6; bcast = 6; return 6; }
        else if ( b.isValInteger() ) { acast = 6; bcast = 6; return 6; }
        else if ( b.isValReal()    ) { acast = 6; bcast = 6; return 6; }
        else if ( b.isValAnion()   ) { acast = 6; bcast = 6; return 6; }
        else if ( b.isValVector()  ) { acast = 6; bcast = 6; return 6; }
        else if ( b.isValMatrix()  ) { acast = 6; bcast = 6; return 6; }
        else if ( b.isValSet()     ) { acast = 7; bcast = 8; return 8; }
        else if ( b.isValString()  ) { acast = 6; bcast = 6; return 6; }
        else                         { acast = 0; bcast = 0; return 0; }
    }

    return 0;
}

void constructError(gentype &res, const std::string &errstrval)
{
    res.makeError(errstrval);

    return;
}

void constructError(const gentype &a, gentype &res, const std::string &errstrval)
{
    std::string errstr = errstrval;

    if ( a.isValError() )
    {
	errstr += ",(";
	errstr += a.cast_string(0);
        errstr += ")";
    }

    res.makeError(errstr);

    return;
}

void constructError(const gentype &a, const gentype &b, gentype &res, const std::string &errstrval)
{
    std::string errstr = errstrval;

    if ( a.isValError() )
    {
	errstr += ",(";
	errstr += a.cast_string(0);
        errstr += ")";
    }

    if ( b.isValError() )
    {
	errstr += ",(";
	errstr += b.cast_string(0);
        errstr += ")";
    }

    res.makeError(errstr);

    return;
}

void constructError(const gentype &a, const gentype &b, const gentype &c, gentype &res, const std::string &errstrval)
{
    std::string errstr = errstrval;

    if ( a.isValError() )
    {
	errstr += ",(";
	errstr += a.cast_string(0);
        errstr += ")";
    }

    if ( b.isValError() )
    {
	errstr += ",(";
	errstr += b.cast_string(0);
        errstr += ")";
    }

    if ( c.isValError() )
    {
	errstr += ",(";
	errstr += c.cast_string(0);
        errstr += ")";
    }

    res.makeError(errstr);

    return;
}

void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, gentype &res, const std::string &errstrval)
{
    std::string errstr = errstrval;

    if ( a.isValError() )
    {
	errstr += ",(";
	errstr += a.cast_string(0);
        errstr += ")";
    }

    if ( b.isValError() )
    {
	errstr += ",(";
	errstr += b.cast_string(0);
        errstr += ")";
    }

    if ( c.isValError() )
    {
	errstr += ",(";
	errstr += c.cast_string(0);
        errstr += ")";
    }

    if ( d.isValError() )
    {
	errstr += ",(";
	errstr += d.cast_string(0);
        errstr += ")";
    }

    res.makeError(errstr);

    return;
}

void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e, gentype &res, const std::string &errstrval)
{
    std::string errstr = errstrval;

    if ( a.isValError() )
    {
	errstr += ",(";
	errstr += a.cast_string(0);
        errstr += ")";
    }

    if ( b.isValError() )
    {
	errstr += ",(";
	errstr += b.cast_string(0);
        errstr += ")";
    }

    if ( c.isValError() )
    {
	errstr += ",(";
	errstr += c.cast_string(0);
        errstr += ")";
    }

    if ( d.isValError() )
    {
	errstr += ",(";
	errstr += d.cast_string(0);
        errstr += ")";
    }

    if ( e.isValError() )
    {
	errstr += ",(";
        errstr += e.cast_string(0);
        errstr += ")";
    }

    res.makeError(errstr);

    return;
}

void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e, const gentype &f, gentype &res, const std::string &errstrval)
{
    std::string errstr = errstrval;

    if ( a.isValError() )
    {
	errstr += ",(";
	errstr += a.cast_string(0);
        errstr += ")";
    }

    if ( b.isValError() )
    {
	errstr += ",(";
	errstr += b.cast_string(0);
        errstr += ")";
    }

    if ( c.isValError() )
    {
	errstr += ",(";
	errstr += c.cast_string(0);
        errstr += ")";
    }

    if ( d.isValError() )
    {
	errstr += ",(";
	errstr += d.cast_string(0);
        errstr += ")";
    }

    if ( e.isValError() )
    {
	errstr += ",(";
        errstr += e.cast_string(0);
        errstr += ")";
    }

    if ( f.isValError() )
    {
	errstr += ",(";
        errstr += f.cast_string(0);
        errstr += ")";
    }

    res.makeError(errstr);

    return;
}

void constructError(gentype &res, const char *errstr)
{
    std::string errstralt = errstr;

    constructError(res,errstralt);

    return;
}

void constructError(const gentype &a, gentype &res, const char *errstr)
{
    std::string errstralt = errstr;

    constructError(a,res,errstralt);

    return;
}

void constructError(const gentype &a, const gentype &b, gentype &res, const char *errstr)
{
    std::string errstralt = errstr;

    constructError(a,b,res,errstralt);

    return;
}

void constructError(const gentype &a, const gentype &b, const gentype &c, gentype &res, const char *errstr)
{
    std::string errstralt = errstr;

    constructError(a,b,c,res,errstralt);

    return;
}

void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, gentype &res, const char *errstr)
{
    std::string errstralt = errstr;

    constructError(a,b,c,d,res,errstralt);

    return;
}

void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e, gentype &res, const char *errstr)
{
    std::string errstralt = errstr;

    constructError(a,b,c,d,e,res,errstralt);

    return;
}

void constructError(const gentype &a, const gentype &b, const gentype &c, const gentype &d, const gentype &e, const gentype &f, gentype &res, const char *errstr)
{
    std::string errstralt = errstr;

    constructError(a,b,c,d,e,f,res,errstralt);

    return;
}




















































// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
//
// Operators start here
//
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================


// Logical operators

int gentype::iseq(const gentype &b) const
{
    if ( isValEqnDir() || b.isValEqnDir() )
    {
        // NB: this is not accurate.  It can report a != b when in fact
        //     a == b.  For example x+y != y+x according to this code.  At
        //     some point this logic needs to be corrected, but it is highly
        //     nontrivial and the benefit is marginal.

        if ( isValEqnDir() && b.isValEqnDir() )
        {
            if ( fnnameind == b.fnnameind )
            {
                return *eqnargs == *(b.eqnargs);
            }
        }

        return 0;
    }
    
    int res = 0;
    int acast = 0;
    int bcast = 0;

    if ( ( res = checkAddCompat(*this,b,acast,bcast,1) ) )
    {
             if ( ( acast == 1  ) && ( bcast == 1  ) ) { res = ( cast_int(0)       == b.cast_int(0)       ); }
        else if ( ( acast == 2  ) && ( bcast == 2  ) ) { res = ( cast_double(0)    == b.cast_double(0)    ); }
        else if ( ( acast == 3  ) && ( bcast == 3  ) ) { res = ( cast_anion(0)     == b.cast_anion(0)     ); }
        else if ( ( acast == 4  ) && ( bcast == 4  ) ) { res = ( cast_vector(0)    == b.cast_vector(0)    ); }
        else if ( ( acast == 5  ) && ( bcast == 5  ) ) { res = ( cast_matrix(0)    == b.cast_matrix(0)    ); }
        else if ( ( acast == 8  ) && ( bcast == 8  ) ) { res = ( cast_set(0)       == b.cast_set(0)       ); }
        else if ( ( acast == 9  ) && ( bcast == 9  ) ) { res = ( cast_dgraph(0)    == b.cast_dgraph(0)    ); }
        else if ( ( acast == 10 ) && ( bcast == 10 ) ) { res = 1;                                            }
        else                                           { res = ( cast_string(0)    == b.cast_string(0)    ); }
    }

    return res;
}

int operator!=(const gentype &a, const gentype &b)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        return !(a==b);
    }

    int res = 0;
    int acast = 0;
    int bcast = 0;

    if ( ( res = checkAddCompat(a,b,acast,bcast,1) ) )
    {
             if ( ( acast == 1  ) && ( bcast == 1  ) ) { res = ( a.cast_int(0)       != b.cast_int(0)       ); }
        else if ( ( acast == 2  ) && ( bcast == 2  ) ) { res = ( a.cast_double(0)    != b.cast_double(0)    ); }
        else if ( ( acast == 3  ) && ( bcast == 3  ) ) { res = ( a.cast_anion(0)     != b.cast_anion(0)     ); }
        else if ( ( acast == 4  ) && ( bcast == 4  ) ) { res = ( a.cast_vector(0)    != b.cast_vector(0)    ); }
        else if ( ( acast == 5  ) && ( bcast == 5  ) ) { res = ( a.cast_matrix(0)    != b.cast_matrix(0)    ); }
        else if ( ( acast == 8  ) && ( bcast == 8  ) ) { res = ( a.cast_set(0)       != b.cast_set(0)       ); }
        else if ( ( acast == 9  ) && ( bcast == 9  ) ) { res = ( a.cast_dgraph(0)    != b.cast_dgraph(0)    ); }
        else if ( ( acast == 10 ) && ( bcast == 10 ) ) { res = 0;                                            }
        else                                           { res = ( a.cast_string(0)    != b.cast_string(0)    ); }
    }

    return res;
}

int operator<(const gentype &a, const gentype &b)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        return 0;
    }

    int res = 0;
    int acast = 0;
    int bcast = 0;

    if ( ( res = checkAddCompat(a,b,acast,bcast,1) ) )
    {
             if ( ( acast == 1  ) && ( bcast == 1  ) ) { res = (  a.cast_int(0)       <  b.cast_int(0)       ); }
        else if ( ( acast == 2  ) && ( bcast == 2  ) ) { res = (  a.cast_double(0)    <  b.cast_double(0)    ); }
        else if ( ( acast == 3  ) && ( bcast == 3  ) ) { res = 0;                                             }
        else if ( ( acast == 4  ) && ( bcast == 4  ) ) { res = (  a.cast_vector(0)    <  b.cast_vector(0)    ); }
        else if ( ( acast == 5  ) && ( bcast == 5  ) ) { res = (  a.cast_matrix(0)    <  b.cast_matrix(0)    ); }
        else if ( ( acast == 8  ) && ( bcast == 8  ) ) { res = (  a.cast_set(0)       <  b.cast_set(0)       ); }
        else if ( ( acast == 9  ) && ( bcast == 9  ) ) { res = 0;                                             }
        else if ( ( acast == 10 ) && ( bcast == 10 ) ) { res = 0;                                             }
        else                                           { res = 0;                                             }
    }

    return res;
}

int operator>(const gentype &a, const gentype &b)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        return 0;
    }

    int res = 0;
    int acast = 0;
    int bcast = 0;

    if ( ( res = checkAddCompat(a,b,acast,bcast,1) ) )
    {
             if ( ( acast == 1  ) && ( bcast == 1  ) ) { res = (  a.cast_int(0)       >  b.cast_int(0)       ); }
        else if ( ( acast == 2  ) && ( bcast == 2  ) ) { res = (  a.cast_double(0)    >  b.cast_double(0)    ); }
        else if ( ( acast == 3  ) && ( bcast == 3  ) ) { res = 0;                                             }
        else if ( ( acast == 4  ) && ( bcast == 4  ) ) { res = (  a.cast_vector(0)    >  b.cast_vector(0)    ); }
        else if ( ( acast == 5  ) && ( bcast == 5  ) ) { res = (  a.cast_matrix(0)    >  b.cast_matrix(0)    ); }
        else if ( ( acast == 8  ) && ( bcast == 8  ) ) { res = (  a.cast_set(0)       >  b.cast_set(0)       ); }
        else if ( ( acast == 9  ) && ( bcast == 9  ) ) { res = 0;                                             }
        else if ( ( acast == 10 ) && ( bcast == 10 ) ) { res = 0;                                             }
        else                                           { res = 0;                                             }
    }

    return res;
}

int operator<=(const gentype &a, const gentype &b)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        return 0;
    }

    int res = 0;
    int acast = 0;
    int bcast = 0;

    if ( ( res = checkAddCompat(a,b,acast,bcast,1) ) )
    {
             if ( ( acast == 1  ) && ( bcast == 1  ) ) { res = (  a.cast_int(0)       <= b.cast_int(0)       ); }
        else if ( ( acast == 2  ) && ( bcast == 2  ) ) { res = (  a.cast_double(0)    <= b.cast_double(0)    ); }
        else if ( ( acast == 3  ) && ( bcast == 3  ) ) { res = 1;                                             }
        else if ( ( acast == 4  ) && ( bcast == 4  ) ) { res = (  a.cast_vector(0)    <= b.cast_vector(0)    ); }
        else if ( ( acast == 5  ) && ( bcast == 5  ) ) { res = (  a.cast_matrix(0)    <= b.cast_matrix(0)    ); }
        else if ( ( acast == 8  ) && ( bcast == 8  ) ) { res = (  a.cast_set(0)       <= b.cast_set(0)       ); }
        else if ( ( acast == 9  ) && ( bcast == 9  ) ) { res = 1;                                             }
        else if ( ( acast == 10 ) && ( bcast == 10 ) ) { res = 1;                                             }
        else                                           { res = 1;                                             }
    }

    return res;
}

int operator>=(const gentype &a, const gentype &b)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        return 0;
    }

    int res = 0;
    int acast = 0;
    int bcast = 0;

    if ( ( res = checkAddCompat(a,b,acast,bcast,1) ) )
    {
             if ( ( acast == 1  ) && ( bcast == 1  ) ) { res = (  a.cast_int(0)       >= b.cast_int(0)       ); }
        else if ( ( acast == 2  ) && ( bcast == 2  ) ) { res = (  a.cast_double(0)    >= b.cast_double(0)    ); }
        else if ( ( acast == 3  ) && ( bcast == 3  ) ) { res = 1;                                             }
        else if ( ( acast == 4  ) && ( bcast == 4  ) ) { res = (  a.cast_vector(0)    >= b.cast_vector(0)    ); }
        else if ( ( acast == 5  ) && ( bcast == 5  ) ) { res = (  a.cast_matrix(0)    >= b.cast_matrix(0)    ); }
        else if ( ( acast == 8  ) && ( bcast == 8  ) ) { res = (  a.cast_set(0)       >= b.cast_set(0)       ); }
        else if ( ( acast == 9  ) && ( bcast == 9  ) ) { res = 1;                                             }
        else if ( ( acast == 10 ) && ( bcast == 10 ) ) { res = 1;                                             }
        else                                           { res = 1;                                             }
    }

    return res;
}

// + posation - unary, return rvalue
// - negation - unary, return rvalue

gentype  operator+(const gentype &a)
{
    if ( a.isValEqnDir() )
    {
        const static gentype res("+x");

        return res(a);
    }

    gentype res = a;

    return setposate(res);
}

gentype  operator-(const gentype &a)
{
    if ( a.isValEqnDir() )
    {
        const static gentype res("-x");

        return res(a);
    }

    gentype res = a;

    return setnegate(res);
}

// + addition       - binary, return rvalue
// - subtraction    - binary, return rvalue
// * multiplication - binary, return rvalue

/*
gentype  operator/(const gentype &a, const gentype &b)
{
    // NB: the old version (commented out) does integer division for
    // integers.  This is more trouble than it is worth, as it makes
    // equations like sin(var(1,1)/2) act in unexpected ways.  If you
    // need integer division, use idiv, eg sin(idiv(var(1,1),2)).
    // Other major problem: the code tends to seek the simplest representation
    // for any given number, so 1.0 is demoted to 1, and 1.0/2.0 will
    // be demoted to 1/2, which is 0 and not 0.5 when integer division
    // is implemented
    //
    //    gentype res;
    //
    //    if ( a.isValInteger() && b.isValInteger() )
    //    {
    //	Special case of integer division
    //
    //	res = a.cast_int(0) / b.cast_int(0);
    //    }
    //
    //    else if ( !(a.isValString()) && !(b.isValString()) )
    //    {
    //        res = a*inv(b);
    //    }
    //
    //    else
    //    {
    //	constructError(a,b,res,"Error: incompatible types in division.");
    //    }

    gentype res(a);

    return res /= b;
}
*/

gentype  operator%(const gentype &a, const gentype &b)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype res("x%y");

        return res(a,b);
    }

    gentype res;

    if ( a.isCastableToIntegerWithoutLoss() && b.isCastableToIntegerWithoutLoss() )
    {
        res = ( a.cast_int(0) ) % ( b.cast_int(0) );
    }

    else
    {
	constructError(a,b,res,"Error: incompatible (non-integer) types in mod.");
    }

    return res;
}

















gentype &gentype::leftmult(const gentype &b)
{
    if ( isCastableToIntegerWithoutLoss() )
    {
             if ( cast_int(0) ==  0 ) { return *this; }
        else if ( cast_int(0) ==  1 ) { fastcopy(b,1); return *this; }
        else if ( cast_int(0) == -1 ) { fastcopy(b,1); negate(); return *this; }
    }

    if ( b.isCastableToIntegerWithoutLoss() )
    {   
        //NB:  need to keep structure of result, even if it is zero
        //     if ( b.cast_int(0) ==  0 ) { *this = zeroint(); return *this; }
        if ( b.cast_int(0) ==  1 ) { return *this; }
        else if ( b.cast_int(0) == -1 ) { negate(); return *this; }
    }

    if ( isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype tempres("x*y");
        gentype res(tempres);

        // Method:
        //
        // - switch contents of a and res quickly (so a is now x*y)
        // - switch contents of lefthand operand in a (multiplication)
        //   with res (which is what a was originally).
        // - operation is now half done.
        // - do fastcopy to overwrite righthand operand in a with b

        if ( varid_isscalar )
        {
            res.varid_isscalar = varid_isscalar;
            res.varid_numpts   = varid_numpts;

            res.varid_xi = varid_xi;
            res.varid_xj = varid_xj;
        }

        if ( b.varid_isscalar )
        {
            res.varid_isscalar = b.varid_isscalar;
            res.varid_numpts   = b.varid_numpts;

            res.varid_xi = b.varid_xi;
            res.varid_xj = b.varid_xj;
        }

        qswap(*this,res);
        qswap((*(eqnargs))("&",0),res);
        (*(eqnargs))("&",1).fastcopy(b,1);

        if ( (*(eqnargs))(zeroint()).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(zeroint()).varid_isscalar;
            varid_numpts   = (*(eqnargs))(zeroint()).varid_numpts;

            varid_xi = (*(eqnargs))(zeroint()).varid_xi;
            varid_xj = (*(eqnargs))(zeroint()).varid_xj;

            (*(eqnargs))("&",zeroint()).varid_isscalar = 0;
        }

        if ( (*(eqnargs))(1).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(1).varid_isscalar;
            varid_numpts   = (*(eqnargs))(1).varid_numpts;

            varid_xi = (*(eqnargs))(1).varid_xi;
            varid_xj = (*(eqnargs))(1).varid_xj;

            (*(eqnargs))("&",1).varid_isscalar = 0;
        }

        return *this;
    }

    // Design decision: vector*vector is the inner product.  That way,
    // the derivative of abs2(x) works correctly for vectorial x
    //
    // Design decision: string*string is 1 if same, -1 otherwise.  That
    // way, inner product can be used for categorical features in the SVM

    if (( isValVector() && b.isValVector() && ( size()    != b.size()    ) )||
        ( isValVector() && b.isValMatrix() && ( size()    != b.numRows() ) )||
        ( isValMatrix() && b.isValVector() && ( numCols() != b.size()    ) )||
        ( isValMatrix() && b.isValMatrix() && ( numCols() == b.numRows() ) ))
    {
        goto errcase;
    }

         if ( isValInteger() && b.isValInteger() ) { ::leftmult(dir_int(),   b.cast_int(0)   ); }
    else if ( isValInteger() && b.isValReal()    ) { ::leftmult(dir_double(),b.cast_double(0)); }
    else if ( isValInteger() && b.isValAnion()   ) { ::leftmult(dir_anion(), b.cast_anion(0) ); }
    else if ( isValReal()    && b.isValInteger() ) { ::leftmult(dir_double(),b.cast_int(0)   ); }
    else if ( isValReal()    && b.isValReal()    ) { ::leftmult(dir_double(),b.cast_double(0)); }
    else if ( isValReal()    && b.isValAnion()   ) { ::leftmult(dir_anion(), b.cast_anion(0) ); }
    else if ( isValAnion()   && b.isValInteger() ) { ::leftmult(dir_anion(), b.cast_double(0)); }
    else if ( isValAnion()   && b.isValReal()    ) { ::leftmult(dir_anion(), b.cast_double(0)); }
    else if ( isValAnion()   && b.isValAnion()   ) { ::leftmult(dir_anion(), b.cast_anion(0) ); }
    else if ( isValVector()  && b.isValInteger() ) { ::leftmult(dir_vector(),b              ); }
    else if ( isValVector()  && b.isValReal()    ) { ::leftmult(dir_vector(),b              ); }
    else if ( isValVector()  && b.isValAnion()   ) { ::leftmult(dir_vector(),b              ); }
    else if ( isValVector()  && b.isValMatrix()  ) { ::leftmult(dir_vector(),b.cast_matrix(0)); } 
    else if ( isValVector()  && b.isValString()  ) { ::leftmult(dir_vector(),b              ); }
    else if ( isValMatrix()  && b.isValInteger() ) { ::leftmult(dir_matrix(),b              ); }
    else if ( isValMatrix()  && b.isValReal()    ) { ::leftmult(dir_matrix(),b              ); }
    else if ( isValMatrix()  && b.isValAnion()   ) { ::leftmult(dir_matrix(),b              ); }
    else if ( isValMatrix()  && b.isValMatrix()  ) { ::leftmult(dir_matrix(),b.cast_matrix(0)); } 
    else if ( isValMatrix()  && b.isValString()  ) { ::leftmult(dir_matrix(),b              ); }
    else if ( isValSet()     && b.isValInteger() ) { ::leftmult(dir_set()   ,b              ); }
    else if ( isValSet()     && b.isValReal()    ) { ::leftmult(dir_set()   ,b              ); }
    else if ( isValSet()     && b.isValAnion()   ) { ::leftmult(dir_set()   ,b              ); }
    else if ( isValSet()     && b.isValString()  ) { ::leftmult(dir_set()   ,b              ); }

    else if ( isValNull()    && b.isValNull()    ) { ; }
    else if ( isValInteger() && b.isValNull()    ) { *this = 0; }
    else if ( isValReal()    && b.isValNull()    ) { *this = 0; }
    else if ( isValAnion()   && b.isValNull()    ) { *this = 0; }
    else if ( isValVector()  && b.isValNull()    ) { *this = 0; }
    else if ( isValMatrix()  && b.isValNull()    ) { *this = 0; }
    else if ( isValSet()     && b.isValNull()    ) { *this = 0; }
    else if ( isValString()  && b.isValNull()    ) { *this = 0; }
    else if ( isValDgraph()  && b.isValNull()    ) { *this = 0; }
    else if ( isValNull()    && b.isValInteger() ) { *this = 0; }
    else if ( isValNull()    && b.isValReal()    ) { *this = 0; }
    else if ( isValNull()    && b.isValAnion()   ) { *this = 0; }
    else if ( isValNull()    && b.isValVector()  ) { *this = 0; }
    else if ( isValNull()    && b.isValMatrix()  ) { *this = 0; }
    else if ( isValNull()    && b.isValSet()     ) { *this = 0; }
    else if ( isValNull()    && b.isValString()  ) { *this = 0; }
    else if ( isValNull()    && b.isValDgraph()  ) { *this = 0; }

    else if ( isValInteger() && b.isValVector()  ) { int     x = cast_int(0);       *this = b; *this *= x; }
    else if ( isValInteger() && b.isValMatrix()  ) { int     x = cast_int(0);       *this = b; *this *= x; } 
    else if ( isValInteger() && b.isValSet()     ) { int     x = cast_int(0);       *this = b; *this *= x; } 
    else if ( isValReal()    && b.isValVector()  ) { double  x = cast_double(0);    *this = b; *this *= x; } 
    else if ( isValReal()    && b.isValMatrix()  ) { double  x = cast_double(0);    *this = b; *this *= x; } 
    else if ( isValReal()    && b.isValSet()     ) { double  x = cast_double(0);    *this = b; *this *= x; } 
    else if ( isValString()  && b.isValVector()  ) { std::string x(cast_string(0)); *this = b; *this *= x; } 
    else if ( isValString()  && b.isValMatrix()  ) { std::string x(cast_string(0)); *this = b; *this *= x; } 
    else if ( isValString()  && b.isValSet()     ) { std::string x(cast_string(0)); *this = b; *this *= x; } 

    else if ( isValAnion()   && b.isValVector()  ) { gentype x(*this); *this = b; rightmult(x); } 
    else if ( isValAnion()   && b.isValMatrix()  ) { gentype x(*this); *this = b; rightmult(x); } 
    else if ( isValAnion()   && b.isValSet()     ) { gentype x(*this); *this = b; rightmult(x); } 

    else if ( isValVector()  && b.isValVector()  ) { gentype temp; twoProductNoConj(temp,cast_vector(0),b.cast_vector(0)); *this = temp; } 
    else if ( isValSet()     && b.isValSet()     ) { double  temp; twoProduct      (temp,cast_set(0),   b.cast_set(0)   ); *this = temp; }
    else if ( isValDgraph()  && b.isValDgraph()  ) { double  temp; twoProduct      (temp,cast_dgraph(0),b.cast_dgraph(0)); *this = temp; }

    else if ( isValMatrix()  && b.isValVector()  ) { *this = ( cast_matrix(0) * b.cast_vector(0) ); }
    else if ( isValString()  && b.isValString()  ) { *this = ( ( *this == b ) ? 1 : -1 ); }

    else if ( isValInteger() && b.isValString()  ) { goto errcase; }
    else if ( isValInteger() && b.isValDgraph()  ) { goto errcase; }
    else if ( isValReal()    && b.isValString()  ) { goto errcase; }
    else if ( isValReal()    && b.isValDgraph()  ) { goto errcase; }
    else if ( isValAnion()   && b.isValString()  ) { goto errcase; }
    else if ( isValAnion()   && b.isValDgraph()  ) { goto errcase; }
    else if ( isValVector()  && b.isValSet()     ) { goto errcase; }
    else if ( isValVector()  && b.isValDgraph()  ) { goto errcase; }
    else if ( isValMatrix()  && b.isValSet()     ) { goto errcase; }
    else if ( isValMatrix()  && b.isValDgraph()  ) { goto errcase; }
    else if ( isValString()  && b.isValInteger() ) { goto errcase; }
    else if ( isValString()  && b.isValReal()    ) { goto errcase; }
    else if ( isValString()  && b.isValAnion()   ) { goto errcase; }
    else if ( isValString()  && b.isValDgraph()  ) { goto errcase; }
    else if ( isValSet()     && b.isValVector()  ) { goto errcase; }
    else if ( isValSet()     && b.isValMatrix()  ) { goto errcase; }
    else if ( isValSet()     && b.isValDgraph()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValInteger() ) { goto errcase; }
    else if ( isValDgraph()  && b.isValReal()    ) { goto errcase; }
    else if ( isValDgraph()  && b.isValAnion()   ) { goto errcase; }
    else if ( isValDgraph()  && b.isValVector()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValMatrix()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValString()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValSet()     ) { goto errcase; }

    else
    {
    errcase:
        std::string errstr("Error: incompatible types in multiplication leftmult(");
        errstr += cast_string(0);
        errstr += ",";
        errstr += b.cast_string(0);
        errstr += ")";

        constructError(*this,b,*this,errstr.c_str());
    }

    return *this;
}

gentype &gentype::rightmult(const gentype &a)
{
    if ( isCastableToIntegerWithoutLoss() )
    {
             if ( cast_int(0) ==  0 ) { return *this; }
        else if ( cast_int(0) ==  1 ) { fastcopy(a,1); return *this; }
        else if ( cast_int(0) == -1 ) { fastcopy(a,1); negate(); return *this; }
    }

    if ( a.isCastableToIntegerWithoutLoss() )
    {
        //NB:  need to keep structure of result, even if it is zero
        //     if ( a.cast_int(0) ==  0 ) { *this = zeroint(); return *this; }
        if ( a.cast_int(0) ==  1 ) { return *this; }
        else if ( a.cast_int(0) == -1 ) { negate(); return *this; }
    }

    if ( a.isValEqnDir() || isValEqnDir() )
    {
        const static gentype tempres("x*y");
        gentype res(tempres);

        if ( a.varid_isscalar )
        {
            res.varid_isscalar = a.varid_isscalar;
            res.varid_numpts   = a.varid_numpts;

            res.varid_xi = a.varid_xi;
            res.varid_xj = a.varid_xj;
        }

        if ( varid_isscalar )
        {
            res.varid_isscalar = varid_isscalar;
            res.varid_numpts   = varid_numpts;

            res.varid_xi = varid_xi;
            res.varid_xj = varid_xj;
        }

        qswap(*this,res);
        qswap((*(eqnargs))("&",1),res);
        (*(eqnargs))("&",0).fastcopy(a,1);

        if ( (*(eqnargs))(zeroint()).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(zeroint()).varid_isscalar;
            varid_numpts   = (*(eqnargs))(zeroint()).varid_numpts;

            varid_xi = (*(eqnargs))(zeroint()).varid_xi;
            varid_xj = (*(eqnargs))(zeroint()).varid_xj;

            (*(eqnargs))("&",zeroint()).varid_isscalar = 0;
        }

        if ( (*(eqnargs))(1).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(1).varid_isscalar;
            varid_numpts   = (*(eqnargs))(1).varid_numpts;

            varid_xi = (*(eqnargs))(1).varid_xi;
            varid_xj = (*(eqnargs))(1).varid_xj;

            (*(eqnargs))("&",1).varid_isscalar = 0;
        }

        return *this;
    }

    // Design decision: vector*vector is the inner product.  That way,
    // the derivative of abs2(x) works correctly for vectorial x
    //
    // Design decision: string*string is 1 if same, -1 otherwise.  That
    // way, inner product can be used for categorical features in the SVM

    if (( a.isValVector() && isValVector() && ( a.size()    != size()    ) )||
        ( a.isValVector() && isValMatrix() && ( a.size()    != numRows() ) )||
        ( a.isValMatrix() && isValVector() && ( a.numCols() != size()    ) )||
        ( a.isValMatrix() && isValMatrix() && ( a.numCols() == numRows() ) ))
    {
        goto errcase;
    }

         if ( a.isValInteger() && isValInteger() ) { ::rightmult(a.cast_int(0)   ,dir_int()   ); }
    else if ( a.isValInteger() && isValReal()    ) { ::rightmult(a.cast_double(0),dir_double()); }
    else if ( a.isValInteger() && isValAnion()   ) { ::rightmult(a.cast_double(0),dir_anion() ); }
    else if ( a.isValInteger() && isValVector()  ) { ::rightmult(a              ,dir_vector()); } 
    else if ( a.isValInteger() && isValMatrix()  ) { ::rightmult(a              ,dir_matrix()); } 
    else if ( a.isValInteger() && isValSet()     ) { ::rightmult(a              ,dir_set()   ); } 
    else if ( a.isValReal()    && isValInteger() ) { ::rightmult(a.cast_double(0),dir_double()); }
    else if ( a.isValReal()    && isValReal()    ) { ::rightmult(a.cast_double(0),dir_double()); }
    else if ( a.isValReal()    && isValAnion()   ) { ::rightmult(a.cast_double(0),dir_anion() ); }
    else if ( a.isValReal()    && isValVector()  ) { ::rightmult(a              ,dir_vector()); } 
    else if ( a.isValReal()    && isValMatrix()  ) { ::rightmult(a              ,dir_matrix()); } 
    else if ( a.isValReal()    && isValSet()     ) { ::rightmult(a              ,dir_set()   ); } 
    else if ( a.isValAnion()   && isValInteger() ) { ::rightmult(a.cast_anion(0) ,dir_anion() ); }
    else if ( a.isValAnion()   && isValReal()    ) { ::rightmult(a.cast_anion(0) ,dir_anion() ); }
    else if ( a.isValAnion()   && isValAnion()   ) { ::rightmult(a.cast_anion(0) ,dir_anion() ); }
    else if ( a.isValAnion()   && isValVector()  ) { ::rightmult(a              ,dir_vector()); } 
    else if ( a.isValAnion()   && isValMatrix()  ) { ::rightmult(a              ,dir_matrix()); } 
    else if ( a.isValAnion()   && isValSet()     ) { ::rightmult(a              ,dir_set()   ); } 
    else if ( a.isValMatrix()  && isValVector()  ) { ::rightmult(a.cast_matrix(0),dir_vector()); }
    else if ( a.isValMatrix()  && isValMatrix()  ) { ::rightmult(a.cast_matrix(0),dir_matrix()); } 
    else if ( a.isValString()  && isValVector()  ) { ::rightmult(a              ,dir_vector()); } 
    else if ( a.isValString()  && isValMatrix()  ) { ::rightmult(a              ,dir_matrix()); } 
    else if ( a.isValString()  && isValSet()     ) { ::rightmult(a              ,dir_set()   ); } 

    else if ( a.isValNull()    && isValNull()    ) { ; }
    else if ( a.isValInteger() && isValNull()    ) { *this = 0; }
    else if ( a.isValReal()    && isValNull()    ) { *this = 0; }
    else if ( a.isValAnion()   && isValNull()    ) { *this = 0; }
    else if ( a.isValVector()  && isValNull()    ) { *this = 0; }
    else if ( a.isValMatrix()  && isValNull()    ) { *this = 0; }
    else if ( a.isValSet()     && isValNull()    ) { *this = 0; }
    else if ( a.isValString()  && isValNull()    ) { *this = 0; }
    else if ( a.isValDgraph()  && isValNull()    ) { *this = 0; }
    else if ( a.isValNull()    && isValInteger() ) { *this = 0; }
    else if ( a.isValNull()    && isValReal()    ) { *this = 0; }
    else if ( a.isValNull()    && isValAnion()   ) { *this = 0; }
    else if ( a.isValNull()    && isValVector()  ) { *this = 0; }
    else if ( a.isValNull()    && isValMatrix()  ) { *this = 0; }
    else if ( a.isValNull()    && isValSet()     ) { *this = 0; }
    else if ( a.isValNull()    && isValString()  ) { *this = 0; }
    else if ( a.isValNull()    && isValDgraph()  ) { *this = 0; }

    else if ( a.isValVector()  && isValInteger() ) { int     x = cast_int(0);       *this = a; *this *= x; }
    else if ( a.isValMatrix()  && isValInteger() ) { int     x = cast_int(0);       *this = a; *this *= x; }
    else if ( a.isValSet()     && isValInteger() ) { int     x = cast_int(0);       *this = a; *this *= x; }
    else if ( a.isValVector()  && isValReal()    ) { double  x = cast_double(0);    *this = a; *this *= x; }
    else if ( a.isValMatrix()  && isValReal()    ) { double  x = cast_double(0);    *this = a; *this *= x; }
    else if ( a.isValSet()     && isValReal()    ) { double  x = cast_double(0);    *this = a; *this *= x; }
    else if ( a.isValVector()  && isValString()  ) { std::string x(cast_string(0)); *this = a; *this *= x; }
    else if ( a.isValMatrix()  && isValString()  ) { std::string x(cast_string(0)); *this = a; *this *= x; }
    else if ( a.isValSet()     && isValString()  ) { std::string x(cast_string(0)); *this = a; *this *= x; }

    else if ( a.isValVector()  && isValAnion()   ) { gentype x(*this); *this = a; leftmult(x); }
    else if ( a.isValMatrix()  && isValAnion()   ) { gentype x(*this); *this = a; leftmult(x); }
    else if ( a.isValSet()     && isValAnion()   ) { gentype x(*this); *this = a; leftmult(x); }

    else if ( a.isValVector()  && isValVector()  ) { gentype temp; twoProductNoConj(temp,a.cast_vector(0),cast_vector(0)); *this = temp; } 
    else if ( a.isValSet()     && isValSet()     ) { double  temp; twoProduct(temp,a.cast_set(0)         ,cast_set(0)   ); *this = temp; }
    else if ( a.isValDgraph()  && isValDgraph()  ) { double  temp; twoProduct(temp,a.cast_dgraph(0)      ,cast_dgraph(0)); *this = temp; }

    else if ( a.isValVector()  && isValMatrix()  ) { *this = ( a.cast_vector(0) * cast_matrix(0) ); }
    else if ( a.isValString()  && isValString()  ) { *this = ( ( a == *this ) ? 1 : -1 ); }

    else if ( a.isValInteger() && isValString()  ) { goto errcase; }
    else if ( a.isValInteger() && isValDgraph()  ) { goto errcase; }
    else if ( a.isValReal()    && isValString()  ) { goto errcase; }
    else if ( a.isValReal()    && isValDgraph()  ) { goto errcase; }
    else if ( a.isValAnion()   && isValString()  ) { goto errcase; }
    else if ( a.isValAnion()   && isValDgraph()  ) { goto errcase; }
    else if ( a.isValVector()  && isValSet()     ) { goto errcase; }
    else if ( a.isValVector()  && isValDgraph()  ) { goto errcase; }
    else if ( a.isValMatrix()  && isValSet()     ) { goto errcase; }
    else if ( a.isValMatrix()  && isValDgraph()  ) { goto errcase; }
    else if ( a.isValString()  && isValNull()    ) { goto errcase; }
    else if ( a.isValString()  && isValInteger() ) { goto errcase; }
    else if ( a.isValString()  && isValReal()    ) { goto errcase; }
    else if ( a.isValString()  && isValAnion()   ) { goto errcase; }
    else if ( a.isValString()  && isValDgraph()  ) { goto errcase; }
    else if ( a.isValSet()     && isValVector()  ) { goto errcase; }
    else if ( a.isValSet()     && isValMatrix()  ) { goto errcase; }
    else if ( a.isValSet()     && isValDgraph()  ) { goto errcase; }
    else if ( a.isValDgraph()  && isValNull()    ) { goto errcase; }
    else if ( a.isValDgraph()  && isValInteger() ) { goto errcase; }
    else if ( a.isValDgraph()  && isValReal()    ) { goto errcase; }
    else if ( a.isValDgraph()  && isValAnion()   ) { goto errcase; }
    else if ( a.isValDgraph()  && isValVector()  ) { goto errcase; }
    else if ( a.isValDgraph()  && isValMatrix()  ) { goto errcase; }
    else if ( a.isValDgraph()  && isValString()  ) { goto errcase; }
    else if ( a.isValDgraph()  && isValSet()     ) { goto errcase; }

    else
    {
    errcase:
        std::string errstr("Error: incompatible types in multiplication rightmult(");
        errstr += a.cast_string(0);
        errstr += ",";
        errstr += cast_string(0);
        errstr += ")";

        constructError(a,*this,*this,errstr.c_str());
    }

    return *this;
}

gentype &gentype::leftmult(const double &b)
{
    int bint = (int) b;
    int bisint = ( b == bint ) ? 1 : 0;
    int isint = isCastableToIntegerWithoutLoss();

         if ( isint && ( cast_int(0) ==  0 ) ) { ; }
    else if ( isint && ( cast_int(0) ==  1 ) ) { force_double() = b; }
    else if ( isint && ( cast_int(0) == -1 ) ) { force_double() = b; negate(); }

    else if ( bisint && ( bint ==  1 ) ) { ; }
    else if ( bisint && ( bint == -1 ) ) { negate(); }

    else if ( isValInteger() && bisint ) {                ::leftmult(dir_int(),   bint); }
    else if ( isValInteger()           ) {                ::leftmult(dir_double(),b   ); }
    else if ( isValReal()    && bisint ) {                ::leftmult(dir_double(),bint); }
    else if ( isValReal()              ) {                ::leftmult(dir_double(),b   ); }
    else if ( isValAnion()   && bisint ) {                ::leftmult(dir_anion(), b   ); }
    else if ( isValAnion()             ) {                ::leftmult(dir_anion(), b   ); }
    else if ( isValVector()  && bisint ) { gentype bg(b); ::leftmult(dir_vector(),bg  ); }
    else if ( isValVector()            ) { gentype bg(b); ::leftmult(dir_vector(),bg  ); }
    else if ( isValMatrix()  && bisint ) { gentype bg(b); ::leftmult(dir_matrix(),bg  ); }
    else if ( isValMatrix()            ) { gentype bg(b); ::leftmult(dir_matrix(),bg  ); }
    else if ( isValSet()     && bisint ) { gentype bg(b); ::leftmult(dir_set()   ,bg  ); }
    else if ( isValSet()               ) { gentype bg(b); ::leftmult(dir_set()   ,bg  ); }

    else if ( isValNull()  && bisint ) { *this = 0; }
    else if ( isValNull()            ) { *this = 0; }

    else if ( isValString()  && bisint ) { goto errcase; }
    else if ( isValString()            ) { goto errcase; }
    else if ( isValDgraph()  && bisint ) { goto errcase; }
    else if ( isValDgraph()            ) { goto errcase; }

    else
    {
    errcase:
        gentype bg(b);

        std::string errstr("Error: incompatible types in multiplication leftmult(");
        errstr += cast_string(0);
        errstr += ",";
        errstr += bg.cast_string(0);
        errstr += ")";

        constructError(*this,bg,*this,errstr.c_str());
    }

    return *this;
}

gentype &gentype::rightmult(const double &a)
{
    int aint = (int) a;
    int aisint = ( a == aint ) ? 1 : 0;
    int isint = isCastableToIntegerWithoutLoss();

         if ( isint && ( cast_int(0) ==  0 ) ) { ; }
    else if ( isint && ( cast_int(0) ==  1 ) ) { force_double() = a; }
    else if ( isint && ( cast_int(0) == -1 ) ) { force_double() = a; negate();  }

    else if ( aisint && ( aint ==  1 ) ) { ; }
    else if ( aisint && ( aint == -1 ) ) { negate(); }

    else if ( aisint && isValInteger() ) {                ::rightmult(aint,dir_int()   ); }
    else if ( aisint && isValReal()    ) {                ::rightmult(a   ,dir_double()); }
    else if ( aisint && isValAnion()   ) {                ::rightmult(a   ,dir_anion() ); }
    else if ( aisint && isValVector()  ) { gentype ag(a); ::rightmult(ag  ,dir_vector()); } 
    else if ( aisint && isValMatrix()  ) { gentype ag(a); ::rightmult(ag  ,dir_matrix()); } 
    else if ( aisint && isValSet()     ) { gentype ag(a); ::rightmult(ag  ,dir_set()   ); } 
    else if (           isValInteger() ) {                ::rightmult(a   ,dir_double()); }
    else if (           isValReal()    ) {                ::rightmult(a   ,dir_double()); }
    else if (           isValAnion()   ) {                ::rightmult(a   ,dir_anion() ); }
    else if (           isValVector()  ) { gentype ag(a); ::rightmult(ag  ,dir_vector()); } 
    else if (           isValMatrix()  ) { gentype ag(a); ::rightmult(ag  ,dir_matrix()); } 
    else if (           isValSet()     ) { gentype ag(a); ::rightmult(ag  ,dir_set()   ); } 

    else if ( aisint && isValNull() ) { *this = 0; }
    else if ( aisint && isValNull() ) { *this = 0; }

    else if ( aisint && isValString() ) { goto errcase; }
    else if ( aisint && isValDgraph() ) { goto errcase; }
    else if (           isValString() ) { goto errcase; }
    else if (           isValDgraph() ) { goto errcase; }

    else
    {
    errcase:
        gentype ag(a);

        std::string errstr("Error: incompatible types in multiplication rightmult(");
        errstr += ag.cast_string(0);
        errstr += ",";
        errstr += cast_string(0);
        errstr += ")";

        constructError(ag,*this,*this,errstr.c_str());
    }

    return *this;
}

gentype &gentype::leftdiv(const gentype &b)
{
    // Based on a leftmult with appropriate modifications.

    if ( isCastableToIntegerWithoutLoss() )
    {
             if ( cast_int(0) ==  0 ) { return *this; }
        else if ( cast_int(0) ==  1 ) { fastcopy(b); inverse(); return *this; }
        else if ( cast_int(0) == -1 ) { fastcopy(b); inverse(); negate(); return *this; }
    }

    if ( b.isCastableToIntegerWithoutLoss() )
    {
             if ( b.cast_int(0) ==  1 ) { return *this; }
        else if ( b.cast_int(0) == -1 ) { negate(); return *this; }
    }

    if ( isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype tempres("x/y");
        gentype res(tempres);

        // Method: see leftmult

        if ( varid_isscalar )
        {
            res.varid_isscalar = varid_isscalar;
            res.varid_numpts   = varid_numpts;

            res.varid_xi = varid_xi;
            res.varid_xj = varid_xj;
        }

        if ( b.varid_isscalar )
        {
            res.varid_isscalar = b.varid_isscalar;
            res.varid_numpts   = b.varid_numpts;

            res.varid_xi = b.varid_xi;
            res.varid_xj = b.varid_xj;
        }

        qswap(*this,res);
        qswap((*(eqnargs))("&",0),res);
        (*(eqnargs))("&",1).fastcopy(b,1);

        if ( (*(eqnargs))(zeroint()).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(zeroint()).varid_isscalar;
            varid_numpts   = (*(eqnargs))(zeroint()).varid_numpts;

            varid_xi = (*(eqnargs))(zeroint()).varid_xi;
            varid_xj = (*(eqnargs))(zeroint()).varid_xj;

            (*(eqnargs))("&",zeroint()).varid_isscalar = 0;
        }

        if ( (*(eqnargs))(1).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(1).varid_isscalar;
            varid_numpts   = (*(eqnargs))(1).varid_numpts;

            varid_xi = (*(eqnargs))(1).varid_xi;
            varid_xj = (*(eqnargs))(1).varid_xj;

            (*(eqnargs))("&",1).varid_isscalar = 0;
        }

        return *this;
    }

    // Note that pseudo-inverse of matrix of size i*j is of size j*i

    if (( isValVector() && b.isValMatrix() && ( size()    != b.numCols() ) )||
        ( isValMatrix() && b.isValMatrix() && ( numCols() == b.numCols() ) ))
    {
        goto errcase;
    }

         if ( isValNull()    && b.isValNull()    ) { ; }
    else if ( isValNull()    && b.isValInteger() ) { dir_double() /= b.cast_int(0);        }
    else if ( isValNull()    && b.isValReal()    ) { dir_double() /= b.cast_double(0);     }
    else if ( isValNull()    && b.isValAnion()   ) { dir_anion()  *= inv(b.cast_anion(0)); }
    else if ( isValNull()    && b.isValVector()  ) { goto errcase; }
    else if ( isValNull()    && b.isValMatrix()  ) { double temp = cast_int(0); *this = b; inverse(); *this *= temp; } 
    else if ( isValNull()    && b.isValString()  ) { goto errcase; }
    else if ( isValNull()    && b.isValSet()     ) { goto errcase; }
    else if ( isValNull()    && b.isValDgraph()  ) { goto errcase; }

    else if ( isValInteger() && b.isValNull()    ) { dir_double() /= b.cast_int(0);        }
    else if ( isValInteger() && b.isValInteger() ) { dir_double() /= b.cast_int(0);        }
    else if ( isValInteger() && b.isValReal()    ) { dir_double() /= b.cast_double(0);     }
    else if ( isValInteger() && b.isValAnion()   ) { dir_anion()  *= inv(b.cast_anion(0)); }
    else if ( isValInteger() && b.isValVector()  ) { goto errcase; } 
    else if ( isValInteger() && b.isValMatrix()  ) { double temp = cast_int(0); *this = b; inverse(); *this *= temp; } 
    else if ( isValInteger() && b.isValString()  ) { goto errcase; }
    else if ( isValInteger() && b.isValSet()     ) { goto errcase; }
    else if ( isValInteger() && b.isValDgraph()  ) { goto errcase; }

    else if ( isValReal()    && b.isValNull()    ) { dir_double() /= b.cast_int(0);        }
    else if ( isValReal()    && b.isValInteger() ) { dir_double() /= b.cast_int(0);        }
    else if ( isValReal()    && b.isValReal()    ) { dir_double() /= b.cast_double(0);     }
    else if ( isValReal()    && b.isValAnion()   ) { dir_anion()  *= inv(b.cast_anion(0)); }
    else if ( isValReal()    && b.isValVector()  ) { goto errcase; } 
    else if ( isValReal()    && b.isValMatrix()  ) { double temp = cast_double(0);  *this = b; inverse(); *this *= temp;         } 
    else if ( isValReal()    && b.isValString()  ) { goto errcase; }
    else if ( isValReal()    && b.isValSet()     ) { goto errcase; }
    else if ( isValReal()    && b.isValDgraph()  ) { goto errcase; }

    else if ( isValAnion()   && b.isValNull()    ) { dir_anion()  /= b.cast_double(0);      }
    else if ( isValAnion()   && b.isValInteger() ) { dir_anion()  /= b.cast_double(0);      }
    else if ( isValAnion()   && b.isValReal()    ) { dir_anion()  /= b.cast_double(0);      }
    else if ( isValAnion()   && b.isValAnion()   ) { dir_anion()  *= inv(b.cast_anion(0));  }
    else if ( isValAnion()   && b.isValVector()  ) { goto errcase; }
    else if ( isValAnion()   && b.isValMatrix()  ) { gentype temp(cast_anion(0)); *this = b; inverse(); rightmult(temp); } 
    else if ( isValAnion()   && b.isValString()  ) { goto errcase; }
    else if ( isValAnion()   && b.isValSet()     ) { goto errcase; }
    else if ( isValAnion()   && b.isValDgraph()  ) { goto errcase; }

    else if ( isValVector()  && b.isValNull()    ) { dir_vector() *= inv(b); }
    else if ( isValVector()  && b.isValInteger() ) { dir_vector() *= inv(b); }
    else if ( isValVector()  && b.isValReal()    ) { dir_vector() *= inv(b); }
    else if ( isValVector()  && b.isValAnion()   ) { dir_vector() *= inv(b); }
    else if ( isValVector()  && b.isValVector()  ) { goto errcase; } 
    else if ( isValVector()  && b.isValMatrix()  ) { dir_vector() *= inv(b.cast_matrix(0)); } 
    else if ( isValVector()  && b.isValString()  ) { goto errcase; }
    else if ( isValVector()  && b.isValSet()     ) { goto errcase; }
    else if ( isValVector()  && b.isValDgraph()  ) { goto errcase; }

    else if ( isValMatrix()  && b.isValNull()    ) { dir_matrix() *= inv(b); }
    else if ( isValMatrix()  && b.isValInteger() ) { dir_matrix() *= inv(b); }
    else if ( isValMatrix()  && b.isValReal()    ) { dir_matrix() *= inv(b); }
    else if ( isValMatrix()  && b.isValAnion()   ) { dir_matrix() *= inv(b); }
    else if ( isValMatrix()  && b.isValVector()  ) { goto errcase; }
    else if ( isValMatrix()  && b.isValMatrix()  ) { dir_matrix() *= inv(b.cast_matrix(0)); }  
    else if ( isValMatrix()  && b.isValString()  ) { goto errcase; }
    else if ( isValMatrix()  && b.isValSet()     ) { goto errcase; }
    else if ( isValMatrix()  && b.isValDgraph()  ) { goto errcase; }

    else if ( isValString()  && b.isValNull()    ) { goto errcase; }
    else if ( isValString()  && b.isValInteger() ) { goto errcase; }
    else if ( isValString()  && b.isValReal()    ) { goto errcase; }
    else if ( isValString()  && b.isValAnion()   ) { goto errcase; }
    else if ( isValString()  && b.isValVector()  ) { goto errcase; }
    else if ( isValString()  && b.isValMatrix()  ) { goto errcase; } 
    else if ( isValString()  && b.isValString()  ) { goto errcase; } 
    else if ( isValString()  && b.isValSet()     ) { goto errcase; } 
    else if ( isValString()  && b.isValDgraph()  ) { goto errcase; }

    else if ( isValSet()     && b.isValNull()    ) { dir_set() *= inv(b); }
    else if ( isValSet()     && b.isValInteger() ) { dir_set() *= inv(b); }
    else if ( isValSet()     && b.isValReal()    ) { dir_set() *= inv(b); }
    else if ( isValSet()     && b.isValAnion()   ) { dir_set() *= inv(b); }
    else if ( isValSet()     && b.isValVector()  ) { goto errcase; }
    else if ( isValSet()     && b.isValMatrix()  ) { goto errcase; }
    else if ( isValSet()     && b.isValString()  ) { goto errcase; }
    else if ( isValSet()     && b.isValSet()     ) { goto errcase; }
    else if ( isValSet()     && b.isValDgraph()  ) { goto errcase; }

    else if ( isValDgraph()  && b.isValNull()    ) { goto errcase; }
    else if ( isValDgraph()  && b.isValInteger() ) { goto errcase; }
    else if ( isValDgraph()  && b.isValReal()    ) { goto errcase; }
    else if ( isValDgraph()  && b.isValAnion()   ) { goto errcase; }
    else if ( isValDgraph()  && b.isValVector()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValMatrix()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValString()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValSet()     ) { goto errcase; }
    else if ( isValDgraph()  && b.isValDgraph()  ) { goto errcase; }

    else
    {
    errcase:
        std::string errstr("Error: incompatible types in division leftdiv(");
        errstr += cast_string(0);
        errstr += ",";
        errstr += b.cast_string(0);
        errstr += ")";

        constructError(*this,b,*this,errstr.c_str());
    }

    return *this;
}

gentype &gentype::rightdiv(const gentype &b)
{
    // Based on a leftdiv with appropriate modifications.
    // (note reversal of b and a in the function definition)

    if ( isCastableToIntegerWithoutLoss() )
    {
             if ( cast_int(0) ==  0 ) { return *this; }
        else if ( cast_int(0) ==  1 ) { fastcopy(b); inverse(); return *this; }
        else if ( cast_int(0) == -1 ) { fastcopy(b); inverse(); negate(); return *this; }
    }

    if ( b.isCastableToIntegerWithoutLoss() )
    {
             if ( b.cast_int(0) ==  1 ) { return *this; }
        else if ( b.cast_int(0) == -1 ) { negate(); return *this; }
    }

    if ( isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype tempres("x\\y"); // Note right division
        gentype res(tempres);

        // Method: see leftmult

        if ( b.varid_isscalar )
        {
            res.varid_isscalar = b.varid_isscalar;
            res.varid_numpts   = b.varid_numpts;

            res.varid_xi = b.varid_xi;
            res.varid_xj = b.varid_xj;
        }

        if ( varid_isscalar )
        {
            res.varid_isscalar = varid_isscalar;
            res.varid_numpts   = varid_numpts;

            res.varid_xi = varid_xi;
            res.varid_xj = varid_xj;
        }

        qswap(*this,res);
        qswap((*(eqnargs))("&",1),res);
        (*(eqnargs))("&",0).fastcopy(b,1);

        if ( (*(eqnargs))(zeroint()).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(zeroint()).varid_isscalar;
            varid_numpts   = (*(eqnargs))(zeroint()).varid_numpts;

            varid_xi = (*(eqnargs))(zeroint()).varid_xi;
            varid_xj = (*(eqnargs))(zeroint()).varid_xj;

            (*(eqnargs))("&",zeroint()).varid_isscalar = 0;
        }

        if ( (*(eqnargs))(1).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(1).varid_isscalar;
            varid_numpts   = (*(eqnargs))(1).varid_numpts;

            varid_xi = (*(eqnargs))(1).varid_xi;
            varid_xj = (*(eqnargs))(1).varid_xj;

            (*(eqnargs))("&",1).varid_isscalar = 0;
        }

        return *this;
    }

    // Note that pseudo-inverse of matrix of size i*j is of size j*i

    if (( isValVector() && b.isValMatrix() && ( size()    != b.numRows() ) )||
        ( isValMatrix() && b.isValMatrix() && ( numRows() == b.numRows() ) ))
    {
        goto errcase;
    }

    // Note that the code only differs from leftdiv when dealing with
    // non-commutative fields

         if ( isValNull()    && b.isValNull()    ) { ; }
    else if ( isValNull()    && b.isValInteger() ) { dir_double() /= b.cast_int(0);    }
    else if ( isValNull()    && b.isValReal()    ) { dir_double() /= b.cast_double(0); }
    else if ( isValNull()    && b.isValAnion()   ) { ::rightmult(inv(b.cast_anion(0)),dir_anion());  }
    else if ( isValNull()    && b.isValVector()  ) { goto errcase; } 
    else if ( isValNull()    && b.isValMatrix()  ) { double temp = cast_int(0);     *this = b; inverse(); *this *= temp;         } 
    else if ( isValNull()    && b.isValString()  ) { goto errcase; }
    else if ( isValNull()    && b.isValSet()     ) { goto errcase; }
    else if ( isValNull()    && b.isValDgraph()  ) { goto errcase; }

    else if ( isValInteger() && b.isValNull()    ) { dir_double() /= b.cast_int(0);    }
    else if ( isValInteger() && b.isValInteger() ) { dir_double() /= b.cast_int(0);    }
    else if ( isValInteger() && b.isValReal()    ) { dir_double() /= b.cast_double(0); }
    else if ( isValInteger() && b.isValAnion()   ) { ::rightmult(inv(b.cast_anion(0)),dir_anion());  }
    else if ( isValInteger() && b.isValVector()  ) { goto errcase; } 
    else if ( isValInteger() && b.isValMatrix()  ) { double temp = cast_int(0);     *this = b; inverse(); *this *= temp;         } 
    else if ( isValInteger() && b.isValString()  ) { goto errcase; }
    else if ( isValInteger() && b.isValSet()     ) { goto errcase; }
    else if ( isValInteger() && b.isValDgraph()  ) { goto errcase; }

    else if ( isValReal()    && b.isValNull()    ) { dir_double() /= b.cast_int(0);    }
    else if ( isValReal()    && b.isValInteger() ) { dir_double() /= b.cast_int(0);    }
    else if ( isValReal()    && b.isValReal()    ) { dir_double() /= b.cast_double(0); }
    else if ( isValReal()    && b.isValAnion()   ) { ::rightmult(inv(b.cast_anion(0)),dir_anion());  }
    else if ( isValReal()    && b.isValVector()  ) { goto errcase; } 
    else if ( isValReal()    && b.isValMatrix()  ) { double temp = cast_double(0);  *this = b; inverse(); *this *= temp;         } 
    else if ( isValReal()    && b.isValString()  ) { goto errcase; }
    else if ( isValReal()    && b.isValSet()     ) { goto errcase; }
    else if ( isValReal()    && b.isValDgraph()  ) { goto errcase; }

    else if ( isValAnion()   && b.isValNull()    ) { dir_anion()  /= (double) b.cast_int(0);    }
    else if ( isValAnion()   && b.isValInteger() ) { dir_anion()  /= (double) b.cast_int(0);    }
    else if ( isValAnion()   && b.isValReal()    ) { dir_anion()  /=          b.cast_double(0); }
    else if ( isValAnion()   && b.isValAnion()   ) { ::rightmult(inv(b.cast_anion(0)),dir_anion());  }
    else if ( isValAnion()   && b.isValVector()  ) { goto errcase; }
    else if ( isValAnion()   && b.isValMatrix()  ) { gentype temp(cast_anion(0)); *this = b; inverse(); leftmult(temp); } 
    else if ( isValAnion()   && b.isValString()  ) { goto errcase; }
    else if ( isValAnion()   && b.isValSet()     ) { goto errcase; }
    else if ( isValAnion()   && b.isValDgraph()  ) { goto errcase; }

    else if ( isValVector()  && b.isValNull()    ) { dir_vector() *= inv(b); }
    else if ( isValVector()  && b.isValInteger() ) { dir_vector() *= inv(b); }
    else if ( isValVector()  && b.isValReal()    ) { dir_vector() *= inv(b); }
    else if ( isValVector()  && b.isValAnion()   ) { ::rightmult(inv(b),dir_vector()); }
    else if ( isValVector()  && b.isValVector()  ) { goto errcase; } 
    else if ( isValVector()  && b.isValMatrix()  ) { ::rightmult(inv(b.cast_matrix(0)),dir_vector()); } 
    else if ( isValVector()  && b.isValString()  ) { goto errcase; }
    else if ( isValVector()  && b.isValSet()     ) { goto errcase; }
    else if ( isValVector()  && b.isValDgraph()  ) { goto errcase; }

    else if ( isValMatrix()  && b.isValNull()    ) { dir_matrix() *= inv(b); }
    else if ( isValMatrix()  && b.isValInteger() ) { dir_matrix() *= inv(b); }
    else if ( isValMatrix()  && b.isValReal()    ) { dir_matrix() *= inv(b); }
    else if ( isValMatrix()  && b.isValAnion()   ) { ::rightmult(inv(b),dir_matrix()); }
    else if ( isValMatrix()  && b.isValVector()  ) { goto errcase; }
    else if ( isValMatrix()  && b.isValMatrix()  ) { ::rightmult(inv(b.cast_matrix(0)),dir_matrix()); }  
    else if ( isValMatrix()  && b.isValString()  ) { goto errcase; }
    else if ( isValMatrix()  && b.isValSet()     ) { goto errcase; }
    else if ( isValMatrix()  && b.isValDgraph()  ) { goto errcase; }

    else if ( isValString()  && b.isValNull()    ) { goto errcase; }
    else if ( isValString()  && b.isValInteger() ) { goto errcase; }
    else if ( isValString()  && b.isValReal()    ) { goto errcase; }
    else if ( isValString()  && b.isValAnion()   ) { goto errcase; }
    else if ( isValString()  && b.isValVector()  ) { goto errcase; }
    else if ( isValString()  && b.isValMatrix()  ) { goto errcase; } 
    else if ( isValString()  && b.isValString()  ) { goto errcase; } 
    else if ( isValString()  && b.isValSet()     ) { goto errcase; } 
    else if ( isValString()  && b.isValDgraph()  ) { goto errcase; }

    else if ( isValSet()     && b.isValNull()    ) { dir_set() *= inv(b); }
    else if ( isValSet()     && b.isValInteger() ) { dir_set() *= inv(b); }
    else if ( isValSet()     && b.isValReal()    ) { dir_set() *= inv(b); }
    else if ( isValSet()     && b.isValAnion()   ) { ::rightmult(inv(b),dir_set()); }
    else if ( isValSet()     && b.isValVector()  ) { goto errcase; }
    else if ( isValSet()     && b.isValMatrix()  ) { goto errcase; }
    else if ( isValSet()     && b.isValString()  ) { goto errcase; }
    else if ( isValSet()     && b.isValSet()     ) { goto errcase; }
    else if ( isValSet()     && b.isValDgraph()  ) { goto errcase; }

    else if ( isValDgraph()  && b.isValNull()    ) { goto errcase; }
    else if ( isValDgraph()  && b.isValInteger() ) { goto errcase; }
    else if ( isValDgraph()  && b.isValReal()    ) { goto errcase; }
    else if ( isValDgraph()  && b.isValAnion()   ) { goto errcase; }
    else if ( isValDgraph()  && b.isValVector()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValMatrix()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValString()  ) { goto errcase; }
    else if ( isValDgraph()  && b.isValSet()     ) { goto errcase; }
    else if ( isValDgraph()  && b.isValDgraph()  ) { goto errcase; }

    else
    {
    errcase:
        // Note reversal of a and b here

        std::string errstr("Error: incompatible types in division rightdiv(");
        errstr += b.cast_string(0);
        errstr += ",";
        errstr += cast_string(0);
        errstr += ")";

        constructError(*this,b,*this,errstr.c_str());
    }

    return *this;
}

gentype &gentype::leftadd(const gentype &b)
{
    if ( isCastableToIntegerWithoutLoss() )
    {
        if ( cast_int(0) == 0 ) { fastcopy(b); return *this; }
    }

    if ( b.isCastableToIntegerWithoutLoss() )
    {
        if ( b.cast_int(0) == 0 ) { return *this; }
    }

    if ( isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype tempres("x+y");
        gentype res(tempres);

        // Method: see leftmult

        if ( varid_isscalar )
        {
            res.varid_isscalar = varid_isscalar;
            res.varid_numpts   = varid_numpts;

            res.varid_xi = varid_xi;
            res.varid_xj = varid_xj;
        }

        if ( b.varid_isscalar )
        {
            res.varid_isscalar = b.varid_isscalar;
            res.varid_numpts   = b.varid_numpts;

            res.varid_xi = b.varid_xi;
            res.varid_xj = b.varid_xj;
        }

        qswap(*this,res);
        qswap((*(eqnargs))("&",0),res);
        (*(eqnargs))("&",1).fastcopy(b,1);

        if ( (*(eqnargs))(zeroint()).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(zeroint()).varid_isscalar;
            varid_numpts   = (*(eqnargs))(zeroint()).varid_numpts;

            varid_xi = (*(eqnargs))(zeroint()).varid_xi;
            varid_xj = (*(eqnargs))(zeroint()).varid_xj;

            (*(eqnargs))("&",zeroint()).varid_isscalar = 0;
        }

        if ( (*(eqnargs))(1).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(1).varid_isscalar;
            varid_numpts   = (*(eqnargs))(1).varid_numpts;

            varid_xi = (*(eqnargs))(1).varid_xi;
            varid_xj = (*(eqnargs))(1).varid_xj;

            (*(eqnargs))("&",1).varid_isscalar = 0;
        }

        return *this;
    }

    if ( isfasttype() && b.isfasttype() )
    {
             if ( isValNull()    && b.isValNull()    ) { ; }
        else if ( isValNull()    && b.isValInteger() ) { dir_int()    += b.cast_int(0);    }
        else if ( isValNull()    && b.isValReal()    ) { dir_double() += b.cast_double(0); }
        else if ( isValNull()    && b.isValAnion()   ) { dir_anion()  += b.cast_anion(0);  }
        else if ( isValInteger() && b.isValNull()    ) { dir_int()    += b.cast_int(0);    }
        else if ( isValInteger() && b.isValInteger() ) { dir_int()    += b.cast_int(0);    }
        else if ( isValInteger() && b.isValReal()    ) { dir_double() += b.cast_double(0); }
        else if ( isValInteger() && b.isValAnion()   ) { dir_anion()  += b.cast_anion(0);  }
        else if ( isValReal()    && b.isValNull()    ) { dir_double() += b.cast_int(0);    }
        else if ( isValReal()    && b.isValInteger() ) { dir_double() += b.cast_int(0);    }
        else if ( isValReal()    && b.isValReal()    ) { dir_double() += b.cast_double(0); }
        else if ( isValReal()    && b.isValAnion()   ) { dir_anion()  += b.cast_anion(0);  }
        else if ( isValAnion()   && b.isValNull()    ) { dir_anion()  += b.cast_double(0);  }
        else if ( isValAnion()   && b.isValInteger() ) { dir_anion()  += b.cast_double(0);  }
        else if ( isValAnion()   && b.isValReal()    ) { dir_anion()  += b.cast_double(0); }
        else if ( isValAnion()   && b.isValAnion()   ) { dir_anion()  += b.cast_anion(0);  }

        return *this;
    }

    int acast = 0;
    int bcast = 0;
    int rescast = checkAddCompat(*this,b,acast,bcast);

    if ( rescast && ( rescast != 6 ) )
    {
             if ( ( acast == 1  ) && ( bcast == 1  ) ) { dir_int()    += b.cast_int(0);    }
        else if ( ( acast == 2  ) && ( bcast == 2  ) ) { dir_double() += b.cast_double(0); }
        else if ( ( acast == 3  ) && ( bcast == 3  ) ) { dir_anion()  += b.cast_anion(0);  }
        else if ( ( acast == 4  ) && ( bcast == 4  ) ) { dir_vector() += b.cast_vector(0); }
        else if ( ( acast == 4  ) && ( bcast == 7  ) ) { dir_vector() += b;               }
        else if ( ( acast == 5  ) && ( bcast == 5  ) ) { dir_matrix() += b.cast_matrix(0); }
        else if ( ( acast == 5  ) && ( bcast == 7  ) ) { dir_matrix() += b;               }
        else if ( ( acast == 8  ) && ( bcast == 7  ) ) { dir_set()    += b;               }
        else if ( ( acast == 8  ) && ( bcast == 8  ) ) { dir_set()    += b.cast_set(0);    }
        else if ( ( acast == 7  ) && ( bcast == 10 ) ) { *this = b; }
        else if ( ( acast == 10 ) && ( bcast == 7  ) ) { ; }
        else if ( ( acast == 10 ) && ( bcast == 10 ) ) { ; }
        else if ( ( acast == 7  ) && ( bcast == 4  ) ) { *this = ( *this + b.cast_vector(0) ); }
        else if ( ( acast == 7  ) && ( bcast == 8  ) ) { *this = ( *this + b.cast_set(0)    ); }
        else                                           { *this = ( *this + b.cast_matrix(0) ); }
    }

    else if ( rescast == 6 )
    {
        std::string leftpart  = cast_string(0);
        std::string rightpart = b.cast_string(0);

        leftpart += rightpart;
        makeString(leftpart);
    }

    else
    {
        constructError(*this,b,*this,"Error: incompatible types in addition.");
    }

    return *this;
}

gentype &gentype::leftsub(const gentype &b)
{
    if ( isCastableToIntegerWithoutLoss() )
    {
        if ( cast_int(0) == 0 ) { fastcopy(b); negate(); return *this; }
    }

    if ( b.isCastableToIntegerWithoutLoss() )
    {
        if ( b.cast_int(0) == 0 ) { return *this; }
    }

    if ( isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype tempres("x-y");
        gentype res(tempres);
        // Method: see leftmult

        if ( varid_isscalar )
        {
            res.varid_isscalar = varid_isscalar;
            res.varid_numpts   = varid_numpts;

            res.varid_xi = varid_xi;
            res.varid_xj = varid_xj;
        }

        if ( b.varid_isscalar )
        {
            res.varid_isscalar = b.varid_isscalar;
            res.varid_numpts   = b.varid_numpts;

            res.varid_xi = b.varid_xi;
            res.varid_xj = b.varid_xj;
        }

        qswap(*this,res);
        qswap((*(eqnargs))("&",0),res);
        (*((*(eqnargs))("&",1).eqnargs))("&",0).fastcopy(b,1);

        if ( (*(eqnargs))(zeroint()).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(zeroint()).varid_isscalar;
            varid_numpts   = (*(eqnargs))(zeroint()).varid_numpts;

            varid_xi = (*(eqnargs))(zeroint()).varid_xi;
            varid_xj = (*(eqnargs))(zeroint()).varid_xj;

            (*(eqnargs))("&",zeroint()).varid_isscalar = 0;
        }

        if ( (*(eqnargs))(1).varid_isscalar )
        {
            varid_isscalar = (*(eqnargs))(1).varid_isscalar;
            varid_numpts   = (*(eqnargs))(1).varid_numpts;

            varid_xi = (*(eqnargs))(1).varid_xi;
            varid_xj = (*(eqnargs))(1).varid_xj;

            (*(eqnargs))("&",1).varid_isscalar = 0;
        }

        return *this;
    }

    if ( isfasttype() && b.isfasttype() )
    {
             if ( isValNull()    && b.isValNull()    ) { ; }
        else if ( isValNull()    && b.isValInteger() ) { dir_int()    -= b.cast_int(0);    }
        else if ( isValNull()    && b.isValReal()    ) { dir_double() -= b.cast_double(0); }
        else if ( isValNull()    && b.isValAnion()   ) { dir_anion()  -= b.cast_anion(0);  }
        else if ( isValInteger() && b.isValNull()    ) { dir_int()    -= b.cast_int(0);    }
        else if ( isValInteger() && b.isValInteger() ) { dir_int()    -= b.cast_int(0);    }
        else if ( isValInteger() && b.isValReal()    ) { dir_double() -= b.cast_double(0); }
        else if ( isValInteger() && b.isValAnion()   ) { dir_anion()  -= b.cast_anion(0);  }
        else if ( isValReal()    && b.isValNull()    ) { dir_double() -= b.cast_int(0);    }
        else if ( isValReal()    && b.isValInteger() ) { dir_double() -= b.cast_int(0);    }
        else if ( isValReal()    && b.isValReal()    ) { dir_double() -= b.cast_double(0); }
        else if ( isValReal()    && b.isValAnion()   ) { dir_anion()  -= b.cast_anion(0);  }
        else if ( isValAnion()   && b.isValNull()    ) { dir_anion()  -= b.cast_double(0);  }
        else if ( isValAnion()   && b.isValInteger() ) { dir_anion()  -= b.cast_double(0);  }
        else if ( isValAnion()   && b.isValReal()    ) { dir_anion()  -= b.cast_double(0); }
        else if ( isValAnion()   && b.isValAnion()   ) { dir_anion()  -= b.cast_anion(0);  }

        return *this;
    }

    int acast = 0;
    int bcast = 0;
    int rescast = checkAddCompat(*this,b,acast,bcast);

    if ( rescast && ( rescast != 6 ) )
    {
             if ( ( acast == 1  ) && ( bcast == 1  ) ) { dir_int()    -= b.cast_int(0);    }
        else if ( ( acast == 2  ) && ( bcast == 2  ) ) { dir_double() -= b.cast_double(0); }
        else if ( ( acast == 3  ) && ( bcast == 3  ) ) { dir_anion()  -= b.cast_anion(0);  }
        else if ( ( acast == 4  ) && ( bcast == 4  ) ) { dir_vector() -= b.cast_vector(0); }
        else if ( ( acast == 4  ) && ( bcast == 7  ) ) { dir_vector() -= b;               }
        else if ( ( acast == 5  ) && ( bcast == 5  ) ) { dir_matrix() -= b.cast_matrix(0); }
        else if ( ( acast == 5  ) && ( bcast == 7  ) ) { dir_matrix() -= b;               }
        else if ( ( acast == 8  ) && ( bcast == 7  ) ) { dir_set()    -= b;               }
        else if ( ( acast == 8  ) && ( bcast == 8  ) ) { dir_set()    -= b.cast_set(0);    }
        else if ( ( acast == 7  ) && ( bcast == 10 ) ) { *this = b; negate(); }
        else if ( ( acast == 10 ) && ( bcast == 7  ) ) { ; }
        else if ( ( acast == 10 ) && ( bcast == 10 ) ) { ; }
        else if ( ( acast == 7  ) && ( bcast == 4  ) ) { *this = ( *this - b.cast_vector(0) ); }
        else if ( ( acast == 7  ) && ( bcast == 8  ) ) { *this = ( *this - b.cast_set(0)    ); }
        else                                           { *this = ( *this - b.cast_matrix(0) ); }
    }

    else
    {
        constructError(*this,b,*this,"Error: incompatible types in subtraction.");
    }

    return *this;
}





















gentype &operator+=(gentype &a, const int &bb)
{
    if ( bb == 0 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = bb;
	    }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a += b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_int()    += bb;          }
    else if ( a.isValReal()    ) {                a.dir_double() += bb;          }
    else if ( a.isValAnion()   ) {                a.dir_anion()  += (double) bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() += b;           }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() += b;           }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    += b;           }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot add a graph and an int."); }

    else
    {
        constructError(a,a,"Error: incompatible types in addition.");
    }

    return a;
}

gentype &operator-=(gentype &a, const int &bb)
{
    if ( bb == 0 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = -bb;
	    }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a -= b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_int()    -= bb;          }
    else if ( a.isValReal()    ) {                a.dir_double() -= bb;          }
    else if ( a.isValAnion()   ) {                a.dir_anion()  -= (double) bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() -= b;           }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() -= b;           }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    -= b;           }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot subtract an int from a graph."); }

    else
    {
        constructError(a,a,"Error: incompatible types in subtraction.");
    }

    return a;
}

gentype &operator*=(gentype &a, const int &bb)
{
    //NB:  need to keep structure of result, even if it is zero
    //if ( bb == 0 )
    //{
    //    a = zeroint();
    //}

    if ( bb == 1 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = zeroint();
	    }

            else if ( a.cast_int(0) == 1 )
            {
                a = bb;
            }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a *= b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_int()    *= bb;          }
    else if ( a.isValReal()    ) {                a.dir_double() *= bb;          }
    else if ( a.isValAnion()   ) {                a.dir_anion()  *= (double) bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() *= b;           }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() *= b;           }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    *= b;           }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot multiply a graph and a int."); }

    else
    {
        std::string errstr("Error: incompatible types in multiplication ");
        errstr += a.cast_string(0);
        errstr += "*=double";

        constructError(a,a,errstr.c_str());
    }

    return a;
}

gentype &operator+=(gentype &a, const double  &bb)
{
    if ( bb == 0 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = bb;
	    }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a += b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_double() += bb; }
    else if ( a.isValReal()    ) {                a.dir_double() += bb; }
    else if ( a.isValAnion()   ) {                a.dir_anion()  += bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() += b;  }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() += b;  }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    += b;  }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot add a graph and a double."); }

    else
    {
        constructError(a,a,"Error: incompatible types in addition.");
    }

    return a;
}

gentype &operator-=(gentype &a, const double  &bb)
{
    if ( bb == 0 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = -bb;
	    }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a -= b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_double() -= bb; }
    else if ( a.isValReal()    ) {                a.dir_double() -= bb; }
    else if ( a.isValAnion()   ) {                a.dir_anion()  -= bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() -= b;  }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() -= b;  }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    -= b;  }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot subtract a double from a graph."); }

    else
    {
        constructError(a,a,"Error: incompatible types in subtraction.");
    }

    return a;
}

gentype &operator*=(gentype &a, const double  &bb)
{
    //NB:  need to keep structure of result, even if it is zero
    //if ( bb == 0 )
    //{
    //    a = zeroint();
    //}

    if ( bb == 1 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = zeroint();
	    }

            else if ( a.cast_int(0) == 1 )
            {
                a = bb;
            }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a *= b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_double() *= bb; }
    else if ( a.isValReal()    ) {                a.dir_double() *= bb; }
    else if ( a.isValAnion()   ) {                a.dir_anion()  *= bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() *= b;  }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() *= b;  }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    *= b;  }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot multiply a graph and a double."); }

    else
    {
        std::string errstr("Error: incompatible types in multiplication ");
        errstr += a.cast_string(0);
        errstr += "*=double";

        constructError(a,a,errstr.c_str());
    }

    return a;
}

gentype &operator+=(gentype &a, const d_anion  &bb)
{
    if ( bb == 0 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = bb;
	    }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a += b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_anion()  += bb; }
    else if ( a.isValReal()    ) {                a.dir_anion()  += bb; }
    else if ( a.isValAnion()   ) {                a.dir_anion()  += bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() += b;  }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() += b;  }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    += b;  }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot add a graph and a double."); }

    else
    {
        constructError(a,a,"Error: incompatible types in addition.");
    }

    return a;
}

gentype &operator-=(gentype &a, const d_anion  &bb)
{
    if ( bb == 0 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = -bb;
	    }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a -= b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_anion()  -= bb; }
    else if ( a.isValReal()    ) {                a.dir_anion()  -= bb; }
    else if ( a.isValAnion()   ) {                a.dir_anion()  -= bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() -= b;  }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() -= b;  }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    -= b;  }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot subtract a double from a graph."); }

    else
    {
        constructError(a,a,"Error: incompatible types in subtraction.");
    }

    return a;
}

gentype &operator*=(gentype &a, const d_anion  &bb)
{
    //NB:  need to keep structure of result, even if it is zero
    //if ( bb == 0 )
    //{
    //    a = zeroint();
    //}

    if ( bb == 1 )
    {
        ;
    }

    else if ( a.isValEqnDir() )
    {
	if ( a.isCastableToIntegerWithoutLoss() )
	{
	    if ( !(a.cast_int(0)) )
	    {
                a = zeroint();
	    }

            else if ( a.cast_int(0) == 1 )
            {
                a = bb;
            }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            gentype b(bb);

            a *= b;
	}
    }

    else if ( a.isValNull()    ) { ; }
    else if ( a.isValInteger() ) {                a.dir_anion()  *= bb; }
    else if ( a.isValReal()    ) {                a.dir_anion()  *= bb; }
    else if ( a.isValAnion()   ) {                a.dir_anion()  *= bb; }
    else if ( a.isValVector()  ) { gentype b(bb); a.dir_vector() *= b;  }
    else if ( a.isValMatrix()  ) { gentype b(bb); a.dir_matrix() *= b;  }
    else if ( a.isValSet()     ) { gentype b(bb); a.dir_set()    *= b;  }
    else if ( a.isValDgraph()  ) { constructError(a,a,"Error: cannot multiply a graph and a double."); }

    else
    {
        std::string errstr("Error: incompatible types in multiplication ");
        errstr += a.cast_string(0);
        errstr += "*=double";

        constructError(a,a,errstr.c_str());
    }

    return a;
}



















































































// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
//
// Generic form functions for various types of maths functions
//
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================


// fname        = the function name
// elmfn        = the elementwise function
// anionfn      = the anion valued function used otherwise (NULL if restanionfn used instead)
// restanionfn  = the real-value, anion-argument form
// doublefn     = the double valued function
// intfn        = the integer-valued function, if it is defined and returns an integer.
//                use NULL if no such function is available.
// outRangeTest = returns 0 if the function will return a real for the given real
//                argument, 1 if it must return an anion for the given real argument
// andor = 0 if logic test is combined with OR, 1 if logic is AND



int trueOR(double dummy);
int falseOR(double dummy);
int invertOR(int a);
int andOR(int a, int b);
int orOR(int a, int b);

int trueOR(double dummy)
{
    (void) dummy;

    return 1;
}

int falseOR(double dummy)
{
    (void) dummy;

    return 0;
}

int invertOR(int a)
{
    return !a;
}

int andOR(int a, int b)
{
    return a && b;
}

int orOR(int a, int b)
{
    return a || b;
}


gentype &OP_elementwiseDefaultCallA(gentype &a, const gentype &fnbare, const char *fname, gentype &(*op_elmfn)(gentype &), d_anion (*anionfn)(const d_anion &), double (*restanionfn)(const d_anion &), double (*doublefn)(double), int (*intfn)(int), int (*outRangeTest)(double));
gentype &elementwiseDefaultCallB(gentype &res, const gentype &a, const gentype &b, const char *fname, gentype (*elmfn)(const gentype &, const void *), gentype (*specfn)(const gentype &, const gentype &));
gentype &elementwiseDefaultCallC(gentype &res, const gentype &a, const gentype &b, const char *fname, gentype (*elmfn)(const gentype &, const gentype &), gentype (*fn)(const gentype &, const gentype &), int (*intfn)(int,int));
gentype &maxmincommonform(gentype &res, const gentype &a, const char *fname, gentype (*matfn)(const Matrix<gentype> &, int &, int &), gentype (*vecfn)(const Vector<gentype> &, int &), gentype (*basevecfn)(const Vector<gentype> &), gentype (*setfn)(const Set<gentype> &), int &i, int &j, int getarg, int absres, int allres);
gentype &zaxmincommonform(gentype &res, const gentype &a, const char *fname, const gentype &(*matfn)(const Matrix<gentype> &, int &, int &), const gentype &(*vecfn)(const Vector<gentype> &, int &), const gentype &(*basevecfn)(const Vector<gentype> &), const gentype &(*setfn)(const Set<gentype> &), int &i, int &j, int getarg, int absres, int allres);
gentype &binLogicForm(gentype &res, const gentype &a, const gentype &b, const char *opname, int (*elmfn)(const gentype &, const gentype &), int andor);

gentype &OP_elementwiseDefaultCallA(gentype &a, const gentype &fnbare, const char *fname, gentype &(*elmfn)(gentype &), d_anion (*anionfn)(const d_anion &), double (*restanionfn)(const d_anion &), double (*doublefn)(double), int (*intfn)(int), int (*outRangeTest)(double))
{
    if ( a.isValEqnDir() )
    {
        //phantomxyz
        //std::string resstr;
        //gentype res;
        //
        //resstr = fname;
        //resstr += "(x)";
        //res = resstr;
        //a = res(a);

        a = fnbare(a);
    }

    else if ( anionfn != NULL )
    {
             if ( a.isValStrErr()                       ) { std::string errstr = fname; errstr += " ill-defined for string"; constructError(a,a,errstr); }
        else if ( a.isValDgraph()                       ) { std::string errstr = fname; errstr += " ill-defined for dgraph"; constructError(a,a,errstr); }
        else if ( a.isValSet()                          ) { (a.dir_set()   ).applyon(elmfn); }
        else if ( a.isValMatrix()                       ) { (a.dir_matrix()).applyon(elmfn); }
        else if ( a.isValVector()                       ) { (a.dir_vector()).applyon(elmfn); }
        else if ( a.isValAnion()                        ) { a.dir_anion()  = anionfn( a.cast_anion(0) ); }
        else if ( outRangeTest(a.cast_double(0))         ) { a.dir_anion()  = anionfn( a.cast_anion(0) ); }
        else if ( a.isValReal()                         ) { a.dir_double() = doublefn(a.cast_double(0)); }
        else if ( a.isValInteger() && ( intfn == NULL ) ) { a.dir_double() = doublefn(a.cast_double(0)); }
        else if ( a.isValNull()                         ) { ; }
        else                                              { a.dir_int()    = intfn(   a.cast_int(0)   ); }
    }

    else if ( restanionfn != NULL )
    {
             if ( a.isValStrErr()                       ) { std::string errstr = fname; errstr += " ill-defined for string"; constructError(a,a,errstr); }
        else if ( a.isValDgraph()                       ) { std::string errstr = fname; errstr += " ill-defined for dgraph"; constructError(a,a,errstr); }
        else if ( a.isValSet()                          ) { (a.dir_set()   ).applyon(elmfn); }
        else if ( a.isValMatrix()                       ) { (a.dir_matrix()).applyon(elmfn); }
        else if ( a.isValVector()                       ) { (a.dir_vector()).applyon(elmfn); }
        else if ( a.isValAnion()                        ) { a.dir_anion()  = restanionfn(a.cast_anion(0) ); }
        else if ( outRangeTest(a.cast_double(0))         ) { a.dir_anion()  = restanionfn(a.cast_anion(0) ); }
        else if ( a.isValReal()                         ) { a.dir_double() = doublefn(   a.cast_double(0)); }
        else if ( a.isValInteger() && ( intfn == NULL ) ) { a.dir_double() = doublefn(   a.cast_double(0)); }
        else if ( a.isValNull()                         ) { ; }
        else                                              { a.dir_int()    = intfn(      a.cast_int(0)   ); }
    }

    else if ( doublefn != NULL )
    {
             if ( a.isValStrErr()                       ) { std::string errstr = fname; errstr += " ill-defined for string"; constructError(a,a,errstr); }
        else if ( a.isValDgraph()                       ) { std::string errstr = fname; errstr += " ill-defined for dgraph"; constructError(a,a,errstr); }
        else if ( a.isValSet()                          ) { (a.dir_set()   ).applyon(elmfn); }
        else if ( a.isValMatrix()                       ) { (a.dir_matrix()).applyon(elmfn); }
        else if ( a.isValVector()                       ) { (a.dir_vector()).applyon(elmfn); }
        else if ( a.isValAnion()                        ) { std::string errstr = fname; errstr += " ill-defined for anions"; constructError(a,a,errstr); }
        else if ( outRangeTest(a.cast_double(0))         ) { std::string errstr = fname; errstr += " ill-defined for out-of-range reals"; constructError(a,a,errstr); }
        else if ( a.isValReal()                         ) { a.dir_double() = doublefn(a.cast_double(0)); }
        else if ( a.isValInteger() && ( intfn == NULL ) ) { a.dir_double() = doublefn(a.cast_double(0)); }
        else if ( a.isValNull()                         ) { ; }
        else                                              { a.dir_int()    = intfn(   a.cast_int(0)   ); }
    }

    else
    {
             if ( a.isValStrErr()                       ) { std::string errstr = fname; errstr += " ill-defined for string"; constructError(a,a,errstr); }
        else if ( a.isValDgraph()                       ) { std::string errstr = fname; errstr += " ill-defined for dgraph"; constructError(a,a,errstr); }
        else if ( a.isValSet()                          ) { (a.dir_set()   ).applyon(elmfn); }
        else if ( a.isValMatrix()                       ) { (a.dir_matrix()).applyon(elmfn); }
        else if ( a.isValVector()                       ) { (a.dir_vector()).applyon(elmfn); }
        else if ( a.isValAnion()                        ) { std::string errstr = fname; errstr += " ill-defined for anions"; constructError(a,a,errstr); }
        else if ( outRangeTest(a.cast_double(0))         ) { std::string errstr = fname; errstr += " ill-defined for out-of-range reals"; constructError(a,a,errstr); }
        else if ( a.isValReal()                         ) { std::string errstr = fname; errstr += " ill-defined for reals"; constructError(a,a,errstr); }
        else if ( a.isValInteger() && ( intfn == NULL ) ) { std::string errstr = fname; errstr += " ill-defined for integers out of range"; constructError(a,a,errstr); }
        else if ( a.isValNull()                         ) { ; }
        else                                              { a.dir_int() = intfn(a.cast_int(0)); }
    }

    return a;
}

gentype &elementwiseDefaultCallB(gentype &res, const gentype &a, const gentype &b, const char *fname, gentype (*elmfn)(const gentype &, const void *), gentype (*specfn)(const gentype &, const gentype &))
{
    if ( a.isValEqnDir() )
    {
        std::string resstr;

	resstr = fname;
        resstr += "(x,y)";
        res = resstr;
	res = res(a,b);
    }

    else
    {
	     if ( a.isValStrErr() ) { std::string errstr = fname; errstr += " ill-defined for string"; constructError(a,res,errstr); }
        else if ( a.isValDgraph() ) { std::string errstr = fname; errstr += " ill-defined for dgraph"; constructError(a,res,errstr); }
        else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(elmfn, (const void *) &b); res = temp; }
	else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(elmfn, (const void *) &b); res = temp; }
	else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(elmfn, (const void *) &b); res = temp; }
	else                        { res = specfn(a,b); }
    }

    return res;
}

// NB: elementwise operations propogate, so for example if you have to matrices of matrices
// and you elementwise multiply them then the individual elements will be elementwise multiplied
// together.

gentype &elementwiseDefaultCallC(gentype &res, const gentype &a, const gentype &b, const char *fname, gentype (*elmfn)(const gentype &, const gentype &), gentype (*fn)(const gentype &, const gentype &), int (*intfn)(int, int))
{
    if ( a.isValMatrix() || b.isValMatrix() )
    {
        // Case 1: either operand is a matrix, so recurse, either by
        // testing dimensions agree and applying elementwise or by
        // matricising either a and b as required and retrying.

	if ( !a.isValMatrix() )
	{
	    Matrix<gentype> aaa(b.numRows(),b.numCols());
	    aaa = a;
	    gentype aa = aaa;
            elementwiseDefaultCallC(res,aa,b,fname,elmfn,fn,intfn);
	}

	else if ( !b.isValMatrix() )
	{
	    Matrix<gentype> bbb(a.numRows(),a.numCols());
	    bbb = b;
	    gentype bb = bbb;
            elementwiseDefaultCallC(res,a,bb,fname,elmfn,fn,intfn);
	}

	else if ( ( a.numRows() == b.numRows() ) && ( a.numCols() == b.numCols() ) )
	{
	    int i,j;

            Matrix<gentype> mres(a.cast_matrix(0));
	    Matrix<gentype> aa(a.cast_matrix(0));
	    Matrix<gentype> bb(b.cast_matrix(0));

            if ( a.numRows() && a.numCols() )
            {
                for ( i = 0 ; i < a.numRows() ; i++ )
	        {
	            for ( j = 0 ; j < a.numCols() ; j++ )
	            {
                        mres("&",i,j) = elmfn(aa(i,j),bb(i,j));
	            }
		}
            }

            res = mres;
	}

	else
	{
	    std::string errstr = fname;
	    errstr += " matrix dimensions do not agree";
	    constructError(a,b,res,errstr);
	}
    }

    else if ( a.isValVector() || b.isValVector() )
    {
        // Case 2: either operand is a vector, so recurse, either by
        // testing dimensions agree and applying elementwise or by
        // vectorising either a and b as required and retrying.

	if ( !a.isValVector() )
	{
	    Vector<gentype> aaa(b.size());
	    aaa = a;
	    gentype aa(aaa);
            elementwiseDefaultCallC(res,aa,b,fname,elmfn,fn,intfn);
	}

	else if ( !b.isValVector() )
	{
	    Vector<gentype> bbb(a.size());
	    bbb = b;
	    gentype bb(bbb);
            elementwiseDefaultCallC(res,a,bb,fname,elmfn,fn,intfn);
	}

	else if ( a.size() == b.size() )
	{
	    int i;

            Vector<gentype> vres(a.cast_vector(0));
	    Vector<gentype> aa(a.cast_vector(0));
	    Vector<gentype> bb(b.cast_vector(0));

            if ( a.size() )
            {
	        for ( i = 0 ; i < a.size() ; i++ )
	        {
		    vres("&",i) = elmfn(aa(i),bb(i));
		}
	    }

            res = vres;
	}

	else
	{
	    std::string errstr = fname;
	    errstr += " vector dimensions do not agree";
	    constructError(a,b,res,errstr);
	}
    }

    else if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        // Case 3: either operand is a function, so leave as-is.

        std::string resstr = fname;
	resstr += "(x,y)";
	res = resstr;
	res = res(a,b);
    }

    else if ( fn != NULL )
    {
        // Case 4: no vectors, matrices or equations present.  Proceed with
        // non-elementwise function if well defined.

        res = fn(a,b);
    }

    else if ( a.isValNull() && b.isValNull() )
    {
        res.makeNull();
    }

    else if ( a.isCastableToIntegerWithoutLoss() && b.isCastableToIntegerWithoutLoss() && ( intfn != NULL ) )
    {
        // Case 5: no vectors, matrices or equations present, no
        // non-elementwise function defined, but both are integers (or
        // castable to without loss), and integer function is defined, so
        // use that.

        res = intfn(a.cast_int(0),b.cast_int(0));
    }

    else
    {
        // Case 6: fail-through.

        std::string errstr = fname;
	errstr += " defined only for bool/integer.";
	constructError(a,b,res,errstr);
    }

    return res;
}

gentype &maxmincommonform(gentype &res, const gentype &a, const char *fname, gentype (*matfn)(const Matrix<gentype> &, int &, int &), gentype (*vecfn)(const Vector<gentype> &, int &), gentype (*basevecfn)(const Vector<gentype> &), gentype (*setfn)(const Set<gentype> &), int &i, int &j, int getarg, int absres, int allres)
{
    i = 0;
    j = 0;
    std::string fnname = fname;

    if ( a.isValEqnDir() )
    {
	std::string resstr = fname;
        resstr += "(x)";
	res = resstr;
        res = res(a);
    }

    if ( a.isValStrErr() )
    {
        constructError(a,res,"String "+fnname+" not implemented");
    }

    else if ( a.isValMatrix() && ( matfn != NULL ) )
    {
        res = matfn(a.cast_matrix(0),i,j);

        if ( getarg )
        {
            Vector<gentype> ii(2);

            ii("&",0) = i;
            ii("&",1) = j;

            res = ii;
        }
    }

    else if ( a.isValMatrix() )
    {
        constructError(a,res,"Matrix "+fnname+" not implemented");
    }

    else if ( a.isValVector() && ( vecfn != NULL ) )
    {
        res = vecfn(a.cast_vector(0),i);

        if ( getarg && allres )
        {
            Vector<gentype> ii(0);
            gentype iii(i);

            if ( a.size() )
            {
                int k;
                gentype kk;

                for ( k = 0 ; k < a.size() ; k++ )
                {
                    kk = k;

                    if ( !absres )
                    {
                        if ( derefv(a,iii) == derefv(a,kk) )
                        {
                            ii.add(ii.size());
                            ii("&",ii.size()-1) = kk;
                        }
                    }

                    else
                    {
                        if ( abs2(derefv(a,iii)) == abs2(derefv(a,kk)) )
                        {
                            ii.add(ii.size());
                            ii("&",ii.size()-1) = kk;
                        }
                    }
                }
            }

            gentype iiii(ii);
            Vector<gentype> ayeayecaptn(1);
            ayeayecaptn("&",0) = iiii;
            res = ayeayecaptn;
        }

        else if ( getarg )
        {
            Vector<gentype> ii(1);
            ii("&",0) = i;
            res = ii;
        }
    }

    else if ( a.isValSet() && ( setfn != NULL ) )
    {
        res = setfn(a.cast_set(0));
    }

    else if ( a.isValVector() && ( basevecfn != NULL ) )
    {
        res = basevecfn(a.cast_vector(0));
    }

    else if ( a.isValVector() )
    {
        constructError(a,res,"Vector "+fnname+" not implemented");
    }

    else if ( getarg )
    {
        Vector<gentype> ii(zeroint());
        res = ii;
    }

    else if ( absres )
    {
        res = abs2(a);
    }

    else
    {
        res = a;
    }

    return res;
}

gentype &zaxmincommonform(gentype &res, const gentype &a, const char *fname, const gentype &(*matfn)(const Matrix<gentype> &, int &, int &), const gentype &(*vecfn)(const Vector<gentype> &, int &), const gentype &(*basevecfn)(const Vector<gentype> &), const gentype &(*setfn)(const Set<gentype> &), int &i, int &j, int getarg, int absres, int allres)
{
    i = 0;
    j = 0;
    std::string fnname = fname;

    if ( a.isValEqnDir() )
    {
	std::string resstr = fname;
        resstr += "(x)";
	res = resstr;
        res = res(a);
    }

    if ( a.isValStrErr() )
    {
        constructError(a,res,"String "+fnname+" not implemented");
    }

    else if ( a.isValMatrix() && ( matfn != NULL ) )
    {
        res = matfn(a.cast_matrix(0),i,j);

        if ( getarg )
        {
            Vector<gentype> ii(2);

            ii("&",0) = i;
            ii("&",1) = j;

            res = ii;
        }
    }

    else if ( a.isValMatrix() )
    {
        constructError(a,res,"Matrix "+fnname+" not implemented");
    }

    else if ( a.isValVector() && ( vecfn != NULL ) )
    {
        res = vecfn(a.cast_vector(0),i);

        if ( getarg && allres )
        {
            Vector<gentype> ii(0);
            gentype iii(i);

            if ( a.size() )
            {
                int k;
                gentype kk;

                for ( k = 0 ; k < a.size() ; k++ )
                {
                    kk = k;

                    if ( !absres )
                    {
                        if ( derefv(a,iii) == derefv(a,kk) )
                        {
                            ii.add(ii.size());
                            ii("&",ii.size()-1) = kk;
                        }
                    }

                    else
                    {
                        if ( abs2(derefv(a,iii)) == abs2(derefv(a,kk)) )
                        {
                            ii.add(ii.size());
                            ii("&",ii.size()-1) = kk;
                        }
                    }
                }
            }

            gentype iiii(ii);
            Vector<gentype> ayeayecaptn(1);
            ayeayecaptn("&",0) = iiii;
            res = ayeayecaptn;
        }

        else if ( getarg )
        {
            Vector<gentype> ii(1);
            ii("&",0) = i;
            res = ii;
        }
    }

    else if ( a.isValSet() && ( setfn != NULL ) )
    {
        res = setfn(a.cast_set(0));
    }

    else if ( a.isValVector() && ( basevecfn != NULL ) )
    {
        res = basevecfn(a.cast_vector(0));
    }

    else if ( a.isValVector() )
    {
        constructError(a,res,"Vector "+fnname+" not implemented");
    }

    else if ( getarg )
    {
        Vector<gentype> ii(zeroint());
        res = ii;
    }

    else if ( absres )
    {
        res = abs2(a);
    }

    else
    {
        res = a;
    }

    return res;
}


gentype &binLogicForm(gentype &res, const gentype &a, const gentype &b, const char *opname, int (*elmfn)(const gentype &, const gentype &), int andor)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
	std::string extname = opname;
        extname += "(x,y)";
	res = res(a,b);

        return res;
    }

    if ( a.isValMatrix() && b.isValMatrix() && ( a.numRows() == b.numRows() ) && ( a.numCols() == b.numCols() ) )
    {
        if ( a.numRows() && a.numCols() )
        {
            int i,j;
            Matrix<gentype> am(a.cast_matrix(0));
            Matrix<gentype> bm(b.cast_matrix(0));
            gentype temp;

            res = andor;

            for ( i = 0 ; i < a.numRows() ; i++ )
            {
                for ( j = 0 ; j < a.numCols() ; j++ )
                {
                    if ( andor )
                    {
                        res = land(res,binLogicForm(temp,am(i,j),bm(i,j),opname,elmfn,andor));
                    }

                    else
                    {
                        res = lor(res,binLogicForm(temp,am(i,j),bm(i,j),opname,elmfn,andor));
                    }
                }
            }
        }

        else
        {
            res = 1;
        }
    }

    else if ( a.isValVector() && b.isValVector() && ( a.size() == b.size() ) )
    {
        if ( a.size() )
        {
            int i;
            Vector<gentype> av(a.cast_vector(0));
            Vector<gentype> bv(b.cast_vector(0));
            gentype temp;

            res = andor;

            for ( i = 0 ; i < a.size() ; i++ )
            {
                if ( andor )
                {
                    res = land(res,binLogicForm(temp,av(i),bv(i),opname,elmfn,andor));
                }

                else
                {
                    res = lor(res,binLogicForm(temp,av(i),bv(i),opname,elmfn,andor));
                }
            }
        }

        else
        {
            res = 1;
        }
    }

    else if ( a.isValSet() && b.isValSet() && ( a.size() == b.size() ) )
    {
        if ( a.size() )
        {
            int i;
            Vector<gentype> av((a.cast_set(0)).all());
            Vector<gentype> bv((b.cast_set(0)).all());
            gentype temp;

            res = andor;

            for ( i = 0 ; i < a.size() ; i++ )
            {
                if ( andor )
                {
                    res = land(res,binLogicForm(temp,av(i),bv(i),opname,elmfn,andor));
                }

                else
                {
                    res = lor(res,binLogicForm(temp,av(i),bv(i),opname,elmfn,andor));
                }
            }
        }

        else
        {
            res = 1;
        }
    }

    else if ( a.isValMatrix() || b.isValMatrix() || a.isValVector() || b.isValVector() || a.isValSet() || b.isValSet() )
    {
        res = zeroint();
    }

    else
    {
	res = elmfn(a,b);
    }

    return res;
}











// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
//
// Non-trivial maths functions
//
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================
// =========================================================================================================================================================================


gentype realDeriv(const gentype &i, const gentype &j, const gentype &a)
{
    const static int varInd = getfnind("var");
    const static int VarInd = getfnind("Var");
    const static int gvarInd = getfnind("gvar");
    const static int gVarInd = getfnind("gVar");
    const static int realDerivInd = getfnind("realDeriv");

    int fnnameind = a.getfnnameind();

    if ( a.isValEqnDir() && !(a.realDerivDefinedDir()) && ( ( realDerivInd == fnnameind ) || ( fnnameind == gvarInd ) || ( fnnameind == gVarInd ) || ( fnnameind == varInd ) || ( VarInd == fnnameind ) ) )
    {
        const static gentype res("realDeriv(x,y,z)");

        return res(i,j,a);
    }

    else if ( a.isValEqnDir() && !(a.realDerivDefinedDir()) )
    {
        gentype res;

        constructError(i,j,a,res,"Derivative not defined for this function.");

        return res;
    }

    gentype res = a;
    res.realDeriv(i,j);

    return res;
}

gentype idiv(const gentype &a, const gentype &b)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype res("idiv(x,y)");

        return res(a,b);
    }

    gentype res;

    // Design choice: integer division should work for all numbers that can be
    // cast as integers without loss of precision, regardless of "actual" type.

    if ( a.isCastableToIntegerWithoutLoss() && b.isCastableToIntegerWithoutLoss() )
    {
        if ( b.cast_int(0) )
        {
            res = a.cast_int(0) / b.cast_int(0);
        }

        else
        {
            constructError(a,b,res,"divide by zero error");
        }
    }

    else
    {
	constructError(a,b,res,"idiv defined only for integers");
    }

    return res;
}

gentype &OP_idiv(gentype &a, const gentype &b)
{
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype res("idiv(x,y)");

        return a = res(a,b);
    }

    // Design choice: integer division should work for all numbers that can
    // be cast as integers without loss of precision, regardless of "actual"
    // type.

    if ( a.isCastableToIntegerWithoutLoss() && b.isCastableToIntegerWithoutLoss() )
    {
        if ( b.cast_int(0) )
        {
            a = a.cast_int(0) / b.cast_int(0);
        }

        else
        {
            constructError(a,b,a,"divide by zero error");
        }
    }

    else
    {
        constructError(a,b,a,"idiv defined only for integers");
    }

    return a;
}

gentype dubpow(const gentype &a, const void *b);
gentype dubpow(const gentype &a, const void *b)
{
    return pow(a,*((const gentype *) b));
}

gentype genpowintern(const gentype &a, const gentype &b, const char *powname, d_anion (*anionpow)(const d_anion &, const d_anion &));
gentype genpowintern(const gentype &a, const gentype &b, const char *powname, d_anion (*anionpow)(const d_anion &, const d_anion &))
{
    gentype res;
    std::string strpowname = powname;
                                         
    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
	if ( b.isCastableToIntegerWithoutLoss() )
	{
	    if ( b.cast_int(0) == 0 )
	    {
                res = 1;
	    }

	    else if ( b.cast_int(0) == 1 )
	    {
                res = a;
	    }

	    else
	    {
		goto basecase;
	    }
	}

	else
	{
	basecase:
            const static gentype resx(strpowname+"(x,y)");

            return resx(a,b);
	}

	return res;
    }

         if ( a.isValStrErr() ) { constructError(a,b,res,strpowname+" ill-defined for string"); }
    else if ( a.isValSet()    ) { constructError(a,b,res,strpowname+" ill-defined for set"   ); }
    else if ( a.isValDgraph() ) { constructError(a,b,res,strpowname+" ill-defined for dgraph"); }
    else if ( a.isValVector() ) { constructError(a,b,res,strpowname+" ill-defined for vector"); }
    else if ( b.isValStrErr() ) { constructError(a,b,res,strpowname+" ill-defined for string exponent"); }
    else if ( b.isValMatrix() ) { constructError(a,b,res,strpowname+" ill-defined for matrix exponent"); }
    else if ( b.isValVector() ) { constructError(a,b,res,strpowname+" ill-defined for vector exponent"); }

    else if ( a.isValMatrix() )
    {
	if ( !( b.isCastableToIntegerWithoutLoss() ) )
	{
            constructError(a,b,res,strpowname+" ill-defined for matrices raised to non-integer exponents");
	}

	else
	{
	    int bint = b.cast_int(0);
	    int absbint = ( bint < 0 ) ? -bint : bint;
	    int i;

	    if ( !bint )
	    {
		Matrix<gentype> thisisanemptymatrixifyouhadntguessedthatalreadyyousillymanyouwhatonearthwereyouthinkingyoureallyneedtochangematrixhtotreatemptymatrixlikeunitmatrix;

                res = thisisanemptymatrixifyouhadntguessedthatalreadyyousillymanyouwhatonearthwereyouthinkingyoureallyneedtochangematrixhtotreatemptymatrixlikeunitmatrix;
	    }

	    else
	    {
		for ( i = 0 ; i < absbint ; i++ )
		{
		    if ( !i )
		    {
			res = a;
		    }

		    else
		    {
			res *= a;
		    }
		}

		if ( bint < 0 )
		{
		    res = inv(res);
		}
	    }
	}
    }

    else
    {
        if ( b.isCastableToIntegerWithoutLoss() )
	{
	    int bint = b.cast_int(0);
	    int absbint = ( bint < 0 ) ? -bint : bint;
	    int i;

	    res = 1;

	    for ( i = 0 ; i < absbint ; i++ )
	    {
		res *= a;
	    }

	    if ( bint < 0 )
	    {
                res = inv(res);
	    }
	}

	else if ( a.isValAnion() || b.isValAnion() )
	{
            res = anionpow(a.cast_anion(0),b.cast_anion(0));
	}

        else if ( ( a.cast_double(0) < 0 ) || ( b.cast_double(0) < 0 ) )
        {
            res = anionpow(a.cast_anion(0), b.cast_anion(0));
        }

        else
        {
            res = pow(a.cast_double(0), b.cast_double(0));
        }
    }

    return res;
}

gentype powintern(const gentype &a, const gentype &b);
gentype powintern(const gentype &a, const gentype &b)
{
    return genpowintern(a,b,"pow",pow);
}

gentype Powintern(const gentype &a, const gentype &b);
gentype Powintern(const gentype &a, const gentype &b)
{
    return genpowintern(a,b,"Pow",Pow);
}

gentype powlintern(const gentype &a, const gentype &b);
gentype powlintern(const gentype &a, const gentype &b)
{
    return genpowintern(a,b,"powl",powl);
}

gentype Powlintern(const gentype &a, const gentype &b);
gentype Powlintern(const gentype &a, const gentype &b)
{
    return genpowintern(a,b,"Powl",Powl);
}

gentype powrintern(const gentype &a, const gentype &b);
gentype powrintern(const gentype &a, const gentype &b)
{
    return genpowintern(a,b,"powr",powr);
}

gentype Powrintern(const gentype &a, const gentype &b);
gentype Powrintern(const gentype &a, const gentype &b)
{
    return genpowintern(a,b,"Powr",Powr);
}

gentype ifthenelse(const gentype &a, const gentype &b, const gentype &c)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
	if ( b == c )
	{
            res = b;
	}

	else
	{
            const static gentype resx("ifthenelse(x,y,z)");

            return resx(a,b,c);
	}

	return res;
    }

    if ( a.isValMatrix() )
    {
	if ( b.isValMatrix() && c.isValMatrix() )
	{
	    if ( ( a.numRows() != b.numRows() ) || ( a.numRows() != c.numRows() ) || ( a.numCols() != b.numCols() ) || ( a.numCols() != c.numCols() ) )
	    {
		constructError(a,b,c,res,"matrix dimensions must agree in ifthenelse elementwise matrix form");
	    }

	    else if ( a.numRows() && a.numCols() )
	    {
		int i,j;

		Matrix<gentype> tempres(a.numRows(),a.numCols());
		Matrix<gentype> tempa(a.cast_matrix(0));
                Matrix<gentype> tempb(b.cast_matrix(0));
                Matrix<gentype> tempc(c.cast_matrix(0));

		for ( i = 0 ; i < a.numRows() ; i++ )
		{
		    for ( j = 0 ; j < a.numCols() ; j++ )
		    {
			tempres("&",i,j) = ifthenelse(tempa(i,j),tempb(i,j),tempc(i,j));
		    }
		}

                res = tempres;
	    }

	    else
	    {
                res = a;
	    }
	}

	else
	{
	    if ( a.numRows() && a.numCols() )
	    {
		int i,j;

                Matrix<gentype> tempres(a.numRows(),a.numCols());
		Matrix<gentype> tempa(a.cast_matrix(0));

		for ( i = 0 ; i < a.numRows() ; i++ )
		{
		    for ( j = 0 ; j < a.numCols() ; j++ )
		    {
			tempres("&",i,j) = ifthenelse(tempa(i,j),b,c);
		    }
		}

                res = tempres;
	    }

	    else
	    {
                res = a;
	    }
	}
    }

    else if ( a.isValVector() )
    {
	if ( b.isValVector() && c.isValVector() )
	{
	    if ( ( a.size() != b.size() ) || ( a.size() != c.size() ) )
	    {
		constructError(a,b,c,res,"vector dimensions must agree in ifthenelse elementwise vector form");
	    }

	    else if ( a.size() )
	    {
		int i;

		Vector<gentype> tempres(a.size());
                Vector<gentype> tempa(a.cast_vector(0));
                Vector<gentype> tempb(b.cast_vector(0));
                Vector<gentype> tempc(c.cast_vector(0));

		for ( i = 0 ; i < a.size() ; i++ )
		{
		    tempres("&",i) = ifthenelse(tempa(i),tempb(i),tempc(i));
		}

                res = tempres;
	    }

	    else
	    {
                res = a;
	    }
	}

	else
	{
	    if ( a.size() )
	    {
		int i;

                Vector<gentype> tempres(a.size());
                Vector<gentype> tempa(a.cast_vector(0));

		for ( i = 0 ; i < a.size() ; i++ )
		{
		    tempres("&",i) = ifthenelse(tempa(i),b,c);
		}

                res = tempres;
	    }

	    else
	    {
                res = a;
	    }
	}
    }

    else if ( !a.isCastableToIntegerWithoutLoss() )
    {
	constructError(a,b,c,res,"ifthenelse requires integer (bool) first argument");
    }

    else
    {
	res = a.cast_int(0) ? b : c;
    }

    return res;
}












// Basic typing functions
//
// Design decision: if it looks like a duck and quacks like a duck...
//
// Ints are also real and anion
// Reals are also anions (and may be ints)
// Anions may be ints and/or reals

gentype &OP_isnull(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isnull(x)");
        x = res(x);
    }

    else
    {
        x = x.isCastableToNullWithoutLoss();
    }

    return x;
}

gentype &OP_isint(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isint(x)");
        x = res(x);
    }

    else
    {
        x = x.isCastableToIntegerWithoutLoss();
    }

    return x;
}

gentype &OP_isreal(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isreal(x)");
        x   = res(x);
    }

    else
    {
        x = x.isCastableToRealWithoutLoss();
    }

    return x;
}

gentype &OP_isanion(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isanion(x)");
        x   = res(x);
    }

    else
    {
        x = x.isCastableToAnionWithoutLoss();
    }

    return x;
}

gentype &OP_isvector(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isvector(x)");
        x   = res(x);
    }

    else
    {
        x = x.isValVector();
    }

    return x;
}

gentype &OP_ismatrix(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("ismatrix(x)");
        x   = res(x);
    }

    else
    {
        x = x.isValMatrix();
    }

    return x;
}

gentype &OP_isset(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("ismatrix(x)");
        x   = res(x);
    }

    else
    {
        x = x.isValSet();
    }

    return x;
}

gentype &OP_isdgraph(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("ismatrix(x)");
        x   = res(x);
    }

    else
    {
        x = x.isValDgraph();
    }

    return x;
}

gentype &OP_isstring(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isstring(x)");
        x   = res(x);
    }

    else
    {
        x = x.isValStrErr();
    }

    return x;
}

gentype &OP_iserror(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("iserror(x)");
        x   = res(x);
    }

    else
    {
        x = x.isValError();
    }

    return x;
}

gentype &OP_isvnan(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isvnan(x)");
        x = res(x);
    }

    else
    {
        x = testisvnan(x);
    }

    return x;
}

gentype &OP_isinf(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isinf(x)");
        x = res(x);
    }

    else
    {
        x = testisinf(x);
    }

    return x;
}

gentype &OP_ispinf(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("ispinf(x)");
        x = res(x);
    }

    else
    {
        x = testispinf(x);
    }

    return x;
}

gentype &OP_isninf(gentype &x)
{
    if ( x.isValEqnDir() )
    {
        const static gentype res("isninf(x)");
        x = res(x);
    }

    else
    {
        x = testisninf(x);
    }

    return x;
}

gentype &OP_size(gentype &a)
{
    if ( a.isValEqnDir() )
    {
        const static gentype res("size(x)");
        a   = res(a);
    }

    else
    {
        a = a.size();
    }

    return a;
}

gentype &OP_numRows(gentype &a)
{
    if ( a.isValEqnDir() )
    {
        const static gentype res("numRows(x)");
        a   = res(a);
    }

    else
    {
        a = a.numRows();
    }

    return a;
}

gentype &OP_numCols(gentype &a)
{
    if ( a.isValEqnDir() )
    {
        const static gentype res("numCols(x)");
        a   = res(a);
    }

    else
    {
        a = a.numCols();
    }

    return a;
}








// Basic anion and vector and construction

gentype eps_comm(const gentype &n, const gentype &q, const gentype &r, const gentype &s)
{
    gentype res;

    if ( n.isValEqnDir() || q.isValEqnDir() || r.isValEqnDir() || s.isValEqnDir() )
    {
        const static gentype resx("eps_comm(x,y,z,v)");
        return resx(n,q,r,s);
    }

         if ( !n.isCastableToIntegerWithoutLoss()  ) { constructError(n,q,r,s,res,"eps_comm must have integer arguments"); }
    else if ( !q.isCastableToIntegerWithoutLoss()  ) { constructError(n,q,r,s,res,"eps_comm must have integer arguments"); }
    else if ( !r.isCastableToIntegerWithoutLoss()  ) { constructError(n,q,r,s,res,"eps_comm must have integer arguments"); }
    else if ( !s.isCastableToIntegerWithoutLoss()  ) { constructError(n,q,r,s,res,"eps_comm must have integer arguments"); }
    else if ( n.cast_int(0) < 0                     ) { constructError(n,q,r,s,res,"eps_comm must have n >= 0"); }
    else if ( q.cast_int(0) < 0                     ) { constructError(n,q,r,s,res,"eps_comm must have q >= 0"); }
    else if ( r.cast_int(0) < 0                     ) { constructError(n,q,r,s,res,"eps_comm must have r >= 0"); }
    else if ( s.cast_int(0) < 0                     ) { constructError(n,q,r,s,res,"eps_comm must have s >= 0"); }
    else if ( q.cast_int(0) > pow(2.0,n.cast_int(0))-1 ) { constructError(n,q,r,s,res,"eps_comm must have q < 2^n-1"); }
    else if ( r.cast_int(0) > pow(2.0,n.cast_int(0))-1 ) { constructError(n,q,r,s,res,"eps_comm must have r < 2^n-1"); }
    else if ( s.cast_int(0) > pow(2.0,n.cast_int(0))-1 ) { constructError(n,q,r,s,res,"eps_comm must have s < 2^n-1"); }
    else
    {
	int nn = n.cast_int(0);
	int qq = q.cast_int(0);
	int rr = r.cast_int(0);
	int ss = s.cast_int(0);

	if ( !qq || !rr || !ss )
	{
            res = zeroint();
	}

	else
	{
	    res = epsilon(nn,qq-1,rr-1,ss-1);
	}
    }

    return res;
}

gentype eps_assoc(const gentype &n, const gentype &q, const gentype &r, const gentype &s, const gentype &t)
{
    gentype res;

    if ( n.isValEqnDir() || q.isValEqnDir() || r.isValEqnDir() || s.isValEqnDir() || t.isValEqnDir() )
    {
        const static gentype resx("eps_assoc(x,y,z,v,w)");
        return resx(n,q,r,s,t);
    }

         if ( !n.isCastableToIntegerWithoutLoss()             ) { constructError(n,q,r,s,res,"eps_assoc must have integer arguments"); }
    else if ( !q.isCastableToIntegerWithoutLoss()             ) { constructError(n,q,r,s,res,"eps_assoc must have integer arguments"); }
    else if ( !r.isCastableToIntegerWithoutLoss()             ) { constructError(n,q,r,s,res,"eps_assoc must have integer arguments"); }
    else if ( !s.isCastableToIntegerWithoutLoss()             ) { constructError(n,q,r,s,res,"eps_assoc must have integer arguments"); }
    else if ( !t.isCastableToIntegerWithoutLoss()             ) { constructError(n,q,r,s,res,"eps_assoc must have integer arguments"); }
    else if ( n.cast_int(0) < 0                                ) { constructError(n,q,r,s,res,"eps_assoc must have n >= 0"); }
    else if ( q.cast_int(0) < 0                                ) { constructError(n,q,r,s,res,"eps_assoc must have q >= 0"); }
    else if ( r.cast_int(0) < 0                                ) { constructError(n,q,r,s,res,"eps_assoc must have r >= 0"); }
    else if ( s.cast_int(0) < 0                                ) { constructError(n,q,r,s,res,"eps_assoc must have s >= 0"); }
    else if ( t.cast_int(0) < 0                                ) { constructError(n,q,r,s,res,"eps_assoc must have t >= 0"); }
    else if ( q.cast_int(0) > pow(2.0,(double) n.cast_int(0))-1 ) { constructError(n,q,r,s,res,"eps_assoc must have q < 2^n-1"); }
    else if ( r.cast_int(0) > pow(2.0,(double) n.cast_int(0))-1 ) { constructError(n,q,r,s,res,"eps_assoc must have r < 2^n-1"); }
    else if ( s.cast_int(0) > pow(2.0,(double) n.cast_int(0))-1 ) { constructError(n,q,r,s,res,"eps_assoc must have s < 2^n-1"); }
    else if ( t.cast_int(0) > pow(2.0,(double) n.cast_int(0))-1 ) { constructError(n,q,r,s,res,"eps_assoc must have t < 2^n-1"); }
    else
    {
	int nn = n.cast_int(0);
	int qq = q.cast_int(0);
	int rr = r.cast_int(0);
	int ss = s.cast_int(0);
	int tt = t.cast_int(0);

	if ( !qq || !rr || !ss || !tt )
	{
            res = zeroint();
	}

	else
	{
	    res = epsilon(nn,qq-1,rr-1,ss-1,tt=1);
	}
    }

    return res;
}

gentype im_complex(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("im_quat(x)");
        return resx(x);
    }

         if ( !x.isCastableToIntegerWithoutLoss() ) { constructError(x,res,"im_complex must have integer arguments"); }
    else if ( x.cast_int(0) < 0                    ) { constructError(x,res,"im_complex argument must be 0 or 1"); }
    else if ( x.cast_int(0) > 1                    ) { constructError(x,res,"im_complex argument must be 0 or 1"); }
    else
    {
	d_anion tres(1);
        tres("&",x.cast_int(0)) = 1.0;
        res = tres;
    }

    return res;
}

gentype im_quat(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("im_quat(x)");
        return resx(x);
    }

         if ( !x.isCastableToIntegerWithoutLoss() ) { constructError(x,res,"im_quat must have integer arguments"); }
    else if ( x.cast_int(0) < 0                    ) { constructError(x,res,"im_quat argument must be 0,1,2,3"); }
    else if ( x.cast_int(0) > 3                    ) { constructError(x,res,"im_quat argument must be 0,1,2,3"); }
    else
    {
	d_anion tres(2);
        tres("&",x.cast_int(0)) = 1.0;
        res = tres;
    }

    return res;
}

gentype im_octo(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("im_octo(x)");
        return resx(x);
    }

         if ( !x.isCastableToIntegerWithoutLoss() ) { constructError(x,res,"im_octo must have integer arguments"); }
    else if ( x.cast_int(0) < 0                    ) { constructError(x,res,"im_octo argument must be 0,1,2,3,4,5,6,7"); }
    else if ( x.cast_int(0) > 7                    ) { constructError(x,res,"im_octo argument must be 0,1,2,3,4,5,6,7"); }
    else
    {
	d_anion tres(3);
        tres("&",x.cast_int(0)) = 1.0;
	res = tres;
    }

    return res;
}

gentype im_anion(const gentype &x, const gentype &y)
{
    gentype res;

    if ( x.isValEqnDir() || y.isValEqnDir() )
    {
        const static gentype resx("im_anion(x,y)");
        return resx(x,y);
    }

         if ( !y.isCastableToIntegerWithoutLoss()              ) { constructError(x,y,res,"im_anion must have integer arguments"); }
    else if ( !x.isCastableToIntegerWithoutLoss()              ) { constructError(x,y,res,"im_anion must have integer arguments"); }
    else if ( y.cast_int(0) < 0                                 ) { constructError(x,y,res,"im_anion argument must be 0,1,...,2^n-1"); }
    else if ( y.cast_int(0) > pow(2.0,(double) x.cast_int(0))-1  ) { constructError(x,y,res,"im_anion argument must be 0,1,...,2^n-1"); }
    else
    {
	d_anion tres(x.cast_int(0));
        tres("&",y.cast_int(0)) = 1.0;
        res = tres;
   }

    return res;
}

gentype Im_complex(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("im_quat(x)");
        return resx(x);
    }

         if ( !x.isCastableToIntegerWithoutLoss() ) { constructError(x,res,"im_complex must have integer arguments"); }
    else if ( x.cast_int(0) < 0                    ) { constructError(x,res,"im_complex argument must be 0 or 1"); }
    else if ( x.cast_int(0) > 1                    ) { constructError(x,res,"im_complex argument must be 0 or 1"); }
    else
    {
	d_anion tres(1);
        tres("&",x.cast_int(0)) = x.cast_int(0) ? -1.0 : 1.0;
        res = tres;
    }

    return res;
}

gentype Im_quat(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("im_quat(x)");
        return resx(x);
    }

         if ( !x.isCastableToIntegerWithoutLoss() ) { constructError(x,res,"im_quat must have integer arguments"); }
    else if ( x.cast_int(0) < 0                    ) { constructError(x,res,"im_quat argument must be 0,1,2,3"); }
    else if ( x.cast_int(0) > 3                    ) { constructError(x,res,"im_quat argument must be 0,1,2,3"); }
    else
    {
	d_anion tres(2);
        tres("&",x.cast_int(0)) = x.cast_int(0) ? -1.0 : 1.0;
        res = tres;
    }

    return res;
}

gentype Im_octo(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("im_octo(x)");
        return resx(x);
    }

         if ( !x.isCastableToIntegerWithoutLoss() ) { constructError(x,res,"im_octo must have integer arguments"); }
    else if ( x.cast_int(0) < 0                    ) { constructError(x,res,"im_octo argument must be 0,1,2,3,4,5,6,7"); }
    else if ( x.cast_int(0) > 7                    ) { constructError(x,res,"im_octo argument must be 0,1,2,3,4,5,6,7"); }
    else
    {
	d_anion tres(3);
        tres("&",x.cast_int(0)) = x.cast_int(0) ? -1.0 : 1.0;
	res = tres;
    }

    return res;
}

gentype Im_anion(const gentype &x, const gentype &y)
{
    gentype res;

    if ( x.isValEqnDir() || y.isValEqnDir() )
    {
        const static gentype resx("im_anion(x,y)");
        return resx(x,y);
    }

         if ( !y.isCastableToIntegerWithoutLoss()              ) { constructError(x,y,res,"im_anion must have integer arguments"); }
    else if ( !x.isCastableToIntegerWithoutLoss()              ) { constructError(x,y,res,"im_anion must have integer arguments"); }
    else if ( y.cast_int(0) < 0                                 ) { constructError(x,y,res,"im_anion argument must be 0,1,...,2^n-1"); }
    else if ( y.cast_int(0) > pow(2.0,(double) x.cast_int(0))-1  ) { constructError(x,y,res,"im_anion argument must be 0,1,...,2^n-1"); }
    else
    {
	d_anion tres(x.cast_int(0));
        tres("&",x.cast_int(0)) = x.cast_int(0) ? -1.0 : 1.0;
        res = tres;
   }

    return res;
}


gentype vect_const(const gentype &x, const gentype &y)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("vect_const(x,y)");
        return resx(x,y);
    }

         if ( !x.isCastableToIntegerWithoutLoss() ) { constructError(x,y,res,"vect_const must have integer vector size"); }
    else if ( x.cast_int(0) < 0                    ) { constructError(x,y,res,"vect_const must have non-negative size"); }
    else
    {
	Vector<gentype> tres(x.cast_int(0));
	tres = y;
        res = tres;
    }

    return res;
}

gentype vect_unit(const gentype &x, const gentype &y)
{
    gentype res;

    if ( x.isValEqnDir() || y.isValEqnDir() )
    {
        const static gentype resx("vect_unit(x,y)");
        return resx(x,y);
    }

         if ( !x.isCastableToIntegerWithoutLoss()   ) { constructError(x,y,res,"vect_unit must have integer arguments"); }
    else if ( !y.isCastableToIntegerWithoutLoss()   ) { constructError(x,y,res,"vect_unit must have integer arguments"); }
    else if ( x.cast_int(0) < 0                      ) { constructError(x,y,res,"vect_unit must have non-negative size"); }
    else if ( y.cast_int(0) < 0                      ) { constructError(x,y,res,"vect_unit must have non-negative index"); }
    else if ( y.cast_int(0) >= x.cast_int(0)          ) { constructError(x,y,res,"vect_unit must have index less than vector size"); }
    else
    {
	Vector<gentype> tres(x.cast_int(0));
        const static gentype tempo(zeroint());
	tres = tempo;
        tres("&",y.cast_int(0)) = 1;
        res = tres;
    }

    return res;
}

gentype ivect(const gentype &x, const gentype &y, const gentype &z)
{
    gentype res;

    if ( x.isValEqnDir() || y.isValEqnDir() || z.isValEqnDir() )
    {
        const static gentype resx("ivect(x,y,z)");
        return resx(x,y,z);
    }

    if ( !(x.isCastableToRealWithoutLoss()) || !(y.isCastableToRealWithoutLoss()) || !(z.isCastableToRealWithoutLoss()) )
    {
        constructError(x,y,z,res,"ivect arguments must be ordered scalars.");

        return res;
    }

    if ( x.isCastableToIntegerWithoutLoss() && y.isCastableToIntegerWithoutLoss() && z.isCastableToIntegerWithoutLoss() )
    {
        int xx = x.cast_int(0);
        int yy = y.cast_int(0);
        int zz = z.cast_int(0);

        Vector<gentype> vres;

        while ( xx <= zz )
        {
            vres.add(vres.size());

            vres("&",vres.size()-1) = xx;

            xx += yy;
        }

        res = vres;
    }

    else
    {
        double xx = x.cast_double(0);
        double yy = y.cast_double(0);
        double zz = z.cast_double(0);

        Vector<gentype> vres;

        while ( xx <= zz )
        {
            vres.add(vres.size());

            vres("&",vres.size()-1) = xx;

            xx += yy;
        }

        res = vres;
    }

    return res;
}

gentype cayleyDickson(const gentype &x, const gentype &y)
{
    gentype res;

    if ( x.isValEqnDir() || y.isValEqnDir() )
    {
        const static gentype resx("cayleyDickson(x,y)");
        return resx(x,y);
    }

	 if ( x.isValStrErr() || y.isValStrErr() ) { constructError(x,y,res,"CayleyDickson not defined for strings."); }
    else if ( x.isValMatrix() || y.isValMatrix() ) { constructError(x,y,res,"CayleyDickson not defined for matrices."); }
    else if ( x.isValVector() || y.isValVector() ) { constructError(x,y,res,"CayleyDickson not defined for vectors."); }
    else if ( x.isValSet()    || y.isValSet()    ) { constructError(x,y,res,"CayleyDickson not defined for sets."); }
    else if ( x.isValDgraph() || y.isValDgraph() ) { constructError(x,y,res,"CayleyDickson not defined for dgraphs."); }
    else                                           { d_anion temp(x.cast_anion(0),y.cast_anion(0)); res = temp; }

    return res;
}

gentype CayleyDickson(const gentype &x, const gentype &y)
{
    gentype res;

    if ( x.isValEqnDir() || y.isValEqnDir() )
    {
        const static gentype resx("CayleyDickson(x,y)");
        return resx(x,y);
    }

	 if ( x.isValStrErr() || y.isValStrErr() ) { constructError(x,y,res,"CayleyDickson not defined for strings."); }
    else if ( x.isValMatrix() || y.isValMatrix() ) { constructError(x,y,res,"CayleyDickson not defined for matrices."); }
    else if ( x.isValVector() || y.isValVector() ) { constructError(x,y,res,"CayleyDickson not defined for vectors."); }
    else if ( x.isValSet()    || y.isValSet()    ) { constructError(x,y,res,"CayleyDickson not defined for sets."); }
    else if ( x.isValDgraph() || y.isValDgraph() ) { constructError(x,y,res,"CayleyDickson not defined for dgraphs."); }
    else                                           { d_anion temp(conj(x.cast_anion(0)),-y.cast_anion(0)); res = temp; }

    return res;
}





gentype eye(const gentype &i, const gentype &j)
{
    gentype res;

    if ( i.isValEqn() || j.isValEqn() )
    {
        const static gentype resx("eye(x,y)");
        return resx(i,j);
    }

    if ( i.isCastableToIntegerWithoutLoss() && j.isCastableToIntegerWithoutLoss() && ( (int) i >= 0 ) && ( (int) j >= 0 ) )
    {
        res.force_matrix((int) i, (int) j);

        res.dir_matrix() = zerogentype();

        int ii;
        int ij = ( ( (int) i ) < ( (int) j ) ) ? ( (int) i ) : ( (int) j );

        for ( ii = 0 ; ii < ij ; ii++ )
        {
            res.dir_matrix()("&",ii,ii) = onedblgentype();
        }
    }

    else
    {
        constructError(i,j,res,"eye size must be non-negative integers");
    }

    return res;
}









// Permutations, combinations and other integer stuff

gentype kronDelta(const gentype &i, const gentype &j)
{
    gentype res;

    if ( i.isValEqn() || j.isValEqn() )
    {
        const static gentype resx("kronDelta(x,y)");
        return resx(i,j);
    }

    return eq(i,j);
}

gentype diracDelta(const gentype &x)
{
    gentype res;

    if ( x.isValEqn() )
    {
        const static gentype resx("diracDelta(x)");
        return resx(x);
    }

    return ifthenelse(eq(x,zerointgentype()),pinf(),res);
}

gentype ekronDelta(const gentype &i, const gentype &j)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() )
    {
        const static gentype resx("ekronDelta(x,y)");
        return resx(i,j);
    }

    return eeq(i,j);
}

gentype ediracDelta(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("ediracDelta(x)");
        return resx(x);
    }

    return ifthenelse(eeq(x,zerointgentype()),pinf(),res);
}

gentype perm(const gentype &i, const gentype &j)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() )
    {
        const static gentype resx("perm(x,y)");
        return resx(i,j);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() )
    {
	if ( i.isValMatrix() )
	{
	    if ( j.isValMatrix() )
	    {
		if ( ( i.numRows() != j.numRows() ) || ( i.numCols() != j.numCols() ) )
		{
		    constructError(i,j,res,"matrix dimensions must agree in perm elementwise matrix form");
		}

		else if ( i.numRows() && i.numCols() )
		{
		    int ii,jj;

		    Matrix<gentype> tempres(i.numRows(),i.numCols());
		    Matrix<gentype> tempi(i.cast_matrix(0));
		    Matrix<gentype> tempj(j.cast_matrix(0));

		    for ( ii = 0 ; ii < i.numRows() ; ii++ )
		    {
			for ( jj = 0 ; jj < i.numCols() ; jj++ )
			{
			    tempres("&",ii,jj) = perm(tempi(ii,jj),tempj(ii,jj));
			}
		    }

		    res = tempres;
		}

		else
		{
		    res = i;
		}
	    }

	    else
	    {
		if ( i.numRows() && i.numCols() )
		{
		    int ii,jj;

		    Matrix<gentype> tempres(i.numRows(),i.numCols());
		    Matrix<gentype> tempi(i.cast_matrix(0));

		    for ( ii = 0 ; ii < i.numRows() ; ii++ )
		    {
			for ( jj = 0 ; jj < i.numCols() ; jj++ )
			{
			    tempres("&",ii,jj) = perm(tempi(ii,jj),j);
			}
		    }

		    res = tempres;
		}

		else
		{
		    res = i;
		}
	    }
	}

	else if ( i.isValVector() )
	{
	    if ( j.isValVector() )
	    {
		if ( i.size() != j.size() )
		{
		    constructError(i,j,res,"vector dimensions must agree in perm elementwise vector form");
		}

		else if ( i.size() )
		{
		    int ii;

		    Vector<gentype> tempres(i.size());
		    Vector<gentype> tempi(i.cast_vector(0));
		    Vector<gentype> tempj(j.cast_vector(0));

		    for ( ii = 0 ; ii < i.size() ; ii++ )
		    {
			tempres("&",ii) = perm(tempi(ii),tempj(ii));
		    }

		    res = tempres;
		}

		else
		{
		    res = i;
		}
	    }

	    else
	    {
		if ( i.size() )
		{
		    int ii;

		    Vector<gentype> tempres(i.size());
		    Vector<gentype> tempi(i.cast_vector(0));

		    for ( ii = 0 ; ii < i.size() ; ii++ )
		    {
			tempres("&",ii) = perm(tempi(ii),j);
		    }

		    res = tempres;
		}

		else
		{
		    res = i;
		}
	    }
	}

	else
	{
	    constructError(i,j,res,"perm only defined for integers");
	}
    }

    else
    {
	int ii = i.cast_int(0);
	int jj = j.cast_int(0);
	int kk;
        int result;

	if ( ( ii >= 0 ) && ( jj >= 0 ) && ( ii >= jj ) )
	{
	    result = 1;

	    if ( ii > jj )
	    {
		for ( kk = jj+1 ; kk <= ii ; kk++ )
		{
                    result *= kk;
		}
	    }

	    res = result;
	}

	else
	{
            constructError(i,j,res,"perm assumes i >= j >= 0");
	}
    }

    return res;
}

gentype comb(const gentype &i, const gentype &j)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() )
    {
        const static gentype resx("comb(x,y)");
        return resx(i,j);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() )
    {
	if ( i.isValMatrix() )
	{
	    if ( j.isValMatrix() )
	    {
		if ( ( i.numRows() != j.numRows() ) || ( i.numCols() != j.numCols() ) )
		{
		    constructError(i,j,res,"matrix dimensions must agree in comb elementwise matrix form");
		}

		else if ( i.numRows() && i.numCols() )
		{
		    int ii,jj;

		    Matrix<gentype> tempres(i.numRows(),i.numCols());
		    Matrix<gentype> tempi(i.cast_matrix(0));
		    Matrix<gentype> tempj(j.cast_matrix(0));

		    for ( ii = 0 ; ii < i.numRows() ; ii++ )
		    {
			for ( jj = 0 ; jj < i.numCols() ; jj++ )
			{
			    tempres("&",ii,jj) = comb(tempi(ii,jj),tempj(ii,jj));
			}
		    }

		    res = tempres;
		}

		else
		{
		    res = i;
		}
	    }

	    else
	    {
		if ( i.numRows() && i.numCols() )
		{
		    int ii,jj;

		    Matrix<gentype> tempres(i.numRows(),i.numCols());
		    Matrix<gentype> tempi(i.cast_matrix(0));

		    for ( ii = 0 ; ii < i.numRows() ; ii++ )
		    {
			for ( jj = 0 ; jj < i.numCols() ; jj++ )
			{
			    tempres("&",ii,jj) = comb(tempi(ii,jj),j);
			}
		    }

		    res = tempres;
		}

		else
		{
		    res = i;
		}
	    }
	}

	else if ( i.isValVector() )
	{
	    if ( j.isValVector() )
	    {
		if ( i.size() != j.size() )
		{
		    constructError(i,j,res,"vector dimensions must agree in comb elementwise vector form");
		}

		else if ( i.size() )
		{
		    int ii;

		    Vector<gentype> tempres(i.size());
		    Vector<gentype> tempi(i.cast_vector(0));
		    Vector<gentype> tempj(j.cast_vector(0));

		    for ( ii = 0 ; ii < i.size() ; ii++ )
		    {
			tempres("&",ii) = comb(tempi(ii),tempj(ii));
		    }

		    res = tempres;
		}

		else
		{
		    res = i;
		}
	    }

	    else
	    {
		if ( i.size() )
		{
		    int ii;

		    Vector<gentype> tempres(i.size());
		    Vector<gentype> tempi(i.cast_vector(0));

		    for ( ii = 0 ; ii < i.size() ; ii++ )
		    {
			tempres("&",ii) = comb(tempi(ii),j);
		    }

		    res = tempres;
		}

		else
		{
		    res = i;
		}
	    }
	}

	else
	{
	    constructError(i,j,res,"comb only defined for integers");
	}
    }

    else
    {
	int n = i.cast_int(0);
	int r = j.cast_int(0);

	if ( ( n >= 0 ) && ( r >= 0 ) && ( n >= r ) )
	{
            res = xnCr(n,r);
	}

	else
	{
	    constructError(i,j,res,"comb assumes n >= r >= 0");
	}
    }

    return res;
}

gentype fact(const gentype &i)
{
    gentype res;

    if ( i.isValEqnDir() )
    {
        const static gentype resx("fact(x)");
        return resx(i);
    }

    if ( i.isValMatrix() )
    {
        Matrix<gentype> temp(i.cast_matrix(0));
        temp.applyon(fact);
        res = temp;
    }

    else if ( i.isValVector() )
    {
        Vector<gentype> temp(i.cast_vector(0));
        temp.applyon(gamma);
        res = temp;
    }

    else if ( !i.isCastableToIntegerWithoutLoss() )
    {
        res = gamma(add(i,oneintgentype()));
    }

    else if ( i.cast_int(0) < 0 )
    {
        res = gamma(add(i,oneintgentype()));
    }

    else if ( i.cast_int(0) > MAXINTFACT )
    {
        res = gamma(add(i,oneintgentype()));
    }

    else
    {
        res = xnfact(i.cast_int(0));
    }

    return res;
}










// Given a vector of functions - eg:
//
// [ x y ]
//
// we can't just use the standard vectorial norm as we
// can't cast x to double!  So instead we need the following.

gentype abs1vec(const Vector<gentype> &a);
gentype abs2vec(const Vector<gentype> &a);
gentype abspvec(const Vector<gentype> &a, double p);
gentype absinfvec(const Vector<gentype> &a);

gentype norm1vec(const Vector<gentype> &a);
gentype norm2vec(const Vector<gentype> &a);
gentype normpvec(const Vector<gentype> &a, double p);

gentype abs1vec(const Vector<gentype> &a)
{
    int i;
    gentype result;

    if ( a.infsize() )
    {
        result = norm1(a);
    }

    else if ( a.size() )
    {
	result = norm1(a(0));

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += norm1(a(i));
	    }
	}
    }

    return result;
}

gentype abs2vec(const Vector<gentype> &a)
{
    int i;
    gentype result;

    if ( a.infsize() )
    {
        result = norm2(a);
    }

    else if ( a.size() )
    {
	result = norm2(a(0));

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += norm2(a(i));
	    }
	}
    }

    return sqrt(result);
}

gentype abspvec(const Vector<gentype> &a, double p)
{
    int i;
    gentype result;

    if ( a.infsize() )
    {
        result = normp(a,p);
    }

    else if ( a.size() )
    {
	result = normp(a(0),p);

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += normp(a(i),p);
	    }
	}
    }

    gentype invp(1/p);

    return pow(result,invp);
}

gentype absinfvec(const Vector<gentype> &a)
{
    int i;
    gentype result(a);
    gentype temp;
    double dres = 0.0;
    double dtemp = 0.0;

    if ( a.infsize() )
    {
        result = absinf(a);
    }

    else if ( a.size() )
    {
        for ( i = 0 ; i < a.size() ; i++ )
        {
            temp = absinf(a(i));

            if ( temp.isValEqn() )
            {
                return result;
            }

            dtemp = (double) temp;

            dres = ( dtemp > dres ) ? dtemp : dres;
	}

        result = dres;
    }

    else
    {
        result = dres;
    }

    return result;
}

gentype norm1vec(const Vector<gentype> &a)
{
    int i;
    gentype result;

    if ( a.infsize() )
    {
        result = norm1(a);
    }

    else if ( a.size() )
    {
	result = norm1(a(0));

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += norm1(a(i));
	    }
	}
    }

    return result;
}

gentype norm2vec(const Vector<gentype> &a)
{
    int i;
    gentype result;

    if ( a.infsize() )
    {
        result = norm2(a);
    }

    else if ( a.size() )
    {
	result = norm2(a(0));

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += norm2(a(i));
	    }
	}
    }

    return result;
}

gentype normpvec(const Vector<gentype> &a, double p)
{
    int i;
    gentype result;

    if ( a.infsize() )
    {
        result = normp(a,p);
    }

    else if ( a.size() )
    {
	result = normp(a(0),p);

	if ( a.size() > 1 )
	{
	    for ( i = 1 ; i < a.size() ; i++ )
	    {
		result += normp(a(i),p);
	    }
	}
    }

    return result;
}



// Non-elementwise maths functions

gentype abs1(const gentype &a)
{
    gentype res;

    if ( a.scalarfn_isscalarfn() && !a.isValVector() )
    {
        return norm1(a);
    }

    else if ( a.isValEqnDir() )
    {
        const static gentype resx("abs1(x)");
        return resx(a);
    }

    else if ( a.isValStrErr()  ) { constructError(a,res,"String norm not implemented"); }
    else if ( a.isValSet()     ) { res = abs1(a.cast_set(0)); }
    else if ( a.isValDgraph()  ) { res = abs1(a.cast_dgraph(0)); }
    else if ( a.isValMatrix()  ) { constructError(a,res,"Matrix norm not implemented"); }
    else if ( a.isValVector()  ) { res = abs1vec(a.cast_vector(0)); }
    else if ( a.isValAnion()   ) { res = abs1(a.cast_anion(0)); }
    else if ( a.isValReal()    ) { res = abs1(a.cast_double(0)); }
    else if ( a.isValInteger() ) { res = abs1(a.cast_int(0)); }
    else                         { res = a; }

    return res;
}

gentype abs2(const gentype &a)
{
    gentype res;

    if ( a.scalarfn_isscalarfn() && !a.isValVector() )
    {
        return sqrt(norm2(a));
    }

    else if ( a.isValEqnDir() )
    {
        const static gentype resx("abs2(x)");
        return resx(a);
    }

    else if ( a.isValStrErr()  ) { constructError(a,res,"String norm not implemented"); }
    else if ( a.isValSet()     ) { res = abs2(a.cast_set(0)); }
    else if ( a.isValDgraph()  ) { res = abs2(a.cast_dgraph(0)); }
    else if ( a.isValMatrix()  ) { constructError(a,res,"Matrix norm not implemented"); }
    else if ( a.isValVector()  ) { res = abs2vec(a.cast_vector(0)); }
    else if ( a.isValAnion()   ) { res = abs2(a.cast_anion(0)); }
    else if ( a.isValReal()    ) { res = abs2(a.cast_double(0)); }
    else if ( a.isValInteger() ) { res = abs2(a.cast_int(0)); }
    else                         { res = a; }

    return res;
}

gentype absp(const gentype &a, const gentype &q)
{
    gentype res;

    if ( a.scalarfn_isscalarfn() && !a.isValVector() && q.isCastableToRealWithoutLoss() )
    {
        return pow(normp(a,q),inv(q));
    }

    else if ( a.isValEqnDir() || q.isValEqnDir() )
    {
        const static gentype resx("absp(x,y)");
        return resx(a,q);
    }

    else if ( !q.isCastableToRealWithoutLoss() )
    {
        constructError(a,q,res,"p must be real or integer for p-norm");
    }

    else if ( a.isValStrErr()  ) { constructError(a,q,res,"String norm not implemented"); }
    else if ( a.isValSet()     ) { res = absp(a.cast_set(0),q.cast_double(0)); }
    else if ( a.isValDgraph()  ) { res = absp(a.cast_dgraph(0),q.cast_double(0)); }
    else if ( a.isValMatrix()  ) { constructError(a,q,res,"Matrix norm not implemented"); }
    else if ( a.isValVector()  ) { res = abspvec(a.cast_vector(0),q.cast_double(0)); }
    else if ( a.isValAnion()   ) { res = absp(a.cast_anion(0),q.cast_double(0)); }
    else if ( a.isValReal()    ) { res = absp(a.cast_double(0),q.cast_double(0)); }
    else if ( a.isValInteger() ) { res = absp(a.cast_int(0),q.cast_double(0)); }
    else                         { res = a; }

    return res;
}

gentype absinf(const gentype &a)
{
    gentype res;

    if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 0 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        res.zero();
        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();

        int i;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        double temp = 0;
        double restemp = 0;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            if ( ( temp = (double) norm1(a(xa)) ) > restemp )
            {
                restemp = temp;
            }
        }

        res = restemp;
        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.scalarfn_isscalarfn() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numvar = a.scalarfn_i().size();
        int numpts = a.scalarfn_numpts();
        int numtot = (int) pow(numpts,numvar);

        int i,j;
        Vector<int> k(numvar);

        k = zeroint();

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        double temp = 0;
        double restemp = 0;

        for ( i = 0 ; i < numtot ; i++ )
        {
            for ( j = 0 ; j < numvar ; j++ )
            {
                xa("&",a.scalarfn_i()(j))("&",a.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
            }

            if ( ( temp = (double) abs1(a(xa)) ) > restemp )
            {
                restemp = temp;
            }

            for ( j = 0 ; j < numvar ; j++ )
            {
                k("&",j)++;

                if ( k(j) >= numpts )
                {
                    k("&",j) = 0;
                }

                else
                {
                    break;
                }
            }
        }

        res = restemp;
        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.isValEqnDir() )
    {
        const static gentype resx("absinf(x)");
        return resx(a);
    }

    else if ( a.isValStrErr()  ) { constructError(a,res,"String norm not implemented"); }
    else if ( a.isValMatrix()  ) { constructError(a,res,"Matrix norm not implemented"); }
    else if ( a.isValSet()     ) { res = absinf(a.cast_set(0)); }
    else if ( a.isValDgraph()  ) { res = absinf(a.cast_dgraph(0)); }
    else if ( a.isValVector()  ) { res = absinfvec(a.cast_vector(0)); }
    else if ( a.isValAnion()   ) { res = absinf(a.cast_anion(0)); }
    else if ( a.isValReal()    ) { res = absinf(a.cast_double(0)); }
    else if ( a.isValInteger() ) { res = absinf(a.cast_int(0)); }
    else                         { res = a; }

    return res;
}

gentype norm1(const gentype &a)
{
    gentype res;

    if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 0 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        res.zero();
        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 1 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();

        int i;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa,bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(xa);
            aa.finalise();

            if ( aa.isValEqnDir() )
            {
                const static gentype resx("norm1(x)");
                bb = resx(aa);
            }

            else if ( aa.isValStrErr()  ) { constructError(aa,bb,"String norm not implemented"); }
            else if ( aa.isValSet()     ) { bb = norm1(aa.cast_set(0)); }
            else if ( aa.isValDgraph()  ) { bb = norm1(aa.cast_dgraph(0)); }
            else if ( aa.isValMatrix()  ) { constructError(aa,bb,"Matrix norm not implemented"); }
            else if ( aa.isValVector()  ) { bb = norm1vec(aa.cast_vector(0)); }
            else if ( aa.isValAnion()   ) { bb = norm1(aa.cast_anion(0)); }
            else if ( aa.isValReal()    ) { bb = norm1(aa.cast_double(0)); }
            else if ( aa.isValInteger() ) { bb = norm1(aa.cast_int(0)); }
            else                          { bb = aa; }

            bb /= ((double) numpts);

            res += bb;
        }

        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.scalarfn_isscalarfn() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numvar = a.scalarfn_i().size();
        int numpts = a.scalarfn_numpts();
        int numtot = (int) pow(numpts,numvar);

        int i,j;
        Vector<int> k(numvar);

        k = zeroint();

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa,bb;

        for ( i = 0 ; i < numtot ; i++ )
        {
            for ( j = 0 ; j < numvar ; j++ )
            {
                xa("&",a.scalarfn_i()(j))("&",a.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
            }

            aa = a(xa);
            aa.finalise();

            if ( aa.isValEqnDir() )
            {
                const static gentype resx("norm1(x)");
                bb = resx(aa);
            }

            else if ( aa.isValStrErr()  ) { constructError(aa,bb,"String norm not implemented"); }
            else if ( aa.isValSet()     ) { bb = norm1(aa.cast_set(0)); }
            else if ( aa.isValDgraph()  ) { bb = norm1(aa.cast_dgraph(0)); }
            else if ( aa.isValMatrix()  ) { constructError(aa,bb,"Matrix norm not implemented"); }
            else if ( aa.isValVector()  ) { bb = norm1vec(aa.cast_vector(0)); }
            else if ( aa.isValAnion()   ) { bb = norm1(aa.cast_anion(0)); }
            else if ( aa.isValReal()    ) { bb = norm1(aa.cast_double(0)); }
            else if ( aa.isValInteger() ) { bb = norm1(aa.cast_int(0)); }
            else                          { bb = aa; }

            bb /= ((double) numtot);

            res += bb;

            for ( j = 0 ; j < numvar ; j++ )
            {
                k("&",j)++;

                if ( k(j) >= numpts )
                {
                    k("&",j) = 0;
                }

                else
                {
                    break;
                }
            }
        }

        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.isValEqnDir() )
    {
        const static gentype resx("norm1(x)");
        return resx(a);
    }

    else if ( a.isValStrErr()  ) { constructError(a,res,"String norm not implemented"); }
    else if ( a.isValSet()     ) { res = norm1(a.cast_set(0)); }
    else if ( a.isValDgraph()  ) { res = norm1(a.cast_dgraph(0)); }
    else if ( a.isValMatrix()  ) { constructError(a,res,"Matrix norm not implemented"); }
    else if ( a.isValVector()  ) { res = norm1vec(a.cast_vector(0)); }
    else if ( a.isValAnion()   ) { res = norm1(a.cast_anion(0)); }
    else if ( a.isValReal()    ) { res = norm1(a.cast_double(0)); }
    else if ( a.isValInteger() ) { res = norm1(a.cast_int(0)); }
    else                         { res = a; }

    return res;
}

gentype norm2(const gentype &a)
{
    gentype res;

    if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 0 ) )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        res.zero();
        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 1 ) )
    {
//errstream() << "phantomxg 0\n";
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();

//errstream() << "phantomxg 1: " << numpts << "\n";
        int i;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa,bb;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);
//errstream() << "phantomxg 2(" << i << "): " << xa << "\n";

            aa = a(xa);
//errstream() << "phantomxg 3(" << i << "): " << aa << "\n";
            aa.finalise();
//errstream() << "phantomxg 4(" << i << "): " << aa << "\n";

            if ( aa.isValEqnDir() )
            {
                const static gentype resx("norm2(x)");
                bb = resx(aa);
            }

            else if ( aa.isValStrErr()  ) { constructError(aa,bb,"String norm not implemented"); }
            else if ( aa.isValSet()     ) { bb = norm2(aa.cast_set(0)); }
            else if ( aa.isValDgraph()  ) { bb = norm2(aa.cast_dgraph(0)); }
            else if ( aa.isValMatrix()  ) { constructError(aa,bb,"Matrix norm2 not implemented"); }
            else if ( aa.isValVector()  ) { bb = norm2vec(aa.cast_vector(0)); }
            else if ( aa.isValAnion()   ) { bb = norm2(aa.cast_anion(0)); }
            else if ( aa.isValReal()    ) { bb = norm2(aa.cast_double(0)); }
            else if ( aa.isValInteger() ) { bb = norm2(aa.cast_int(0)); }
            else                          { bb = aa; }

            bb /= ((double) numpts);
//errstream() << "phantomxg 5(" << i << "): " << bb << "\n";

            res += bb;
//errstream() << "phantomxg 6(" << i << "): " << res << "\n";
        }

//errstream() << "phantomxg 7: " << res << "\n";
        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.scalarfn_isscalarfn() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numvar = a.scalarfn_i().size();
        int numpts = a.scalarfn_numpts();
        int numtot = (int) pow(numpts,numvar);

        int i,j;
        Vector<int> k(numvar);

        k = zeroint();

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa,bb;

        for ( i = 0 ; i < numtot ; i++ )
        {
            for ( j = 0 ; j < numvar ; j++ )
            {
                xa("&",a.scalarfn_i()(j))("&",a.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
            }

            aa = a(xa);
            aa.finalise();

            if ( aa.isValEqnDir() )
            {
                const static gentype resx("norm2(x)");
                bb = resx(aa);
            }

            else if ( aa.isValStrErr()  ) { constructError(aa,bb,"String norm2 not implemented"); }
            else if ( aa.isValSet()     ) { bb = norm2(aa.cast_set(0)); }
            else if ( aa.isValDgraph()  ) { bb = norm2(aa.cast_dgraph(0)); }
            else if ( aa.isValMatrix()  ) { constructError(aa,bb,"Matrix norm2 not implemented"); }
            else if ( aa.isValVector()  ) { bb = norm2vec(aa.cast_vector(0)); }
            else if ( aa.isValAnion()   ) { bb = norm2(aa.cast_anion(0)); }
            else if ( aa.isValReal()    ) { bb = norm2(aa.cast_double(0)); }
            else if ( aa.isValInteger() ) { bb = norm2(aa.cast_int(0)); }
            else                          { bb = aa; }

            bb /= ((double) numtot);

            res += bb;

            for ( j = 0 ; j < numvar ; j++ )
            {
                k("&",j)++;

                if ( k(j) >= numpts )
                {
                    k("&",j) = 0;
                }

                else
                {
                    break;
                }
            }
        }

        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.isValEqnDir() )
    {
        const static gentype resx("norm2(x)");
        return resx(a);
    }

    else if ( a.isValStrErr()  ) { constructError(a,res,"String norm2 not implemented"); }
    else if ( a.isValSet()     ) { res = norm2(a.cast_set(0)); }
    else if ( a.isValDgraph()  ) { res = norm2(a.cast_dgraph(0)); }
    else if ( a.isValMatrix()  ) { constructError(a,res,"Matrix norm2 not implemented"); }
    else if ( a.isValVector()  ) { res = norm2vec(a.cast_vector(0)); }
    else if ( a.isValAnion()   ) { res = norm2(a.cast_anion(0)); }
    else if ( a.isValReal()    ) { res = norm2(a.cast_double(0)); }
    else if ( a.isValInteger() ) { res = norm2(a.cast_int(0)); }
    else                         { res = a; }

    return res;
}

gentype normp(const gentype &a, const gentype &q)
{
    gentype res;

    if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 0 ) && q.isCastableToRealWithoutLoss() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        res.zero();
        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 1 ) && q.isCastableToRealWithoutLoss() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();

        int i;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa,bb;

        double p = q.cast_double(0);

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(xa);
            aa.finalise();

            if ( aa.isValEqnDir() )
            {
                const static gentype resx("normp(x,q)");
                bb = resx(aa,q);
            }

            else if ( aa.isValStrErr()  ) { constructError(aa,bb,"String norm not implemented"); }
            else if ( aa.isValSet()     ) { bb = normp(aa.cast_set(0),p); }
            else if ( aa.isValDgraph()  ) { bb = normp(aa.cast_dgraph(0),p); }
            else if ( aa.isValMatrix()  ) { constructError(aa,bb,"Matrix norm not implemented"); }
            else if ( aa.isValVector()  ) { bb = normpvec(aa.cast_vector(0),p); }
            else if ( aa.isValAnion()   ) { bb = normp(aa.cast_anion(0),p); }
            else if ( aa.isValReal()    ) { bb = normp(aa.cast_double(0),p); }
            else if ( aa.isValInteger() ) { bb = normp((double) aa.cast_int(0),p); }
            else                          { bb = aa; }

            bb /= ((double) numpts);

            res += bb;
        }

        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.scalarfn_isscalarfn() && q.isCastableToRealWithoutLoss() )
    {
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numvar = a.scalarfn_i().size();
        int numpts = a.scalarfn_numpts();
        int numtot = (int) pow(numpts,numvar);

        int i,j;
        Vector<int> k(numvar);

        k = zeroint();

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa,bb;

        double p = q.cast_double(0);

        for ( i = 0 ; i < numtot ; i++ )
        {
            for ( j = 0 ; j < numvar ; j++ )
            {
                xa("&",a.scalarfn_i()(j))("&",a.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
            }

            aa = a(xa);
            aa.finalise();

            if ( aa.isValEqnDir() )
            {
                const static gentype resx("normp(x,y)");
                bb = resx(aa,q);
            }

            else if ( aa.isValStrErr()  ) { constructError(aa,bb,"String norm not implemented"); }
            else if ( aa.isValSet()     ) { bb = normp(aa.cast_set(0),p); }
            else if ( aa.isValDgraph()  ) { bb = normp(aa.cast_dgraph(0),p); }
            else if ( aa.isValMatrix()  ) { constructError(aa,bb,"Matrix norm not implemented"); }
            else if ( aa.isValVector()  ) { bb = normpvec(aa.cast_vector(0),p); }
            else if ( aa.isValAnion()   ) { bb = normp(aa.cast_anion(0),p); }
            else if ( aa.isValReal()    ) { bb = normp(aa.cast_double(0),p); }
            else if ( aa.isValInteger() ) { bb = normp((double) aa.cast_int(0),p); }
            else                          { bb = aa; }

            bb /= ((double) numtot);

            res += bb;

            for ( j = 0 ; j < numvar ; j++ )
            {
                k("&",j)++;

                if ( k(j) >= numpts )
                {
                    k("&",j) = 0;
                }

                else
                {
                    break;
                }
            }
        }

        res.scalarfn_setisscalarfn(0);

        return res;
    }

    else if ( a.isValEqnDir() || q.isValEqnDir() )
    {
        const static gentype resx("normp(x,y)");
        return resx(a,q);
    }

    else if ( !q.isCastableToRealWithoutLoss() )
    {
        constructError(a,q,res,"p must be real or integer for p-norm");
    }

    else if ( a.isValStrErr()  ) { constructError(a,q,res,"String norm not implemented"); }
    else if ( a.isValSet()     ) { res = normp(a.cast_set(0),q.cast_double(0)); }
    else if ( a.isValDgraph()  ) { res = normp(a.cast_dgraph(0),q.cast_double(0)); }
    else if ( a.isValMatrix()  ) { constructError(a,q,res,"Matrix norm not implemented"); }
    else if ( a.isValVector()  ) { res = normpvec(a.cast_vector(0),q.cast_double(0)); }
    else if ( a.isValAnion()   ) { res = normp(a.cast_anion(0),q.cast_double(0)); }
    else if ( a.isValReal()    ) { res = normp(a.cast_double(0),q.cast_double(0)); }
    else if ( a.isValInteger() ) { res = normp(a.cast_double(0),q.cast_double(0)); }
    else                         { res = a; }

    return res;
}

gentype angle(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("angle(x)");
        return resx(a);
    }

         if ( a.isValStrErr()  ) { constructError(a,res,"angle ill-defined for string"); }
    else if ( a.isValSet()     ) { constructError(a,res,"angle ill-defined for sets"); }
    else if ( a.isValDgraph()  ) { constructError(a,res,"angle ill-defined for dgraphs"); }
    else if ( a.isValMatrix()  ) { constructError(a,res,"angle ill-defined for matrix"); }
    else if ( a.isValVector()  ) { res = angle(a.cast_vector(0)); }
    else if ( a.isValAnion()   ) { res = angle(a.cast_anion(0)); }
    else if ( a.isValReal()    ) { res = angle(a.cast_double(0)); }
    else if ( a.isValInteger() ) { res = angle(a.cast_int(0)); }
    else                         { res = a; }

    return res;
}

gentype inv(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("inv(x)");
        return resx(a);
    }

    // NB: inv returns pseudo-inverse for non-square matrices

         if ( a.isValStrErr()  ) { constructError(a,res,"inv ill-defined for string"); }
    else if ( a.isValSet()     ) { constructError(a,res,"inv ill-defined for sets"); }
    else if ( a.isValDgraph()  ) { constructError(a,res,"inv ill-defined for dgraphs"); }
    else if ( a.isValMatrix()  ) { res = inv(a.cast_matrix(0)); }
    else if ( a.isValVector()  ) { constructError(a,res,"inv ill-defined for vector"); }
    else if ( a.isValAnion()   ) { res = inv(a.cast_anion(0)); }
    else if ( a.isValReal()    ) { res = inv(a.cast_double(0)); }
    else if ( a.isValInteger() ) { res = inv(a.cast_int(0)); }
    else                         { res = a; }

    return res;
}










// Various maths functions

gentype dubeabsp(const gentype &a, const void *q);
gentype dubeabsp(const gentype &a, const void *q)
{
    return eabsp(a,*((const gentype *) q));
}

gentype dubenormp(const gentype &a, const void *q);
gentype dubenormp(const gentype &a, const void *q)
{
    return enormp(a,*((const gentype *) q));
}

gentype dubpolyDistr(const gentype &a, const void *q);
gentype dubpolyDistr(const gentype &a, const void *q)
{
    return polyDistr(a,*((const gentype *) q));
}

gentype dubPolyDistr(const gentype &a, const void *q);
gentype dubPolyDistr(const gentype &a, const void *q)
{
    return PolyDistr(a,*((const gentype *) q));
}

double imagx(double x);
double imagx(double x)
{
    return 0;
    return x;
}

int imagx(int x);
int imagx(int x)
{
    return 0;
    return x;
}

int sqrtOR(double a);
int sqrtOR(double a)
{
    return a <= 0;
}

int logOR(double a);
int logOR(double a)
{
    return a <= 0;
}

int log10OR(double a);
int log10OR(double a)
{
    return a <= 0;
}

int asinOR(double a);
int asinOR(double a)
{
    return ( a < -1 ) || ( a > 1 );
}

int acosOR(double a);
int acosOR(double a)
{
    return ( a < -1 ) || ( a > 1 );
}

int acosecOR(double a);
int acosecOR(double a)
{
    // acosec(x) = asin(1/x)
    return ( a > -1 ) && ( a < 1 );
}

int asecOR(double a);
int asecOR(double a)
{
    // asec(x) = acos(1/x)
    return ( a > -1 ) && ( a < 1 );
}

int aversOR(double a);
int aversOR(double a)
{
    // avers(x) = acos(1+x)
    return ( a < -2 ) || ( a > 0 );
}

int acoversOR(double a);
int acoversOR(double a)
{
    // avers(x) = asin(1+x)
    return ( a < -2 ) || ( a > 0 );
}

int ahavOR(double a);
int ahavOR(double a)
{
    // ahav(x) = avers(2x)
    return ( a < -1 ) || ( a > 0 );
}

int aexcosecOR(double a);
int aexcosecOR(double a)
{
    // aexcosec(x) = acosec(1+x)
    return ( a > -2 ) && ( a < 0 );
}

int aexsecOR(double a);
int aexsecOR(double a)
{
    // aexsec(x) = asec(1+x)
    return ( a > -2 ) && ( a < 0 );
}

int acoshOR(double a);
int acoshOR(double a)
{
    return a < 1;
}

int atanhOR(double a);
int atanhOR(double a)
{
    return ( a < -1 ) || ( a > 1 );
}

int asechOR(double a);
int asechOR(double a)
{
    // asech(x) = acosh(1/x)
    return ( a > 1 ) || ( a < 0 );
}

int acothOR(double a);
int acothOR(double a)
{
    // acoth(x) = atanh(1/x)
    return ( a > -1 ) && ( a < 1 );
}

int avershOR(double a);
int avershOR(double a)
{
    // aversh(x) = acosh(1+x)
    return a < 0;
}

int ahavhOR(double a);
int ahavhOR(double a)
{
    // ahavh(x) = aversh(2x)
    return a < 0;
}

int aexsechOR(double a);
int aexsechOR(double a)
{
    // aexsechOR(x) = asech(1+x)
    return ( a > 0 ) || ( a < -1 );
}

int asigmOR(double a);
int asigmOR(double a)
{
    // asigmOR(a) = log((1/a)-1)
    return ( a <= 0 ) || ( a >= 1 );
}

int agdOR(double a);
int agdOR(double a)
{
    // agdOR(a) = 2.atanh(tan(a/2))
    return ( a < -NUMBASE_PI/2 ) || ( a > NUMBASE_PI/2 );
}
























// Various elementwise maths functions that do not yet have complex/anionic implementations

gentype gamma(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("gamma(x)");
        return resx(a);
    }

    // Design note: the double type has a much larger range than the
    // integer type, and gamma is liable to produce very large outputs.
    // For this reason we automatically promote integers to doubles when
    // calculating.  To avoid this, use fact(a-1) instead, or re-instate
    // the following line of code:
    //
    // else if ( a.isValInteger()) { res = fact(a-oneintgentype()); }

         if ( a.isValStrErr() ) { constructError(a,res,"gamma not defined for strings."); }
    else if ( a.isValDgraph() ) { constructError(a,res,"gamma not defined for dgraphs."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(gamma); res = temp; }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(gamma); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(gamma); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,res,"gamma not defined for anions."); }

    else
    {
        int ires = numbase_gamma(res.force_double(),a.cast_double(0));

        if ( ires )
	{
            constructError(a,res,"Error calculating gamma");
	}
    }

    return res;
}

gentype lngamma(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("lngamma(x)");
        return resx(a);
    }

         if ( a.isValStrErr() ) { constructError(a,res,"lngamma not defined for strings."); }
    else if ( a.isValDgraph() ) { constructError(a,res,"lngamma not defined for dgraphs."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(lngamma); res = temp; }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(lngamma); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(lngamma); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,res,"lngamma not defined for anions."); }

    else
    {
        int ires = numbase_lngamma(res.force_double(),a.cast_double(0));

        if ( ires )
	{
            constructError(a,res,"Error calculating lngamma");
	}
    }

    return res;
}

gentype psi(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("psi(x)");
        return resx(x);
    }

         if ( x.isValStrErr() ) { constructError(x,res,"psi not defined for strings."); }
    else if ( x.isValSet()    ) { Set<gentype>    temp(x.cast_set(0));    temp.applyon(psi); res = temp; }
    else if ( x.isValDgraph() ) { constructError(x,res,"psi not defined for dgraphs."); }
    else if ( x.isValMatrix() ) { Matrix<gentype> temp(x.cast_matrix(0)); temp.applyon(psi); res = temp; }
    else if ( x.isValVector() ) { Vector<gentype> temp(x.cast_vector(0)); temp.applyon(psi); res = temp; }
    else if ( x.isValAnion()  ) { constructError(x,res,"psi not defined for anions."); }

    else
    {
        int ires = numbase_psi(res.force_double(),x.cast_double(0));

        if ( ires )
	{
            constructError(x,res,"Error calculating psi");
	}
    }

    return res;
}

gentype dubpsi_n(const gentype &x, const void *i);
gentype dubpsi_n(const gentype &x, const void *i)
{
    return psi_n(*((const gentype *) i),x);
}

gentype psi_n(const gentype &i, const gentype &x)
{
    gentype res;

    if ( i.isValEqnDir() || x.isValEqnDir() )
    {
        const static gentype resx("psi_n(x,y)");
        return resx(i,x);
    }

         if ( x.isValStrErr()                     ) { constructError(i,x,res,"psi_n not defined for strings."); }
    else if ( x.isValSet()                        ) { Set<gentype>    temp(x.cast_set(0));    temp.applyon(dubpsi_n,(void *) &i); res = temp; }
    else if ( x.isValDgraph()                     ) { constructError(i,x,res,"psi_n not defined for dgraphs."); }
    else if ( x.isValMatrix()                     ) { Matrix<gentype> temp(x.cast_matrix(0)); temp.applyon(dubpsi_n,(void *) &i); res = temp; }
    else if ( x.isValVector()                     ) { Vector<gentype> temp(x.cast_vector(0)); temp.applyon(dubpsi_n,(void *) &i); res = temp; }
    else if ( x.isValAnion()                      ) { constructError(i,x,res,"psi_n not defined for anions."); }
    else if ( !i.isCastableToIntegerWithoutLoss() ) { constructError(i,x,res,"psi_n not defined for non-integer n."); }

    else
    {
        int ires = numbase_psi_n(res.force_double(),i.cast_int(0),x.cast_double(0));

        if ( ires )
	{
            constructError(x,res,"Error calculating psi_n");
	}
    }

    return res;
}


gentype dubgamic(const gentype &a, const void *x);
gentype dubgamic(const gentype &a, const void *x)
{
    return gamic(a,*((const gentype *) x));
}

gentype debgamic(const gentype &x, const void *a);
gentype debgamic(const gentype &x, const void *a)
{
    return gamic(*((const gentype *) a),x);
}

gentype gamic(const gentype &a, const gentype &x)
{
    gentype res;

    if ( a.isValEqnDir() || x.isValEqnDir() )
    {
        const static gentype resx("gamic(x,y)");
        return resx(a,x);
    }

         if ( a.isValStrErr() ) { constructError(a,x,res,"gamic not defined for strings."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0)   ); temp.applyon(dubgamic,(void *) &x); res = temp; }
    else if ( a.isValDgraph() ) { constructError(a,x,res,"gamic not defined for dgraphs."); }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(dubgamic,(void *) &x); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(dubgamic,(void *) &x); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,x,res,"gamic not defined for anions."); }
    else if ( x.isValStrErr() ) { constructError(a,x,res,"gamic not defined for strings."); }
    else if ( x.isValSet()    ) { Set<gentype>    temp(x.cast_set(0));    temp.applyon(debgamic,(void *) &a); res = temp; }
    else if ( x.isValDgraph() ) { constructError(a,x,res,"gamic not defined for dgraphs."); }
    else if ( x.isValMatrix() ) { Matrix<gentype> temp(x.cast_matrix(0)); temp.applyon(debgamic,(void *) &a); res = temp; }
    else if ( x.isValVector() ) { Vector<gentype> temp(x.cast_vector(0)); temp.applyon(debgamic,(void *) &a); res = temp; }
    else if ( x.isValAnion()  ) { constructError(a,x,res,"gamic not defined for anions."); }

    else
    {
        int ires = numbase_gamma_inc(res.force_double(),a.cast_double(0),x.cast_double(0));

        if ( ires )
	{
            constructError(x,res,"Error calculating gamma_inc");
	}
    }

    return res;
}

gentype zeta(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("zeta(x)");
        return resx(a);
    }

    // Design note: zeta almost always evaluates to double, so we simply
    // assume this for all cases.

         if ( a.isValStrErr() ) { constructError(a,res,"zeta not defined for strings."); }
    else if ( a.isValDgraph() ) { constructError(a,res,"zeta not defined for dgraphs."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(zeta); res = temp; }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(zeta); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(zeta); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,res,"zeta not defined for anions."); }

    else
    {
        int ires = numbase_zeta(res.force_double(),a.cast_double(0));

        if ( ires )
	{
            constructError(a,res,"Error calculating zeta");
	}
    }

    return res;
}

gentype lambertW(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("lambertW(x)");
        return resx(a);
    }

    // Design note: lambert W almost always evaluates to double, so we simply
    // assume this for all cases.

         if ( a.isValStrErr() ) { constructError(a,res,"lambert W not defined for strings."); }
    else if ( a.isValDgraph() ) { constructError(a,res,"lambert W not defined for dgraphs."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(lambertW); res = temp; }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(lambertW); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(lambertW); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,res,"lambert W not defined for anions."); }

    else
    {
        int ires = numbase_lambertW(res.force_double(),a.cast_double(0));

        if ( ires )
	{
            constructError(a,res,"Error calculating lambert W");
	}
    }

    return res;
}

gentype lambertWx(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("lambertWx(x)");
        return resx(a);
    }

    // Design note: lambert W almost always evaluates to double, so we simply
    // assume this for all cases.

         if ( a.isValStrErr() ) { constructError(a,res,"lambert W not defined for strings."); }
    else if ( a.isValDgraph() ) { constructError(a,res,"lambert W not defined for dgraphs."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(lambertWx); res = temp; }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(lambertWx); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(lambertWx); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,res,"lambert W not defined for anions."); }

    else
    {
        int ires = numbase_lambertWx(res.force_double(),a.cast_double(0));

        if ( ires )
	{
            constructError(a,res,"Error calculating lambert W");
	}
    }

    return res;
}

gentype erf(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("erf(x)");
        return resx(x);
    }

         if ( x.isValStrErr() ) { constructError(x,res,"erf not defined for strings."); }
    else if ( x.isValDgraph() ) { constructError(x,res,"erf not defined for dgraphs."); }
    else if ( x.isValSet()    ) { Set<gentype>    temp(x.cast_set(0));    temp.applyon(erf); res = temp; }
    else if ( x.isValMatrix() ) { Matrix<gentype> temp(x.cast_matrix(0)); temp.applyon(erf); res = temp; }
    else if ( x.isValVector() ) { Vector<gentype> temp(x.cast_vector(0)); temp.applyon(erf); res = temp; }
    else if ( x.isValAnion()  ) { constructError(x,res,"erf not defined for anions."); }

    else
    {
        int ires = numbase_erf(res.force_double(),x.cast_double(0));

        if ( ires )
	{
            constructError(x,res,"Error calculating erf");
	}
    }

    return res;
}

gentype erfc(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("erfc(x)");
        return resx(x);
    }

         if ( x.isValStrErr() ) { constructError(x,res,"erfc not defined for strings."); }
    else if ( x.isValDgraph() ) { constructError(x,res,"erfc not defined for dgraphs."); }
    else if ( x.isValSet()    ) { Set<gentype>    temp(x.cast_set(0));    temp.applyon(erfc); res = temp; }
    else if ( x.isValMatrix() ) { Matrix<gentype> temp(x.cast_matrix(0)); temp.applyon(erfc); res = temp; }
    else if ( x.isValVector() ) { Vector<gentype> temp(x.cast_vector(0)); temp.applyon(erfc); res = temp; }
    else if ( x.isValAnion()  ) { constructError(x,res,"erfc not defined for anions."); }

    else
    {
        int ires = numbase_erfc(res.force_double(),x.cast_double(0));

        if ( ires )
	{
            constructError(x,res,"Error calculating erfc");
	}
    }

    return res;
}

gentype dawson(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("dawson(x)");
        return resx(x);
    }

         if ( x.isValStrErr() ) { constructError(x,res,"dawson not defined for strings."); }
    else if ( x.isValDgraph() ) { constructError(x,res,"dawson not defined for dgraphs."); }
    else if ( x.isValSet()    ) { Set<gentype>    temp(x.cast_set(0));    temp.applyon(dawson); res = temp; }
    else if ( x.isValMatrix() ) { Matrix<gentype> temp(x.cast_matrix(0)); temp.applyon(dawson); res = temp; }
    else if ( x.isValVector() ) { Vector<gentype> temp(x.cast_vector(0)); temp.applyon(dawson); res = temp; }
    else if ( x.isValAnion()  ) { constructError(x,res,"dawson not defined for anions."); }

    else
    {
        int ires = numbase_dawson(res.force_double(),x.cast_double(0));

        if ( ires )
	{
            constructError(x,res,"Error calculating dawson");
	}
    }

    return res;
}




/*
gentype emaxv(const gentype &a, const gentype &b)
{
    gentype res;

    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype resx("emaxv(a,b)");
        return resx(a,b);
    }

    if ( a.isValMatrix() ) 
    {
        if ( !b.isValMatrix() )
        {
            constructError(a,b,res,"emaxv have either neither or both matrix");
            return res;
        }

        const Matrix<gentype> &aa = (const Matrix<gentype> &) aa;
        const Matrix<gentype> &bb = (const Matrix<gentype> &) bb;

        if ( aa.numRows() != bb.numRows() )
        {
            constructError(a,b,res,"emaxv matrix row sizes must agree");
            return res;
        }

        if ( aa.numRows() != bb.numRows() )
        {
            constructError(a,b,res,"emaxv matrix column sizes must agree");
            return res;
        }

        int i,j;

        Matrix<gentype> &tres = res.force_matrix().resize(aa.numRows(),aa.numCols());

        for ( i = 0 ; i < aa.numRows() ; i++ )
        {
            for ( j = 0 ; j < aa.numCols() ; j++ )
            {
                tres("&",i,j) = emaxv(aa(i,j),bb(i,j));
            }
        }
    }

    else if ( b.isValVector() ) 
    { 
        if ( !b.isValVector() )
        {
            constructError(a,b,res,"emaxv have either neither or both vector");
            return res;
        }

        const Vector<gentype> &aa = (const Vector<gentype> &) aa;
        const Vector<gentype> &bb = (const Vector<gentype> &) bb;

        if ( aa.size() != bb.size() )
        {
            constructError(a,b,res,"emaxv vector sizes must agree");
            return res;
        }

        int i;

        Vector<gentype> &tres = res.force_vector().resize(aa.numRows(),aa.numCols());

        for ( i = 0 ; i < aa.size() ; i++ )
        {
            tres("&",i,j) = emaxv(aa(i),bb(i));
        }
    }

    else
    {
        gentype tmp = ge(a,b);

        if ( !tmp.isCastableToIntegerWithoutLoss() )
        {
            constructError(a,b,res,"emaxv comparison failed");
            return res;
        }

        res = ((int) tmp) ? a : b;
    }

    return res;
}




gentype eminv(const gentype &a, const gentype &b)
{
    gentype res;

    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype resx("eminv(a,b)");
        return resx(a,b);
    }

    if ( a.isValMatrix() ) 
    {
        if ( !b.isValMatrix() )
        {
            constructError(a,b,res,"emaxv have either neither or both matrix");
            return res;
        }

        const Matrix<gentype> &aa = (const Matrix<gentype> &) aa;
        const Matrix<gentype> &bb = (const Matrix<gentype> &) bb;

        if ( aa.numRows() != bb.numRows() )
        {
            constructError(a,b,res,"eminv matrix row sizes must agree");
            return res;
        }

        if ( aa.numRows() != bb.numRows() )
        {
            constructError(a,b,res,"eminv matrix column sizes must agree");
            return res;
        }

        int i,j;

        Matrix<gentype> &tres = res.force_matrix().resize(aa.numRows(),aa.numCols());

        for ( i = 0 ; i < aa.numRows() ; i++ )
        {
            for ( j = 0 ; j < aa.numCols() ; j++ )
            {
                tres("&",i,j) = eminv(aa(i,j),bb(i,j));
            }
        }
    }

    else if ( b.isValVector() ) 
    { 
        if ( !b.isValVector() )
        {
            constructError(a,b,res,"eminv have either neither or both vector");
            return res;
        }

        const Vector<gentype> &aa = (const Vector<gentype> &) aa;
        const Vector<gentype> &bb = (const Vector<gentype> &) bb;

        if ( aa.size() != bb.size() )
        {
            constructError(a,b,res,"eminv vector sizes must agree");
            return res;
        }

        int i;

        Vector<gentype> &tres = res.force_vector().resize(aa.numRows(),aa.numCols());

        for ( i = 0 ; i < aa.size() )
        {
            tres("&",i,j) = eminv(aa(i),bb(i));
        }
    }

    else
    {
        gentype tmp = le(a,b);

        if ( !tmp.isCastableToIntegerWithoutLoss() )
        {
            constructError(a,b,res,"eminv comparison failed");
            return res;
        }

        res = ((int) tmp) ? a : b;
    }

    return res;
}
*/










// Global function table access

inline gentype fnA(const gentype &i, const gentype &j)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() )
    {
        const static gentype resx("fnA(x,y)");
        return resx(i,j);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,res,"fnA not defined for non-integer function index."); 
    }

    else
    {
        const gentype xa;
        const gentype xb;

        int ia = 0;
        int ib = 0;

        evalgenFunc(i.cast_int(0),j.cast_int(0),xa,ia,xb,ib,res);
    }

    return res;
}

inline gentype fnB(const gentype &i, const gentype &j, const gentype &xa)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() || xa.isValEqnDir() )
    {
        const static gentype resx("fnB(x,y,z)");
        return resx(i,j,xa);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,xa,res,"fnB not defined for non-integer function index."); 
    }

    else
    {
        const gentype xb;

        int ia = 0;
        int ib = 0;

        evalgenFunc(i.cast_int(0),j.cast_int(0),xa,ia,xb,ib,res);
    }

    return res;
}

inline gentype fnC(const gentype &i, const gentype &j, const gentype &xa, const gentype &xb)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() || xa.isValEqnDir() || xb.isValEqnDir() )
    {
        const static gentype resx("fnC(x,y,z,var(0,3))");
        return resx(i,j,xa,xb);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,xa,xb,res,"fnC not defined for non-integer function index."); 
    }

    else
    {
        int ia = 0;
        int ib = 0;

        evalgenFunc(i.cast_int(0),j.cast_int(0),xa,ia,xb,ib,res);
    }

    return res;
}

inline gentype dfnB(const gentype &i, const gentype &j, const gentype &xa, const gentype &ia)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() || xa.isValEqnDir() || ia.isValEqnDir() )
    {
        const static gentype resx("dfnB(x,y,z,var(0,3))");
        return resx(i,j,xa,ia);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() || !ia.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,xa,ia,res,"dfnB not defined for non-integer function index."); 
    }

    else
    {
        const gentype xb;

        int ib = 0;

        evalgenFunc(i.cast_int(0),j.cast_int(0),xa,ia.cast_int(0),xb,ib,res);
    }

    return res;
}

inline gentype dfnC(const gentype &i, const gentype &j, const gentype &xa, const gentype &ia, const gentype &xb, const gentype &ib)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() || xa.isValEqnDir() || ia.isValEqnDir() || xb.isValEqnDir() || ib.isValEqnDir() )
    {
        const static gentype resx("dfnC(x,y,z,var(0,3),var(0,4),var(0,5))");
        return resx(i,j,xa,ia,xb,ib);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() || !ia.isCastableToIntegerWithoutLoss() || !ib.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,xa,ia,xb,ib,res,"dfnC not defined for non-integer function index."); 
    }

    else
    {
        evalgenFunc(i.cast_int(0),j.cast_int(0),xa,ia.cast_int(0),xb,ib.cast_int(0),res);
    }

    return res;
}

inline gentype efnB(const gentype &i, const gentype &j, const gentype &xa)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() || xa.isValEqnDir() )
    {
        const static gentype resx("efnB(x,y,z)");
        return resx(i,j,xa);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,xa,res,"efnB not defined for non-integer function index."); 
    }

    else if ( !xa.isValVector() )
    {
        const Vector<gentype> xb(xa.size());

        int ia = 0;
        int ib = 0;

        evalegenFunc(i.cast_int(0),j.cast_int(0),(const Vector<gentype> &) xa,ia,xb,ib,res.force_vector());
    }

    else
    {
        constructError(i,j,xa,res,"efnB not defined for non-vector argument."); 
    }

    return res;
}

inline gentype efnC(const gentype &i, const gentype &j, const gentype &xa, const gentype &xb)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() || xa.isValEqnDir() || xb.isValEqnDir() )
    {
        const static gentype resx("efnC(x,y,z,var(0,3))");
        return resx(i,j,xa,xb);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,xa,xb,res,"efnC not defined for non-integer function index."); 
    }

    else if ( !xa.isValVector() || !xb.isValVector() )
    {
        int ia = 0;
        int ib = 0;

        evalegenFunc(i.cast_int(0),j.cast_int(0),(const Vector<gentype> &) xa,ia,(const Vector<gentype> &) xb,ib,res.force_vector());
    }

    else
    {
        constructError(i,j,xa,res,"efnB not defined for non-vector argument."); 
    }

    return res;
}

inline gentype edfnB(const gentype &i, const gentype &j, const gentype &xa, const gentype &ia)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() || xa.isValEqnDir() || ia.isValEqnDir() )
    {
        const static gentype resx("edfnB(x,y,z,var(0,3))");
        return resx(i,j,xa,ia);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() || !ia.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,xa,ia,res,"edfnB not defined for non-integer function index."); 
    }

    else
    {
        const Vector<gentype> xb(xa.size());

        int ib = 0;

        evalegenFunc(i.cast_int(0),j.cast_int(0),(const Vector<gentype> &) xa,ia.cast_int(0),xb,ib,res.force_vector());
    }

    return res;
}

inline gentype edfnC(const gentype &i, const gentype &j, const gentype &xa, const gentype &ia, const gentype &xb, const gentype &ib)
{
    gentype res;

    if ( i.isValEqnDir() || j.isValEqnDir() || xa.isValEqnDir() || ia.isValEqnDir() || xb.isValEqnDir() || ib.isValEqnDir() )
    {
        const static gentype resx("edfnC(x,y,z,var(0,3),var(0,4),var(0,5))");
        return resx(i,j,xa,ia,xb,ib);
    }

    if ( !i.isCastableToIntegerWithoutLoss() || !j.isCastableToIntegerWithoutLoss() || !ia.isCastableToIntegerWithoutLoss() || !ib.isCastableToIntegerWithoutLoss() ) 
    { 
        constructError(i,j,xa,ia,xb,ib,res,"edfnC not defined for non-integer function index."); 
    }

    else
    {
        evalegenFunc(i.cast_int(0),j.cast_int(0),(const Vector<gentype> &) xa,ia.cast_int(0),(const Vector<gentype> &) xb,ib.cast_int(0),res.force_vector());
    }

    return res;
}

gentype irand(const gentype &i)
{
    gentype res;

    if ( i.isValEqnDir() )
    {
        const static gentype resx("irand(x)");
        return resx(i);
    }

    if ( i.isCastableToIntegerWithoutLoss() )
    {
        res.force_int() = (svm_rand()%(i.cast_int(0)));
    }

    else if ( i.isValVector() )
    {
        Vector<gentype> &r = res.force_vector(i.size());

        int ii;

        for ( ii = 0 ; ii < i.size() ; ii++ )
        {
            r("&",ii) = irand(i(ii));
        }
    }

    else if ( i.isValMatrix() )
    {
        Matrix<gentype> &r = res.force_matrix(i.numRows(),i.numCols());

        int ii,jj;

        for ( ii = 0 ; ii < i.numRows() ; ii++ )
        {
            for ( jj = 0 ; jj < i.numCols() ; jj++ )
            {
                r("&",ii,jj) = irand(i(ii,jj));
            }
        }
    }

    else
    { 
        constructError(i,res,"irand only defined for integer, vector and matrix."); 
    }

    return res;
}

gentype urand(const gentype &l, const gentype &u)
{
    gentype res;

    if ( l.isValEqnDir() || u.isValEqnDir() )
    {
        const static gentype resx("urand(x,y)");
        return resx(l,u);
    }

    if ( l.isCastableToRealWithoutLoss() && u.isCastableToRealWithoutLoss() ) 
    {
        randfill(res.force_double());
        res.force_double() *= ( u.cast_double(0) - l.cast_double(0) );
        res.force_double() += l.cast_double(0);
    }

    else if ( l.isValVector() && u.isValVector() )
    {
        NiceAssert( l.size() == u.size() );

        Vector<gentype> &r = res.force_vector(l.size());

        int ii;

        for ( ii = 0 ; ii < l.size() ; ii++ )
        {
            r("&",ii) = urand(l(ii),u(ii));
        }
    }

    else if ( l.isValMatrix() && u.isValMatrix() )
    {
        NiceAssert( l.numRows() == u.numRows() );
        NiceAssert( l.numCols() == u.numCols() );

        Matrix<gentype> &r = res.force_matrix(l.numRows(),l.numCols());

        int ii,jj;

        for ( ii = 0 ; ii < l.numRows() ; ii++ )
        {
            for ( jj = 0 ; jj < l.numCols() ; jj++ )
            {
                r("&",ii,jj) = urand(l(ii,jj),u(ii,jj));
            }
        }
    }

    else
    { 
        constructError(l,u,res,"urand only defined for real, vector and matrix."); 
    }

    return res;
}

gentype grand(const gentype &m, const gentype &v)
{
    gentype res;

    if ( m.isValEqnDir() || v.isValEqnDir() )
    {
        const static gentype resx("grand(x,y)");
        return resx(m,v);
    }

    if ( m.isCastableToRealWithoutLoss() && v.isCastableToRealWithoutLoss() ) 
    {
        double &r = res.force_double();

        double mm = m.cast_double(0);
        double vv = v.cast_double(0);

        randnfill(r);

        r *= sqrt(vv);
        r += mm;
    }

    else if ( m.isValVector() && v.isCastableToRealWithoutLoss() )
    {
        Vector<gentype> &r = res.force_vector(m.size());

        const Vector<gentype> &mm = m.cast_vector(0);

        // Assumption: vv is symmetric positive definite

        int ii;

        for ( ii = 0 ; ii < m.size() ; ii++ )
        {
            randnfill(r("&",ii).force_double());
        }

        r *= sqrt(v);
        r += mm;
    }

    else if ( m.isValVector() && v.isValMatrix() )
    {
        // We use a partial Cholesky factorisation to allow for positive *semi*-definite covariances

        // v = LL' = LU (U = L')
        //
        // r = m + L.n
        // r(p) = m(p) + L(p,:).n(:)
        // r(p) = m(p) + L(p,p).n(p)
        // r(p) = m(p) + L(p,p(0:1:s-1)).n(p(0:1:s-1))

        Vector<gentype> &r = res.force_vector(m.size());

        const Vector<gentype> &mm = (m.cast_vector(0));
        const Matrix<gentype> &vv = (v.cast_matrix(0));

        NiceAssert( vv.isSquare() );
        NiceAssert( mm.size() == vv.numRows() );

        // Assumption: vv is symmetric positive definite

        Matrix<gentype> L(m.size(),m.size());
        Vector<int> p(m.size());
        int s;

        vv.naivepartChol(L,p,s);
	
        int ii;

        if ( s > 0 )
        {
            for ( ii = 0 ; ii < s ; ii++ )
            {
                randnfill(r("&",ii).force_double());
            }
        }

        if ( s < m.size() )
        {
            for ( ii = s ; ii < m.size() ; ii++ )
            {
                r("&",ii).force_double() = 0.0;
            }
        }

        retVector<gentype> tmpva;
        retVector<gentype> tmpvb;
        retVector<int>     tmpvc;
        retVector<int>     tmpvd;
        retMatrix<gentype> tmpma;

        r("&",p,tmpva) = (L(p,p(zeroint(),1,s-1,tmpvc),tmpma))*r(p(zeroint(),1,s-1,tmpvd),tmpvb);
        r += mm;


// This version works, but only if v is positive definite, not positive *semi*-definite

//        // c = UU'
//        //
//        // r = m + U.n
//        // r' = m' + n'.U'
//
//        Vector<gentype> &r = res.force_vector(m.size());
//
//        const Vector<gentype> &mm = (m.cast_vector(0));
//        const Matrix<gentype> &vv = (v.cast_matrix(0));
//
//        NiceAssert( vv.isSquare() );
//        NiceAssert( mm.size() == vv.numRows() );
//
//        // Assumption: vv is symmetric positive definite
//
//        Matrix<gentype> cc(m.size(),m.size());
//
//        vv.naiveChol(cc,0);
//	
//        int ii;
//
//        for ( ii = 0 ; ii < m.size() ; ii++ )
//        {
//            randnfill(r("&",ii).force_double());
//        }
//
//        r *= cc.transpose();
//        r += mm;


// This version doesn't work as the SVD code I found is faulty (try [ 1 1 1 1 ; 1 1 1 1 ; 1 1 1 1 ; 1 1 1 1 ] and watch it fail

//        // c = UDV   (V=U')
//        //
//        // r = m + U.sqrt(D).n
//        // r' = m' + n'.sqrt(D).V
//
//        Vector<double> r(m.size());
//
//        const Vector<double> &mm = (const Vector<double> &) m;
//        const Matrix<double> &cc = (const Matrix<double> &) v;
//
//        NiceAssert( cc.isSquare() );
//        NiceAssert( mm.size() == cc.numRows() );
//
//        Matrix<double> uu,vv;
//        Vector<double> dd;
//
//        cc.SVD(uu,dd,vv);
//
//errstream() << "phantomx: uu = " << uu << "\n";	
//errstream() << "phantomx: dd = " << dd << "\n";	
//errstream() << "phantomx: vv = " << vv << "\n";	
//Matrix<double> ddd(dd.size(),dd.size());
//ddd = 0.0;
//ddd.diagoffset(dd);
//uu.transpose();
//errstream() << "phantomx: uu.dd.vv = " << uu*(ddd*vv) << "\n";
//        int ii;
//
//        for ( ii = 0 ; ii < m.size() ; ii++ )
//        {
//            randnfill(r("&",ii));
//
//            r("&",ii) *= sqrt(dd(ii));
//        }
//
////        r *= vv;
//
//        Vector<double> s(r);
//
//        s = ((uu*r)+mm);
//
//        res = r;
    }

    else
    { 
        constructError(m,v,res,"grand only defined for real,real and vector,matrix."); 
    }

    return res;
}

gentype testfn(const gentype &i, const gentype &x)
{
    gentype res;

    if ( i.isValEqnDir() || x.isValEqnDir() )
    {
        eqnfallback:
        const static gentype resx("testfn(x,y)");
        return resx(i,x);
    }

    if ( !(i.isCastableToIntegerWithoutLoss()) )
    {
        constructError(i,x,res,"Index must be integer for testfn.");
    }

    else if ( !(x.isCastableToVectorWithoutLoss()) )
    {
        vecfallback:
        constructError(i,x,res,"x must be castable to real-valued vector for testfn.");
    }

    else
    {
        int ii = (int) i;
        const Vector<gentype> &xx = (const Vector<gentype> &) x;

        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            if ( xx(j).isValEqnDir() )
            {
                goto eqnfallback;
            }

            if ( !(xx(j).isCastableToRealWithoutLoss()) )
            {
                goto vecfallback;
            }
        }

        const Vector<double> &xxx = (const Vector<double> &) x;

        if ( evalTestFn(ii,res.force_double(),xxx) )
        {
            constructError(i,x,res,"x non-feasible in testfn.");
        }
    }

    return res;
}

gentype testfnA(const gentype &i, const gentype &x, const gentype &A)
{
    gentype res;

    if ( i.isValEqnDir() || x.isValEqnDir() || A.isValEqnDir() )
    {
        eqnfallback:
        const static gentype resx("testfnA(x,y,z)");
        return resx(i,x,A);
    }

    if ( !(i.isCastableToIntegerWithoutLoss()) )
    {
        constructError(i,x,A,res,"Index must be integer for testfnA.");
    }

    else if ( !(x.isCastableToVectorWithoutLoss()) )
    {
        vecfallback:
        constructError(i,x,A,res,"x must be castable to real-valued vector for testfnA.");
    }

    else if ( !(A.isCastableToMatrixWithoutLoss()) )
    {
        matfallback:
        constructError(i,x,A,res,"A must be castable to real-valued matrix for testfnA.");
    }

    else
    {
        int ii = (int) i;
        const Vector<gentype> &xx = (const Vector<gentype> &) x;
        const Matrix<gentype> &AA = (const Matrix<gentype> &) A;

        int j,k;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            if ( xx(j).isValEqnDir() )
            {
                goto eqnfallback;
            }

            if ( !(xx(j).isCastableToRealWithoutLoss()) )
            {
                goto vecfallback;
            }
        }

        for ( j = 0 ; j < AA.numRows() ; j++ )
        {
            for ( k = 0 ; k < AA.numCols() ; k++ )
            {
                if ( !(AA(j,k).isCastableToRealWithoutLoss()) )
                {
                    goto matfallback;
                }
            }
        }

        const Vector<double> &xxx = (const Vector<double> &) x;
        const Matrix<double> &AAA = (const Matrix<double> &) A;

        if ( evalTestFn(ii,res.force_double(),xxx,&AAA) )
        {
            constructError(i,x,A,res,"x non-feasible in testfnA.");
        }
    }

    return res;
}

gentype partestfn(const gentype &i, const gentype &M, const gentype &x)
{
    gentype res;

    if ( i.isValEqnDir() || M.isValEqnDir() || x.isValEqnDir() )
    {
        const static gentype resx("partestfn(x,y,z)");
        return resx(i,M,x);
    }

    if ( !(i.isCastableToIntegerWithoutLoss()) )
    {
        constructError(i,M,x,res,"Index must be integer for partestfn.");
    }

    else if ( !(M.isCastableToIntegerWithoutLoss()) )
    {
        constructError(i,M,x,res,"Target dimensino must be integer for partestfn.");
    }

    else if ( !(x.isCastableToVectorWithoutLoss()) )
    {
        vecfallback:
        constructError(i,M,x,res,"x must be castable to real-valued vector for partestfn.");
    }

    else
    {
        int ii = (int) i;
        int MM = (int) M;
        const Vector<gentype> &xx = (const Vector<gentype> &) x;

        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            if ( !(xx(j).isCastableToRealWithoutLoss()) )
            {
                goto vecfallback;
            }
        }

        const Vector<double> &xxx = (const Vector<double> &) x;

        Vector<double> ress(MM);

        if ( evalTestFn(ii,xxx.size(),MM,ress,xxx) )
        {
            constructError(i,M,x,res,"x non-feasible in partestfn.");
        }

        else
        {
            res = ress;
        }
    }

    return res;
}

gentype partestfnA(const gentype &i, const gentype &M, const gentype &x, const gentype &alpha)
{
    gentype res;

    if ( i.isValEqnDir() || M.isValEqnDir() || x.isValEqnDir() || alpha.isValEqnDir() )
    {
        const static gentype resx("partestfnA(x,y,z,v)");
        return resx(i,M,x,alpha);
    }

    if ( !(i.isCastableToIntegerWithoutLoss()) )
    {
        constructError(i,M,x,alpha,res,"Index must be integer for partestfnA.");
    }

    else if ( !(M.isCastableToIntegerWithoutLoss()) )
    {
        constructError(i,M,x,alpha,res,"Target dimension must be integer for partestfnA.");
    }

    else if ( !(x.isCastableToVectorWithoutLoss()) )
    {
        vecfallback:
        constructError(i,M,x,alpha,res,"x must be castable to real-valued vector for partestfnA.");
    }

    else if ( !(alpha.isCastableToRealWithoutLoss()) )
    {
        constructError(i,M,x,alpha,res,"alpha must be castable to real for partestfnA.");
    }

    else
    {
        int ii = (int) i;
        int MM = (int) M;
        const Vector<gentype> &xx = (const Vector<gentype> &) x;
        double Alpha = (double) alpha;

        int j;

        for ( j = 0 ; j < xx.size() ; j++ )
        {
            if ( !(xx(j).isCastableToRealWithoutLoss()) )
            {
                goto vecfallback;
            }
        }

        const Vector<double> &xxx = (const Vector<double> &) x;

        Vector<double> ress(MM);

        if ( evalTestFn(ii,xxx.size(),MM,ress,xxx,Alpha) )
        {
            constructError(i,M,x,alpha,res,"x non-feasible in partestfn.");
        }

        else
        {
            res = ress;
        }
    }

    return res;
}

















// Type conversion functions

gentype ceil(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("ceil(x)");
        return resx(a);
    }

         if ( a.isValStrErr() ) { constructError(a,res,"ceil not defined for strings."); }
    else if ( a.isValDgraph() ) { constructError(a,res,"ceil not defined for dgraphs."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(ceil); res = temp; }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(ceil); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(ceil); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,res,"ceil not defined for anions."); }
    else
    {
	double x = a.cast_double(0);
	int j = a.cast_int(0);

	while ( j < x )
	{
            j++;
	}

        res = j;
    }

    return res;
}

gentype floor(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("floor(x)");
        return resx(a);
    }

         if ( a.isValStrErr() ) { constructError(a,res,"floor not defined for strings."); }
    else if ( a.isValDgraph() ) { constructError(a,res,"floor not defined for dgraphs."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(floor); res = temp; }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(floor); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(floor); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,res,"floor not defined for anions."); }
    else
    {
	double x = a.cast_double(0);
	int j = a.cast_int(0);

	while ( j > x )
	{
            j--;
	}

        res = j;
    }

    return res;
}

gentype rint(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("rint(x)");
        return resx(a);
    }

         if ( a.isValStrErr() ) { constructError(a,res,"rint not defined for strings."); }
    else if ( a.isValDgraph() ) { constructError(a,res,"rint not defined for dgraphs."); }
    else if ( a.isValSet()    ) { Set<gentype>    temp(a.cast_set(0));    temp.applyon(rint); res = temp; }
    else if ( a.isValMatrix() ) { Matrix<gentype> temp(a.cast_matrix(0)); temp.applyon(rint); res = temp; }
    else if ( a.isValVector() ) { Vector<gentype> temp(a.cast_vector(0)); temp.applyon(rint); res = temp; }
    else if ( a.isValAnion()  ) { constructError(a,res,"rint not defined for anions."); }
    else
    {
	double x = a.cast_double(0);
	int ji = a.cast_int(0);
        int jj = ji;

	while ( ji > x )
	{
            ji--;
	}

	while ( jj < x )
	{
            jj++;
	}

	if ( abs2(ji-x) < abs2(jj-x) ) { res = ji; }
	else                           { res = jj; }
    }

    return res;
}


gentype deref(const gentype &a, const gentype &i)
{
    gentype res;

    if ( a.isValEqnDir() || i.isValEqnDir() )
    {
        const static gentype resx("deref(x,y)");
        return resx(a,i);
    }

    else if ( a.isValMatrix() )
    {
        if ( !(i.isValVector()) )
        {
            constructError(a,i,res,"deref index incorrect for matrix.");
        }

        else if ( i.size() != 2 )
        {
            constructError(a,i,res,"bad deref index size for matrix.");
        }

        else
        {
            res = derefm(a,derefv(i,zerointgentype()),derefv(i,oneintgentype()));
        }
    }

    else if ( a.isValVector() )
    {
        if ( !(i.isValVector()) )
        {
            constructError(a,i,res,"deref index incorrect for vector.");
        }

        else if ( i.size() != 1 )
        {
            constructError(a,i,res,"deref index size for vector.");
        }

        else
        {
            res = derefv(a,derefv(i,zerointgentype()));
        }
    }

    else
    {
        if ( !(i.isValVector()) )
        {
            constructError(a,i,res,"deref index incorrect for scalar.");
        }

        else if ( i.size() )
        {
            constructError(a,i,res,"deref index size for scalar.");
        }

        else
        {
            res = a;
        }
    }

    return res;
}



gentype derefv(const gentype &a, const gentype &i)
{
    gentype res;

    if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() >= 1 ) && ( a.scalarfn_j().size() >= 1 ) )
    {
        // This case is basically an infinite-dimensional vector, so treat as such
        // We assume that it is the first argument in the scalar function that is being
        // substituted!

        res = a;

        Vector<int> is(a.scalarfn_i());
        Vector<int> js(a.scalarfn_j());

        int vi = is(0);
        int vj = js(0);

        is.remove(0);
        js.remove(0);

        if ( !is.size() || !js.size() )
        {
            res.scalarfn_setisscalarfn(0);
        }

        else
        {
            res.scalarfn_seti(is);
            res.scalarfn_setj(js);
        }

        SparseVector<SparseVector<gentype> > subvals;

        subvals("&",vi)("&",vj) = i;

        res.substitute(subvals);
    }

    else if ( a.isValEqnDir() || i.isValEqnDir() )
    {
        const static gentype resx("derefv(x,y)");
        return resx(a,i);
    }

    else if ( !(a.isValVector()) )
    {
        constructError(a,i,res,"derefv works only on vectors.");
    }

    else if ( a.infsize() )
    {
        (a.cast_vector(0))(res,i);
    }

    else if ( i.isCastableToIntegerWithoutLoss() )
    {
        if ( ( i.cast_int(0) < 0 ) || ( i.cast_int(0) >= a.size() ) )
        {
            constructError(a,i,res,"index out of range in derefv.");
        }

        else
        {
            res = (a.cast_vector(0))(i.cast_int(0));
        }
    }

    else if ( i.isValVector() )
    {
        int k;
        gentype kk;
        Vector<gentype> resv(i.size());

        if ( i.size() )
        {
            for ( k = 0 ; k < i.size() ; k++ )
            {
                kk = k;
                resv("&",k) = derefv(a,derefv(i,kk));
            }
        }

        res = resv;
    }

    else
    {
        constructError(a,i,res,"malformed index in derefv.");
    }

    return res;
}



gentype derefm(const gentype &a, const gentype &i, const gentype &j)
{
    gentype res;

    if ( a.isValEqnDir() || i.isValEqnDir() || j.isValEqnDir() )
    {
        const static gentype resx("derefm(x,y,z)");
        return resx(a,i,j);
    }

    if ( !(a.isValMatrix()) )
    {
        constructError(a,i,j,res,"derefm works only on matrices.");
    }

    else if ( i.isCastableToIntegerWithoutLoss() && j.isCastableToIntegerWithoutLoss() )
    {
        if ( ( i.cast_int(0) < 0 ) || ( i.cast_int(0) >= a.numRows() ) ||
             ( j.cast_int(0) < 0 ) || ( j.cast_int(0) >= a.numCols() )    )
        {
            constructError(a,i,j,res,"index out of range in derefm.");
        }

        else
        {
            res = (a.cast_matrix(0))(i.cast_int(0),j.cast_int(0));
        }
    }

    else if ( i.isValVector() && j.isCastableToIntegerWithoutLoss() )
    {
        if ( ( j.cast_int(0) < 0 ) || ( j.cast_int(0) >= a.numCols() ) )
        {
            constructError(a,i,j,res,"index out of range in derefm.");
        }

        else
        {
            int k;
            gentype kk;
            Vector<gentype> resv(i.size());

            if ( i.size() )
            {
                for ( k = 0 ; k < i.size() ; k++ )
                {
                    kk = k;
                    resv("&",k) = derefm(a,derefv(i,kk),j);
                }
            }

            res = resv;
        }
    }

    else if ( i.isCastableToIntegerWithoutLoss() && j.isValVector() )
    {
        if ( ( i.cast_int(0) < 0 ) || ( i.cast_int(0) >= a.numCols() ) )
        {
            constructError(a,i,j,res,"index out of range in derefm.");
        }

        else
        {
            int l;
            gentype ll;
            Vector<gentype> resv(j.size());

            if ( i.size() )
            {
                for ( l = 0 ; l < j.size() ; l++ )
                {
                    ll = l;
                    resv("&",l) = derefm(a,i,derefv(j,ll));
                }
            }

            res = resv;
        }
    }

    else if ( i.isValVector() && j.isValVector() )
    {
        int k,l;
        gentype kk,ll;
        Matrix<gentype> resm(i.size(),j.size());

        if ( i.size() && j.size() )
        {
            for ( k = 0 ; k < i.size() ; k++ )
            {
                for ( l = 0 ; l < j.size() ; l++ )
                {
                    kk = k;
                    ll = l;

                    resm("&",k,l) = derefm(a,derefv(i,kk),derefv(j,ll));
                }
            }
        }

        res = resm;
    }

    else
    {
        constructError(a,i,res,"malformed index in derefv.");
    }

    return res;
}



gentype derefa(const gentype &a, const gentype &i)
{
    gentype res;

    if ( a.isValEqnDir() || i.isValEqnDir() )
    {
        const static gentype resx("derefv(x,y)");
        return resx(a,i);
    }

    if ( a.isValVector() && a.infsize() )
    {
        constructError(a,i,res,"derefa can't cast infinite dimensional vector to anion for dereference.");
    }

    else if ( a.isValVector() )
    {
        int k;
        gentype kk;
        Vector<gentype> resv(a.size());

        if ( a.size() )
        {
            for ( k = 0 ; k < a.size() ; k++ )
            {
                kk = k;
                resv("&",k) = derefa(derefv(a,kk),i);
            }
        }

        res = resv;
    }

    else if ( !(a.isCastableToAnionWithoutLoss()) )
    {
        constructError(a,i,res,"derefa works only on anions.");
    }

    else if ( i.isCastableToIntegerWithoutLoss() )
    {
        if ( i.cast_int(0) < 0 )
        {
            constructError(a,i,res,"index out of range in derefa.");
        }

        else
        {
            res = (a.cast_anion(0))(i.cast_int(0));
        }
    }

    else if ( i.isValVector() )
    {
        int k;
        gentype kk;
        Vector<gentype> resv(i.size());

        if ( i.size() )
        {
            for ( k = 0 ; k < i.size() ; k++ )
            {
                kk = k;
                resv("&",k) = derefa(a,derefv(i,kk));
            }
        }

        res = resv;
    }

    else
    {
        constructError(a,i,res,"malformed index in derefa.");
    }

    return res;
}




gentype collapse(const gentype &a)
{
    int i,j,k,l;
    gentype res;

    if ( a.isValMatrix() )
    {
        int badres = 0;
        Vector<int> coldim(a.numCols());
	Vector<int> rowdim(a.numRows());
        Matrix<gentype> aa(a.cast_matrix(0));

	if ( aa.numRows() && aa.numCols() )
	{
	    for ( i = 0 ; ( i < aa.numRows() ) && !badres ; i++ )
	    {
		rowdim("&",i) = aa(i,zeroint()).numRows();

		for ( j = 0 ; j < aa.numCols() ; j++ )
		{
		    coldim("&",j) = aa(zeroint(),j).numCols();

		    if ( ( aa(i,j).numRows() != rowdim(i) ) || ( aa(i,j).numCols() != coldim(j) ) )
		    {
			constructError(a,res,"Element dimensions must agree in collapse.");
                        badres = 1;
                        break;
		    }
		}
	    }

	    if ( !badres )
	    {
		Matrix<gentype> mres(sum(rowdim),sum(coldim));

		k = 0;

                retMatrix<gentype> tmpma;

		for ( i = 0 ; i < aa.numRows() ; i++ )
		{
		    l = 0;

		    for ( j = 0 ; j < aa.numCols() ; j++ )
		    {
			if ( aa(i,j).isValMatrix() || aa(i,j).isValVector() )
			{
			    mres("&",k,1,k+rowdim(i)-1,l,1,l+coldim(j)-1,tmpma) = aa(i,j).cast_matrix(0);
			}

			else
			{
			    mres("&",k,l) = aa(i,j);
			}

			l += coldim(j);
		    }

		    k += rowdim(i);
		}

		res = mres;
	    }
	}

	else
	{
            res = a;
	}
    }

    else if ( a.isValVector() )
    {
        int badres = 0;
        int coldim = -1;
	Vector<int> rowdim(a.size());
        Vector<gentype> aa(a.cast_vector(0));

        if ( aa.infsize() )
        {
            constructError(a,res,"Can't collapse vector of infinite size.");
            badres = 1;
        }

	else if ( aa.size() )
	{
	    for ( i = 0 ; i < aa.size() ; i++ )
	    {
		rowdim("&",i) = aa(i).numRows();
                coldim         = aa(zeroint()).numCols();

		if ( coldim != aa(i).numCols() )
		{
		    constructError(a,res,"Element dimensions must agree in collapse.");
		    badres = 1;
		    break;
		}
	    }

	    if ( !badres )
	    {
		if ( coldim == 1 )
		{
		    Vector<gentype> vres(sum(rowdim));

		    k = 0;

                    retVector<gentype> tmpva;

		    for ( i = 0 ; i < aa.size() ; i++ )
		    {
			if ( aa(i).isValMatrix() || aa(i).isValVector() )
			{
			    vres("&",k,1,k+rowdim(i)-1,tmpva) = aa(i).cast_vector(0);
			}

			else
			{
			    vres("&",k) = aa(i);
			}

			k += rowdim(i);
		    }

                    res = vres;
		}

		else
		{
                    Matrix<gentype> mres(sum(rowdim),coldim);

		    k = 0;

                    retMatrix<gentype> tmpma;

		    for ( i = 0 ; i < aa.size() ; i++ )
		    {
			if ( aa(i).isValMatrix() || aa(i).isValVector() )
			{
			    mres("&",k,1,k+rowdim(i)-1,0,1,coldim-1,tmpma) = aa(i).cast_matrix(0);
			}

			else
			{
			    mres("&",k,0) = aa(k);
			}

			k += rowdim(i);
		    }

                    res = mres;
		}
	    }
	}

	else
	{
            res = a;
	}
    }

    else
    {
	res = a;
    }

    return res;
}






























































gentype fourProd(const gentype &a, const gentype &b, const gentype &c, const gentype &d)
{
    gentype res;

    if ( a.isValEqnDir() || b.isValEqnDir() || c.isValEqnDir() || d.isValEqnDir() )
    {
        const static gentype resx("fourProd(x,y,z,v)");
        return resx(a,b,c,d);
    }

         if ( a.isValStrErr() || b.isValStrErr() || c.isValStrErr() || d.isValStrErr() ) { constructError(a,b,c,d,res,"String 4-product not implemented"); }
    else if ( a.isValSet()    || b.isValSet()    || c.isValSet()    || d.isValSet()    ) { constructError(a,b,c,d,res,"Set 4-product not implemented"); }
    else if ( a.isValDgraph() || b.isValDgraph() || c.isValDgraph() || d.isValDgraph() ) { constructError(a,b,c,d,res,"Dgraph 4-product not implemented"); }
    else if ( a.isValMatrix() || b.isValMatrix() || c.isValMatrix() || d.isValMatrix() ) { constructError(a,b,c,d,res,"Matrix 4-product not implemented"); }
    else if ( a.isValVector() || b.isValVector() || c.isValVector() || d.isValVector() ) { if ( ( a.size() != b.size() ) || ( b.size() != c.size() ) || ( c.size() != d.size() ) ) { constructError(a,b,c,d,res,"Size mismatch in 4-product"); } else { gentype temp; fourProduct(temp,a.cast_vector(0),b.cast_vector(0),c.cast_vector(0),d.cast_vector(0)); res = temp; } }
    else if ( a.isValAnion()  || b.isValAnion()  || c.isValAnion()  || d.isValAnion()  ) { res = (a.cast_anion(0))*(b.cast_anion(0))*(c.cast_anion(0))*(d.cast_anion(0)); }
    else if ( a.isValReal()   || b.isValReal()   || c.isValReal()   || d.isValReal()   ) { res = (a.cast_double(0))*(b.cast_double(0))*(c.cast_double(0))*(d.cast_double(0)); }
    else                                                                                 { res = (a.cast_int(0))*(b.cast_int(0))*(c.cast_int(0))*(d.cast_int(0)); }

    return res;
}

gentype outerProd(const gentype &a, const gentype &b)
{
    gentype res;

    if ( a.isValEqnDir() || b.isValEqnDir() )
    {
        const static gentype resx("outerProd(x,y)");
        return resx(a,b);
    }

         if ( a.isValStrErr() || b.isValStrErr() ) { constructError(a,b,res,"String outer product not implemented"); }
    else if ( a.isValSet()    || b.isValSet()    ) { constructError(a,b,res,"Set outer product not implemented"); }
    else if ( a.isValDgraph() || b.isValDgraph() ) { constructError(a,b,res,"Dgraph outer product not implemented"); }
    else if ( a.isValMatrix() || b.isValMatrix() ) { constructError(a,b,res,"Matrix outer product not implemented"); }
    else if ( a.isValVector() || b.isValVector() ) { res = outerProduct(a.cast_vector(0),b.cast_vector(0)); }
    else if ( a.isValAnion()  || b.isValAnion()  ) { res = (a.cast_anion(0))*conj(b.cast_anion(0)); }
    else if ( a.isValReal()   || b.isValReal()   ) { res = (a.cast_double(0))*(b.cast_double(0)); }
    else                                           { res = (a.cast_int(0))*(b.cast_int(0)); }

    return res;
}

gentype trans(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("trans(x)");
        return resx(a);
    }

    else
    {
	res = a;
        res.transpose();
    }

    return res;
}

gentype det(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("det(x)");
        return resx(a);
    }

         if ( a.isValStrErr() ) { constructError(a,res,"String det not implemented"); }
    else if ( a.isValSet()    ) { constructError(a,res,"Set det not implemented"); }
    else if ( a.isValDgraph() ) { constructError(a,res,"Dgraph det not implemented"); }
    else if ( a.isValMatrix() ) { if ( a.numRows() != a.numCols() ) { constructError(a,res,"Determinant only defined for square matrices."); } else { res = (a.cast_matrix(0)).det(); } }
    else if ( a.isValVector() ) { constructError(a,res,"Vector det not implemented"); }
    else                        { res = a; }

    return res;
}

gentype trace(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("trace(x)");
        return resx(a);
    }

         if ( a.isValStrErr() ) { constructError(a,res,"String trace not implemented"); }
    else if ( a.isValSet()    ) { constructError(a,res,"Set trace not implemented"); }
    else if ( a.isValDgraph() ) { constructError(a,res,"Dgraph trace not implemented"); }
    else if ( a.isValMatrix() ) { if ( a.numRows() != a.numCols() ) { constructError(a,res,"Trace only defined for square matrices."); } else { res = (a.cast_matrix(0)).trace(); } }
    else if ( a.isValVector() ) { constructError(a,res,"Vector trace not implemented"); }
    else                        { res = a; }

    return res;
}

gentype miner(const gentype &a, const gentype &i, const gentype &j)
{
    gentype res;

    if ( a.isValEqnDir() || i.isValEqnDir() || j.isValEqnDir() )
    {
        const static gentype resx("miner(x,y,z)");
        return resx(a,i,j);
    }

         if ( !i.isCastableToIntegerWithoutLoss() ) { constructError(a,i,j,res,"Matrix minor requires integer coefficients"); }
    else if ( !j.isCastableToIntegerWithoutLoss() ) { constructError(a,i,j,res,"Matrix minor requires integer coefficients"); }
    else if ( !a.isValMatrix()                    ) { constructError(a,i,j,res,"Matrix minor only defined for matrices"); }
    else
    {
	if ( a.numRows() != a.numCols() )
	{
	    constructError(a,i,j,res,"miner only defined for square matrices.");
	}

	else if ( !a.numRows() )
	{
	    constructError(a,i,j,res,"miner only defined for non-empty matrices.");
	}

	else if ( ( i.cast_int(0) < 0 ) || ( i.cast_int(0) >= a.numRows() ) )
	{
	    constructError(a,i,j,res,"index out of range for miner.");
	}

	else if ( ( j.cast_int(0) < 0 ) || ( j.cast_int(0) >= a.numRows() ) )
	{
	    constructError(a,i,j,res,"index out of range for miner.");
	}

	else
	{
	    res = (a.cast_matrix(0)).miner(i.cast_int(0),j.cast_int(0));
	}
    }

    return res;
}

gentype cofactor(const gentype &a, const gentype &i, const gentype &j)
{
    gentype res;

    if ( a.isValEqnDir() || i.isValEqnDir() || j.isValEqnDir() )
    {
        const static gentype resx("cofactor(x,y,z)");
        return resx(a,i,j);
    }

         if ( !i.isCastableToIntegerWithoutLoss() ) { constructError(a,i,j,res,"Matrix cofactor requires integer coefficients"); }
    else if ( !j.isCastableToIntegerWithoutLoss() ) { constructError(a,i,j,res,"Matrix cofactor requires integer coefficients"); }
    else if ( !a.isValMatrix()                    ) { constructError(a,i,j,res,"Matrix cofactor only defined for matrices"); }
    else
    {
	if ( a.numRows() != a.numCols() )
	{
	    constructError(a,i,j,res,"cofactor only defined for square matrices.");
	}

	else if ( !a.numRows() )
	{
	    constructError(a,i,j,res,"cofactor only defined for non-empty matrices.");
	}

	else if ( ( i.cast_int(0) < 0 ) || ( i.cast_int(0) >= a.numRows() ) )
	{
	    constructError(a,i,j,res,"index out of range for cofactor.");
	}

	else if ( ( j.cast_int(0) < 0 ) || ( j.cast_int(0) >= a.numRows() ) )
	{
	    constructError(a,i,j,res,"index out of range for cofactor.");
	}

	else
	{
	    res = (a.cast_matrix(0)).cofactor(i.cast_int(0),j.cast_int(0));
	}
    }

    return res;
}

gentype adj(const gentype &a)
{
    gentype res;

    if ( a.isValEqnDir() )
    {
        const static gentype resx("adj(x)");
        return resx(a);
    }

    if ( !a.isValMatrix() ) { constructError(a,res,"adjoint only defined for matrices"); }

    else
    {
	if ( a.numRows() != a.numCols() )
	{
	    constructError(a,res,"adj only defined for square matrices.");
	}

	else if ( !a.numRows() )
	{
	    constructError(a,res,"adj only defined for non-empty matrices.");
	}

	else
	{
	    res = (a.cast_matrix(0)).adj();
	}
    }

    return res;
}







gentype polar (const gentype &x, const gentype &y, const gentype &a) { return div(add(emul(x,exp(emul(y,a))),emul(exp(emul(a,y)),x)),twointgentype()); }
gentype polard(const gentype &x, const gentype &y, const gentype &a) { return div(add(emul(x,exp(emul(y,a))),emul(exp(emul(a,y)),x)),twointgentype()); }
gentype polarx(const gentype &x, const gentype &a)                   { return div(add(emul(x,exp(a)),emul(exp(a),x)),twointgentype());                 }

gentype logb (const gentype &a, const gentype &b) { return mul(halfdblgentype(),add(logbl(a,b),logbr(a,b))); }
gentype logbl(const gentype &a, const gentype &b) { return emul(log(a),einv(log(b)));                        }
gentype logbr(const gentype &a, const gentype &b) { return emul(einv(log(b)),log(a));                        }
gentype Logb (const gentype &a, const gentype &b) { return mul(halfdblgentype(),add(Logbl(a,b),Logbr(a,b))); }
gentype Logbl(const gentype &a, const gentype &b) { return emul(Log(a),einv(Log(b)));                        }
gentype Logbr(const gentype &a, const gentype &b) { return emul(einv(Log(b)),Log(a));                        }

gentype commutate    (const gentype &x, const gentype &y                  ) { return div(add(emul(x,y),neg(emul(y,x))),twointgentype());                 }
gentype anticommutate(const gentype &x, const gentype &y                  ) { return div(add(emul(x,y),    emul(y,x) ),twointgentype());                 }
gentype associate    (const gentype &x, const gentype &y, const gentype &z) { return div(add(emul(x,emul(y,z)),neg(emul(emul(x,y),z))),twointgentype()); }
gentype antiassociate(const gentype &x, const gentype &y, const gentype &z) { return div(add(emul(x,emul(y,z)),    emul(emul(x,y),z) ),twointgentype()); }

// bern(w,x):    returns the Bernstein polynomial of order size(w) (w is a weight vector) evaluated at x
// bernv(w):     returns the Bernstein polynomial *vector* of order size(w) (w is a weight vector)

gentype bern(const gentype &w, const gentype &x)
{
    NiceAssert( w.isCastableToVectorWithoutLoss() );
    NiceAssert( !w.infsize() );

    const Vector<gentype> &ww = w.cast_vector(0);
    int n = ww.size()-1;
    gentype nn(n);
    gentype ov(1.0);
    gentype res;

    if ( n == -1 )
    {
        res = 1.0;
    }

    else
    {
        int j;

        for ( j = 0 ; j <= n ; j++ )
        {
            gentype jj(j);
            gentype tmp;

            tmp = ww(j)*pow(x,jj)*pow(ov-x,nn-jj)*((double) xnCr(n,j));

            if ( !j )
            {
                res = tmp;
            }

            else
            {
                res += tmp;
            }
        }
    }

    return res;
}

gentype funcv(const gentype &f)
{
    // Note that the Functional vector class FuncVector has not actually been defined
    // at this point... however, the basis for *streaming* to such a class has been
    // preempted in vector.h, so we can cheat by creating a string representation 
    // of the result and then streaming that in to a regular vector.

    gentype res;
    Vector<gentype> &rres = res.force_vector(0);

    std::stringstream buffer;

    buffer << "[[ FN f: " << f  << " : 1 ]]";
    buffer >> rres;

    return res;
}

gentype rkhsv(const gentype &k, const gentype &x, const gentype &a)
{
    NiceAssert( x.isCastableToVectorWithoutLoss() );
    NiceAssert( !x.infsize() );

    NiceAssert( a.isCastableToVectorWithoutLoss() );
    NiceAssert( !a.infsize() );

    const Vector<gentype> &xx = x.cast_vector(0);
    const Vector<gentype> &aa = a.cast_vector(0);

    // Note that the RKHS vector class RKHSVector has not actually been defined
    // at this point... however, the basis for *streaming* to such a class has been
    // preempted in vector.h, so we can cheat by creating a string representation 
    // of the result and then streaming that in to a regular vector.

    gentype res;
    Vector<gentype> &rres = res.force_vector(0);

    std::stringstream buffer;

    buffer << "[[ RKHS kernel: " << k.cast_string()  << "\n   RKHS x: " << xx << "\n   RKHS a: " << aa << " ]]";
    buffer >> rres;

    return res;
}

gentype bernv(const gentype &w)
{
    NiceAssert( w.isCastableToVectorWithoutLoss() );
    NiceAssert( !w.infsize() );

    const Vector<gentype> &ww = w.cast_vector(0);

    // Note that the Bernstein vector class BernVector has not actually been defined
    // at this point... however, the basis for *streaming* to such a class has been
    // preempted in vector.h, so we can cheat by creating a string representation 
    // of the result and then streaming that in to a regular vector.

    gentype res;
    Vector<gentype> &rres = res.force_vector(0);

    std::stringstream buffer;

    buffer << "[[ Bern w: " << ww  << " ]]";
    buffer >> rres;

    return res;
}

gentype normDistr(const gentype &x)
{
    gentype res;

    if ( x.isValEqnDir() )
    {
        const static gentype resx("normDistr(x)");
        return resx(x);
    }

         if ( x.isValEqnDir() ) { constructError(x,res,"String normDistr not implemented"); }
    else if ( x.isValSet()    ) { Set<gentype>    temp(x.cast_set(0));    temp.applyon(normDistr); res = temp; }
    else if ( x.isValDgraph() ) { constructError(x,res,"Dgraph normDistr not implemented"); }
    else if ( x.isValMatrix() ) { Matrix<gentype> temp(x.cast_matrix(0)); temp.applyon(normDistr); res = temp; }
    else if ( x.isValVector() ) { Vector<gentype> temp(x.cast_vector(0)); temp.applyon(normDistr); res = temp; }
    else
    {
//        gentype valuea = 0.398942280401;
//        gentype twoval = 2;
//
//        res = mul(valuea,exp(div(mul(neg(x),x),twoval)));
        res = 0.398942280401*exp((-x*x)/2.0);
    }

    return res;
}

gentype polyDistrintern(const gentype &x, const gentype &n);
gentype polyDistrintern(const gentype &x, const gentype &n)
{
    gentype res;

    if ( x.isCastableToAnionWithoutLoss() && n.isCastableToAnionWithoutLoss() )
    {
        const static gentype consa = gamma(div(oneintgentype(),n));
        const static gentype consb = gamma(div(threeintgentype(),n));
        const static gentype consc = sqrt(div(consb,consa));

        res = mul(mul(n,div(consc,(2.0*consa))),exp(mul(neg(pow(consc,n)),pow(x,n))));

//        res = mul(div(mul(n,sqrt(gamma(div(threeintgentype(),n))/gamma(div(oneintgentype(),n)))),mul(twointgentype(),gamma(div(oneintgentype(),n))))),exp(mul(neg(pow(sqrt(gamma(div(threeintgentype(),n))/gamma(div(oneintgentype(),n))),n)),pow(x,n))));
//        res = (mul(n,div(sqrt(div(gamma(div(threeintgentype(),n)),gamma(div(oneintgentype(),n)))),(mul(twointgentype(),gamma(div(oneintgentype(),n)))))))*  exp(mul(-((pow(sqrt(div(gamma(div(threeintgentype(),n)),gamma(div(oneintgentype(),n)))),n)),pow(x,n))));
//        res = (mul(n,div(sqrt(div(gamma(div(threeintgentype(),n)),gamma(div(oneintgentype(),n)))),(mul(twointgentype(),gamma(div(oneintgentype(),n)))))))*exp(mul-((pow(sqrt(div(gamma(div(threeintgentype(),n)),gamma(div(oneintgentype(),n)))),n)),pow(x,n)));
//        res = (mul(n,div(sqrt(div(gamma(div(threeintgentype(),n)),gamma(div(oneintgentype(),n)))),(mul(twointgentype(),gamma(div(oneintgentype(),n)))))))*exp(mul-((sqrt(div(gamma(div(threeintgentype(),n)),gamma(div(oneintgentype(),n))))^n),(x^n)));
//        res = (n*sqrt(gamma(threeintgentype()/n)/gamma(oneintgentype()/n))/(twointgentype()*gamma(oneintgentype()/n)))*exp(-(sqrt(gamma(threeintgentype()/n)/gamma(oneintgentype()/n))^n)*(x^n));
//        res = (n*sqrt(gamma(3/n)/gamma(1/n))/(2*gamma(1/n)))*exp(-(sqrt(gamma(3/n)/gamma(1/n))^n)*(x^n));
    }

    else
    {
        constructError(x,n,res,"polyDistr does not work on strings.");
    }

    return res;
}

gentype PolyDistrintern(const gentype &x, const gentype &n);
gentype PolyDistrintern(const gentype &x, const gentype &n)
{
    gentype res;

    if ( x.isCastableToAnionWithoutLoss() && n.isCastableToAnionWithoutLoss() )
    {
        res = mul(div(mul(n,Sqrt(div(gamma(div(threeintgentype(),n)),gamma(div(oneintgentype(),n))))),mul(twointgentype(),gamma(div(oneintgentype(),n)))),exp(mul(neg(Pow(Sqrt(div(gamma(div(threeintgentype(),n)),gamma(div(oneintgentype(),n)))),n)),Pow(x,n))));
    }

    else
    {
        constructError(x,n,res,"PolyDistr does not work on strings.");
    }

    return res;
}






















































//phantomxyz

gentype pos(const gentype &a) { return  a; }
gentype neg(const gentype &a) { return -a; }

gentype add (const gentype &a, const gentype &b) {                  return a+b;              }
gentype sub (const gentype &a, const gentype &b) {                  return a-b;              }
gentype mul (const gentype &a, const gentype &b) {                  return a*b;              }
gentype div (const gentype &a, const gentype &b) {                  return a/b;              }
gentype mod (const gentype &a, const gentype &b) {                  return a%b;              }
gentype rdiv(const gentype &a, const gentype &b) { gentype temp(b); return temp.rightdiv(a); }

gentype eq(const gentype &a, const gentype &b) { gentype res; return binLogicForm(res,a,b,"eq",operator==,1); }
gentype ne(const gentype &a, const gentype &b) { gentype res; return binLogicForm(res,a,b,"ne",operator!=,0); }
gentype gt(const gentype &a, const gentype &b) { gentype res; return binLogicForm(res,a,b,"gt",operator> ,1); }
gentype ge(const gentype &a, const gentype &b) { gentype res; return binLogicForm(res,a,b,"ge",operator>=,1); }
gentype le(const gentype &a, const gentype &b) { gentype res; return binLogicForm(res,a,b,"le",operator<=,1); }
gentype lt(const gentype &a, const gentype &b) { gentype res; return binLogicForm(res,a,b,"lt",operator< ,1); }

gentype isnull  (const gentype &a) { gentype res = a; return OP_isnull  (res); }
gentype isint   (const gentype &a) { gentype res = a; return OP_isint   (res); }
gentype isreal  (const gentype &a) { gentype res = a; return OP_isreal  (res); }
gentype isanion (const gentype &a) { gentype res = a; return OP_isanion (res); }
gentype isvector(const gentype &a) { gentype res = a; return OP_isvector(res); }
gentype ismatrix(const gentype &a) { gentype res = a; return OP_ismatrix(res); }
gentype isset   (const gentype &a) { gentype res = a; return OP_isset   (res); }
gentype isdgraph(const gentype &a) { gentype res = a; return OP_isdgraph(res); }
gentype isstring(const gentype &a) { gentype res = a; return OP_isstring(res); }
gentype iserror (const gentype &a) { gentype res = a; return OP_iserror (res); }

gentype isvnan(const gentype &a) { gentype res = a; return OP_isvnan(res); }
gentype isinf (const gentype &a) { gentype res = a; return OP_isinf (res); }
gentype ispinf(const gentype &a) { gentype res = a; return OP_ispinf(res); }
gentype isninf(const gentype &a) { gentype res = a; return OP_isninf(res); }

gentype size   (const gentype &a) { gentype res = a; return OP_size   (res); }
gentype numRows(const gentype &a) { gentype res = a; return OP_numRows(res); }
gentype numCols(const gentype &a) { gentype res = a; return OP_numCols(res); }

gentype lnot     (const gentype &a) { gentype res = a; return OP_lnot     (res); }
gentype eabs2    (const gentype &a) { gentype res = a; return OP_eabs2    (res); }
gentype eabs1    (const gentype &a) { gentype res = a; return OP_eabs1    (res); }
gentype eabsinf  (const gentype &a) { gentype res = a; return OP_eabsinf  (res); }
gentype enorm2   (const gentype &a) { gentype res = a; return OP_enorm2   (res); }
gentype enorm1   (const gentype &a) { gentype res = a; return OP_enorm1   (res); }
gentype real     (const gentype &a) { gentype res = a; return OP_real     (res); }
gentype imag     (const gentype &a) { gentype res = a; return OP_imag     (res); }
gentype arg      (const gentype &a) { gentype res = a; return OP_arg      (res); }
gentype eangle   (const gentype &a) { gentype res = a; return OP_eangle   (res); }
gentype einv     (const gentype &a) { gentype res = a; return OP_einv     (res); }
gentype imagd    (const gentype &a) { gentype res = a; return OP_imagd    (res); }
gentype imagx    (const gentype &a) { gentype res = a; return OP_imagx    (res); }
gentype argd     (const gentype &a) { gentype res = a; return OP_argd     (res); }
gentype argx     (const gentype &a) { gentype res = a; return OP_argx     (res); }
gentype Imagd    (const gentype &a) { gentype res = a; return OP_Imagd    (res); }
gentype Argd     (const gentype &a) { gentype res = a; return OP_Argd     (res); }
gentype Argx     (const gentype &a) { gentype res = a; return OP_Argx     (res); }
gentype sgn      (const gentype &a) { gentype res = a; return OP_sgn      (res); }
gentype sqrt     (const gentype &a) { gentype res = a; return OP_sqrt     (res); }
gentype Sqrt     (const gentype &a) { gentype res = a; return OP_Sqrt     (res); }
gentype exp      (const gentype &a) { gentype res = a; return OP_exp      (res); }
gentype tenup    (const gentype &a) { gentype res = a; return OP_tenup    (res); }
gentype log      (const gentype &a) { gentype res = a; return OP_log      (res); }
gentype log10    (const gentype &a) { gentype res = a; return OP_log10    (res); }
gentype Log      (const gentype &a) { gentype res = a; return OP_Log      (res); }
gentype Log10    (const gentype &a) { gentype res = a; return OP_Log10    (res); }
gentype sin      (const gentype &a) { gentype res = a; return OP_sin      (res); }
gentype cos      (const gentype &a) { gentype res = a; return OP_cos      (res); }
gentype tan      (const gentype &a) { gentype res = a; return OP_tan      (res); }
gentype cosec    (const gentype &a) { gentype res = a; return OP_cosec    (res); }
gentype sec      (const gentype &a) { gentype res = a; return OP_sec      (res); }
gentype cot      (const gentype &a) { gentype res = a; return OP_cot      (res); }
gentype vers     (const gentype &a) { gentype res = a; return OP_vers     (res); }
gentype covers   (const gentype &a) { gentype res = a; return OP_covers   (res); }
gentype hav      (const gentype &a) { gentype res = a; return OP_hav      (res); }
gentype excosec  (const gentype &a) { gentype res = a; return OP_excosec  (res); }
gentype exsec    (const gentype &a) { gentype res = a; return OP_exsec    (res); }
gentype asin     (const gentype &a) { gentype res = a; return OP_asin     (res); }
gentype acos     (const gentype &a) { gentype res = a; return OP_acos     (res); }
gentype Asin     (const gentype &a) { gentype res = a; return OP_Asin     (res); }
gentype Acos     (const gentype &a) { gentype res = a; return OP_Acos     (res); }
gentype atan     (const gentype &a) { gentype res = a; return OP_atan     (res); }
gentype acosec   (const gentype &a) { gentype res = a; return OP_acosec   (res); }
gentype asec     (const gentype &a) { gentype res = a; return OP_asec     (res); }
gentype Acosec   (const gentype &a) { gentype res = a; return OP_Acosec   (res); }
gentype Asec     (const gentype &a) { gentype res = a; return OP_Asec     (res); }
gentype acot     (const gentype &a) { gentype res = a; return OP_acot     (res); }
gentype avers    (const gentype &a) { gentype res = a; return OP_avers    (res); }
gentype acovers  (const gentype &a) { gentype res = a; return OP_acovers  (res); }
gentype ahav     (const gentype &a) { gentype res = a; return OP_ahav     (res); }
gentype aexcosec (const gentype &a) { gentype res = a; return OP_aexcosec (res); }
gentype aexsec   (const gentype &a) { gentype res = a; return OP_aexsec   (res); }
gentype Avers    (const gentype &a) { gentype res = a; return OP_Avers    (res); }
gentype Acovers  (const gentype &a) { gentype res = a; return OP_Acovers  (res); }
gentype Ahav     (const gentype &a) { gentype res = a; return OP_Ahav     (res); }
gentype Aexcosec (const gentype &a) { gentype res = a; return OP_Aexcosec (res); }
gentype Aexsec   (const gentype &a) { gentype res = a; return OP_Aexsec   (res); }
gentype sinc     (const gentype &a) { gentype res = a; return OP_sinc     (res); }
gentype cosc     (const gentype &a) { gentype res = a; return OP_cosc     (res); }
gentype tanc     (const gentype &a) { gentype res = a; return OP_tanc     (res); }
gentype sinh     (const gentype &a) { gentype res = a; return OP_sinh     (res); }
gentype cosh     (const gentype &a) { gentype res = a; return OP_cosh     (res); }
gentype tanh     (const gentype &a) { gentype res = a; return OP_tanh     (res); }
gentype cosech   (const gentype &a) { gentype res = a; return OP_cosech   (res); }
gentype sech     (const gentype &a) { gentype res = a; return OP_sech     (res); }
gentype coth     (const gentype &a) { gentype res = a; return OP_coth     (res); }
gentype versh    (const gentype &a) { gentype res = a; return OP_versh    (res); }
gentype coversh  (const gentype &a) { gentype res = a; return OP_coversh  (res); }
gentype havh     (const gentype &a) { gentype res = a; return OP_havh     (res); }
gentype excosech (const gentype &a) { gentype res = a; return OP_excosech (res); }
gentype exsech   (const gentype &a) { gentype res = a; return OP_exsech   (res); }
gentype asinh    (const gentype &a) { gentype res = a; return OP_asinh    (res); }
gentype acosh    (const gentype &a) { gentype res = a; return OP_acosh    (res); }
gentype atanh    (const gentype &a) { gentype res = a; return OP_atanh    (res); }
gentype Acosh    (const gentype &a) { gentype res = a; return OP_Acosh    (res); }
gentype Atanh    (const gentype &a) { gentype res = a; return OP_Atanh    (res); }
gentype acosech  (const gentype &a) { gentype res = a; return OP_acosech  (res); }
gentype asech    (const gentype &a) { gentype res = a; return OP_asech    (res); }
gentype acoth    (const gentype &a) { gentype res = a; return OP_acoth    (res); }
gentype aversh   (const gentype &a) { gentype res = a; return OP_aversh   (res); }
gentype acovrsh  (const gentype &a) { gentype res = a; return OP_acovrsh  (res); }
gentype ahavh    (const gentype &a) { gentype res = a; return OP_ahavh    (res); }
gentype aexcosech(const gentype &a) { gentype res = a; return OP_aexcosech(res); }
gentype aexsech  (const gentype &a) { gentype res = a; return OP_aexsech  (res); }
gentype Asech    (const gentype &a) { gentype res = a; return OP_Asech    (res); }
gentype Acoth    (const gentype &a) { gentype res = a; return OP_Acoth    (res); }
gentype Aversh   (const gentype &a) { gentype res = a; return OP_Aversh   (res); }
gentype Ahavh    (const gentype &a) { gentype res = a; return OP_Ahavh    (res); }
gentype Aexsech  (const gentype &a) { gentype res = a; return OP_Aexsech  (res); }
gentype sinhc    (const gentype &a) { gentype res = a; return OP_sinhc    (res); }
gentype coshc    (const gentype &a) { gentype res = a; return OP_coshc    (res); }
gentype tanhc    (const gentype &a) { gentype res = a; return OP_tanhc    (res); }
gentype sigm     (const gentype &a) { gentype res = a; return OP_sigm     (res); }
gentype gd       (const gentype &a) { gentype res = a; return OP_gd       (res); }
gentype asigm    (const gentype &a) { gentype res = a; return OP_asigm    (res); }
gentype agd      (const gentype &a) { gentype res = a; return OP_agd      (res); }
gentype Asigm    (const gentype &a) { gentype res = a; return OP_Asigm    (res); }
gentype Agd      (const gentype &a) { gentype res = a; return OP_Agd      (res); }


gentype eabsp    (const gentype &a, const gentype &q) { gentype res; return elementwiseDefaultCallB(res,a,q,"eabsp"    ,&dubeabsp    ,&absp           ); }
gentype enormp   (const gentype &a, const gentype &q) { gentype res; return elementwiseDefaultCallB(res,a,q,"enormp"   ,&dubenormp   ,&normp          ); }
gentype polyDistr(const gentype &a, const gentype &q) { gentype res; return elementwiseDefaultCallB(res,a,q,"polyDistr",&dubpolyDistr,&polyDistrintern); }
gentype PolyDistr(const gentype &a, const gentype &q) { gentype res; return elementwiseDefaultCallB(res,a,q,"PolyDistr",&dubPolyDistr,&PolyDistrintern); }

gentype emul (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"emul" ,emul ,mul ,NULL ); }
gentype ediv (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"ediv" ,ediv ,div ,NULL ); }
gentype eidiv(const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"eidiv",eidiv,idiv,NULL ); }
gentype erdiv(const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"erdiv",erdiv,rdiv,NULL ); }
gentype emod (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"emod" ,emod ,mod ,NULL ); }
gentype epow (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"epow" ,epow ,pow ,NULL ); }
gentype Epow (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"Epow" ,Epow ,Pow ,NULL ); }
gentype epowl(const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"epowl",epowl,powl,NULL ); }
gentype Epowl(const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"Epowl",Epowl,Powl,NULL ); }
gentype epowr(const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"epowr",epowr,powr,NULL ); }
gentype Epowr(const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"Epowr",Epowr,Powr,NULL ); }
gentype eeq  (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"eeq"  ,eeq  ,eq  ,NULL ); }
gentype ene  (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"ene"  ,ene  ,ne  ,NULL ); }
gentype egt  (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"egt"  ,egt  ,gt  ,NULL ); }
gentype ege  (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"ege"  ,ege  ,ge  ,NULL ); }
gentype ele  (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"ele"  ,ele  ,le  ,NULL ); }
gentype elt  (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"elt"  ,elt  ,lt  ,NULL ); }
gentype lor  (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"lor"  ,lor  ,NULL,orOR ); }
gentype land (const gentype &a, const gentype &b) { gentype res; return elementwiseDefaultCallC(res,a,b,"land" ,land ,NULL,andOR); }

gentype max       (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"max"          ,max       ,max   ,NULL,max   ,i,j,0,0,0); }
gentype min       (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"min"          ,min       ,min   ,NULL,min   ,i,j,0,0,0); }
gentype maxabs    (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"maxabs"       ,maxabs    ,maxabs,NULL,maxabs,i,j,0,1,0); }
gentype minabs    (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"minabs"       ,minabs    ,minabs,NULL,minabs,i,j,0,1,0); }
gentype maxdiag   (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"maxdiag"      ,maxdiag   ,NULL  ,NULL,NULL  ,i,j,0,0,0); }
gentype mindiag   (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"mindiag"      ,mindiag   ,NULL  ,NULL,NULL  ,i,j,0,0,0); }
gentype maxabsdiag(const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"maxabsdiag"   ,maxabsdiag,NULL  ,NULL,NULL  ,i,j,0,1,0); }
gentype minabsdiag(const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"minabsdiag"   ,minabsdiag,NULL  ,NULL,NULL  ,i,j,0,1,0); }

gentype argmax       (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmax"       ,max       ,max   ,NULL,NULL,i,j,1,0,0); }
gentype argmin       (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmin"       ,min       ,min   ,NULL,NULL,i,j,1,0,0); }
gentype argmaxabs    (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"argmaxabs"    ,maxabs    ,maxabs,NULL,NULL,i,j,1,1,0); }
gentype argminabs    (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"argminabs"    ,minabs    ,minabs,NULL,NULL,i,j,1,1,0); }
gentype argmaxdiag   (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmaxdiag"   ,maxdiag   ,NULL  ,NULL,NULL,i,j,1,0,0); }
gentype argmindiag   (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmindiag"   ,mindiag   ,NULL  ,NULL,NULL,i,j,1,0,0); }
gentype argmaxabsdiag(const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"argmaxabsdiag",maxabsdiag,NULL  ,NULL,NULL,i,j,1,1,0); }
gentype argminabsdiag(const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"argminabsdiag",minabsdiag,NULL  ,NULL,NULL,i,j,1,1,0); }

gentype allargmax       (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmax"       ,max       ,max   ,NULL,NULL,i,j,1,0,1); }
gentype allargmin       (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmin"       ,min       ,min   ,NULL,NULL,i,j,1,0,1); }
gentype allargmaxabs    (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"argmaxabs"    ,maxabs    ,maxabs,NULL,NULL,i,j,1,1,1); }
gentype allargminabs    (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"argminabs"    ,minabs    ,minabs,NULL,NULL,i,j,1,1,1); }
gentype allargmaxdiag   (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmaxdiag"   ,maxdiag   ,NULL  ,NULL,NULL,i,j,1,0,1); }
gentype allargmindiag   (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmindiag"   ,mindiag   ,NULL  ,NULL,NULL,i,j,1,0,1); }
gentype allargmaxabsdiag(const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"argmaxabsdiag",maxabsdiag,NULL  ,NULL,NULL,i,j,1,1,1); }
gentype allargminabsdiag(const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"argminabsdiag",minabsdiag,NULL  ,NULL,NULL,i,j,1,1,1); }

gentype sum      (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"sum"      ,NULL,NULL  ,sum ,sum   ,i,j,0,0,0); }
gentype prod     (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"prod"     ,NULL,NULL  ,prod,prod  ,i,j,0,0,0); }
gentype Prod     (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"Prod"     ,NULL,NULL  ,Prod,Prod  ,i,j,0,0,0); }
gentype mean     (const gentype &a) { gentype res; int i,j; return maxmincommonform(res,a,"mean"     ,NULL,NULL  ,mean,mean  ,i,j,0,0,0); }
gentype median   (const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"median"   ,NULL,median,NULL,median,i,j,0,0,0); }
gentype argmedian(const gentype &a) { gentype res; int i,j; return zaxmincommonform(res,a,"argmedian",NULL,median,NULL,NULL  ,i,j,1,0,0); }

gentype pow  (const gentype &a, const gentype &b) { return powintern(a,b);  }
gentype powl (const gentype &a, const gentype &b) { return powlintern(a,b); }
gentype powr (const gentype &a, const gentype &b) { return powrintern(a,b); }
gentype Pow  (const gentype &a, const gentype &b) { return Powintern(a,b);  }
gentype Powl (const gentype &a, const gentype &b) { return Powlintern(a,b); }
gentype Powr (const gentype &a, const gentype &b) { return Powrintern(a,b); }































































gentype &OP_pos  (gentype &a) { return a.posate(); }
gentype &OP_neg  (gentype &a) { return a.negate(); }

gentype &OP_add  (gentype &a, const gentype &b) { return a += b;        }
gentype &OP_sub  (gentype &a, const gentype &b) { return a -= b;        }
gentype &OP_mul  (gentype &a, const gentype &b) { return a *= b;        }
gentype &OP_div  (gentype &a, const gentype &b) { return a /= b;        }
gentype &OP_rdiv (gentype &a, const gentype &b) { return a = rdiv(a,b); }
gentype &OP_mod  (gentype &a, const gentype &b) { return a = a%b;       }

gentype &OP_lnot     (gentype &a) { const static gentype fnbare("lnot(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"lnot"     ,&OP_lnot     ,NULL      ,NULL   ,NULL      ,&invertOR,&falseOR   ); }
gentype &OP_eabs2    (gentype &a) { const static gentype fnbare("eabs2(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"eabs2"    ,&OP_eabs2    ,NULL      ,&abs2  ,&abs2     ,&abs2    ,&falseOR   ); }
gentype &OP_eabs1    (gentype &a) { const static gentype fnbare("eabs1(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"eabs1"    ,&OP_eabs1    ,NULL      ,&abs1  ,&abs1     ,&abs1    ,&falseOR   ); }
gentype &OP_eabsinf  (gentype &a) { const static gentype fnbare("eabsinf(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"eabsinf"  ,&OP_eabsinf  ,NULL      ,&absinf,&absinf   ,&absinf  ,&falseOR   ); }
gentype &OP_enorm2   (gentype &a) { const static gentype fnbare("enorm2(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"enorm2"   ,&OP_enorm2   ,NULL      ,&norm2 ,&norm2    ,&norm2   ,&falseOR   ); }
gentype &OP_enorm1   (gentype &a) { const static gentype fnbare("enorm1(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"enorm1"   ,&OP_enorm1   ,NULL      ,&norm1 ,&norm1    ,&norm1   ,&falseOR   ); }
gentype &OP_real     (gentype &a) { const static gentype fnbare("real(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"real"     ,&OP_real     ,NULL      ,&real  ,&real     ,&real    ,&falseOR   ); }
gentype &OP_imag     (gentype &a) { const static gentype fnbare("imag(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"imag"     ,&OP_imag     ,NULL      ,&imag  ,&imag     ,&imag    ,&falseOR   ); }
gentype &OP_arg      (gentype &a) { const static gentype fnbare("arg(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"arg"      ,&OP_arg      ,NULL      ,&arg   ,&arg      ,NULL     ,&falseOR   ); }
gentype &OP_eangle   (gentype &a) { const static gentype fnbare("eangle(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"eangle"   ,&OP_eangle   ,&angle    ,NULL   ,&angle    ,NULL     ,&falseOR   ); }
gentype &OP_einv     (gentype &a) { const static gentype fnbare("einv(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"einv"     ,&OP_einv     ,&inv      ,NULL   ,&inv      ,NULL     ,&falseOR   ); }
gentype &OP_imagd    (gentype &a) { const static gentype fnbare("imagd(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"imagd"    ,&OP_imagd    ,&imagd    ,NULL   ,NULL      ,NULL     ,&trueOR    ); }
gentype &OP_imagx    (gentype &a) { const static gentype fnbare("imagx(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"imagx"    ,&OP_imagx    ,&imagx    ,NULL   ,&imagx    ,&imagx   ,&falseOR   ); }
gentype &OP_argd     (gentype &a) { const static gentype fnbare("argd(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"argd"     ,&OP_argd     ,&argd     ,NULL   ,NULL      ,NULL     ,&trueOR    ); }
gentype &OP_argx     (gentype &a) { const static gentype fnbare("argx(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"argx"     ,&OP_argx     ,&argx     ,NULL   ,NULL      ,NULL     ,&trueOR    ); }
gentype &OP_Imagd    (gentype &a) { const static gentype fnbare("Imagd(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Imagd"    ,&OP_Imagd    ,&Imagd    ,NULL   ,NULL      ,NULL     ,&trueOR    ); }
gentype &OP_Argd     (gentype &a) { const static gentype fnbare("Argd(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"Argd"     ,&OP_Argd     ,&Argd     ,NULL   ,NULL      ,NULL     ,&trueOR    ); }
gentype &OP_Argx     (gentype &a) { const static gentype fnbare("Argx(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"Argx"     ,&OP_Argx     ,&Argx     ,NULL   ,NULL      ,NULL     ,&trueOR    ); }
gentype &OP_sgn      (gentype &a) { const static gentype fnbare("sgn(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"sgn"      ,&OP_sgn      ,&sgn      ,NULL   ,&sgn      ,&sgn     ,&falseOR   ); }
gentype &OP_sqrt     (gentype &a) { const static gentype fnbare("sqrt(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"sqrt"     ,&OP_sqrt     ,&sqrt     ,NULL   ,&sqrt     ,NULL     ,&sqrtOR    ); }
gentype &OP_Sqrt     (gentype &a) { const static gentype fnbare("Sqrt(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"Sqrt"     ,&OP_Sqrt     ,&Sqrt     ,NULL   ,&sqrt     ,NULL     ,&sqrtOR    ); }
gentype &OP_exp      (gentype &a) { const static gentype fnbare("exp(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"exp"      ,&OP_exp      ,&exp      ,NULL   ,&exp      ,NULL     ,&falseOR   ); }
gentype &OP_tenup    (gentype &a) { const static gentype fnbare("tenup(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"tenup"    ,&OP_tenup    ,&tenup    ,NULL   ,&tenup    ,NULL     ,&falseOR   ); }
gentype &OP_log      (gentype &a) { const static gentype fnbare("log(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"log"      ,&OP_log      ,&log      ,NULL   ,&log      ,NULL     ,&logOR     ); }
gentype &OP_log10    (gentype &a) { const static gentype fnbare("log10(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"log10"    ,&OP_log10    ,&log10    ,NULL   ,&log10    ,NULL     ,&log10OR   ); }
gentype &OP_Log      (gentype &a) { const static gentype fnbare("Log(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"Log"      ,&OP_Log      ,&Log      ,NULL   ,&log      ,NULL     ,&logOR     ); }
gentype &OP_Log10    (gentype &a) { const static gentype fnbare("Log10(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Log10"    ,&OP_Log10    ,&Log10    ,NULL   ,&log10    ,NULL     ,&log10OR   ); }
gentype &OP_sin      (gentype &a) { const static gentype fnbare("sin(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"sin"      ,&OP_sin      ,&sin      ,NULL   ,&sin      ,NULL     ,&falseOR   ); }
gentype &OP_cos      (gentype &a) { const static gentype fnbare("cos(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"cos"      ,&OP_cos      ,&cos      ,NULL   ,&cos      ,NULL     ,&falseOR   ); }
gentype &OP_tan      (gentype &a) { const static gentype fnbare("tan(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"tan"      ,&OP_tan      ,&tan      ,NULL   ,&tan      ,NULL     ,&falseOR   ); }
gentype &OP_cosec    (gentype &a) { const static gentype fnbare("cosec(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"cosec"    ,&OP_cosec    ,&cosec    ,NULL   ,&cosec    ,NULL     ,&falseOR   ); }
gentype &OP_sec      (gentype &a) { const static gentype fnbare("sec(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"sec"      ,&OP_sec      ,&sec      ,NULL   ,&sec      ,NULL     ,&falseOR   ); }
gentype &OP_cot      (gentype &a) { const static gentype fnbare("cot(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"cot"      ,&OP_cot      ,&cot      ,NULL   ,&cot      ,NULL     ,&falseOR   ); }
gentype &OP_vers     (gentype &a) { const static gentype fnbare("vers(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"vers"     ,&OP_vers     ,&vers     ,NULL   ,&vers     ,NULL     ,&falseOR   ); }
gentype &OP_covers   (gentype &a) { const static gentype fnbare("covers(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"covers"   ,&OP_covers   ,&covers   ,NULL   ,&covers   ,NULL     ,&falseOR   ); }
gentype &OP_hav      (gentype &a) { const static gentype fnbare("hav(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"hav"      ,&OP_hav      ,&hav      ,NULL   ,&hav      ,NULL     ,&falseOR   ); }
gentype &OP_excosec  (gentype &a) { const static gentype fnbare("excosec(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"excosec"  ,&OP_excosec  ,&excosec  ,NULL   ,&excosec  ,NULL     ,&falseOR   ); }
gentype &OP_exsec    (gentype &a) { const static gentype fnbare("exsec(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"exsec"    ,&OP_exsec    ,&exsec    ,NULL   ,&exsec    ,NULL     ,&falseOR   ); }
gentype &OP_asin     (gentype &a) { const static gentype fnbare("asin(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"asin"     ,&OP_asin     ,&asin     ,NULL   ,&asin     ,NULL     ,&asinOR    ); }
gentype &OP_acos     (gentype &a) { const static gentype fnbare("acos(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"acos"     ,&OP_acos     ,&acos     ,NULL   ,&acos     ,NULL     ,&acosOR    ); }
gentype &OP_Asin     (gentype &a) { const static gentype fnbare("Asin(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"Asin"     ,&OP_Asin     ,&Asin     ,NULL   ,&asin     ,NULL     ,&asinOR    ); }
gentype &OP_Acos     (gentype &a) { const static gentype fnbare("Acos(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"Acos"     ,&OP_Acos     ,&Acos     ,NULL   ,&acos     ,NULL     ,&acosOR    ); }
gentype &OP_atan     (gentype &a) { const static gentype fnbare("atan(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"atan"     ,&OP_atan     ,&atan     ,NULL   ,&atan     ,NULL     ,&falseOR   ); }
gentype &OP_acosec   (gentype &a) { const static gentype fnbare("acosec(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"acosec"   ,&OP_acosec   ,&acosec   ,NULL   ,&acosec   ,NULL     ,&acosecOR  ); }
gentype &OP_asec     (gentype &a) { const static gentype fnbare("asec(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"asec"     ,&OP_asec     ,&asec     ,NULL   ,&asec     ,NULL     ,&asecOR    ); }
gentype &OP_Acosec   (gentype &a) { const static gentype fnbare("Acosec(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"Acosec"   ,&OP_Acosec   ,&Acosec   ,NULL   ,&acosec   ,NULL     ,&acosecOR  ); }
gentype &OP_Asec     (gentype &a) { const static gentype fnbare("Asec(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"Asec"     ,&OP_Asec     ,&Asec     ,NULL   ,&asec     ,NULL     ,&asecOR    ); }
gentype &OP_acot     (gentype &a) { const static gentype fnbare("acot(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"acot"     ,&OP_acot     ,&acot     ,NULL   ,&acot     ,NULL     ,&falseOR   ); }
gentype &OP_avers    (gentype &a) { const static gentype fnbare("avers(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"avers"    ,&OP_avers    ,&avers    ,NULL   ,&avers    ,NULL     ,&aversOR   ); }
gentype &OP_acovers  (gentype &a) { const static gentype fnbare("acovers(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"acovers"  ,&OP_acovers  ,&acovers  ,NULL   ,&acovers  ,NULL     ,&acoversOR ); }
gentype &OP_ahav     (gentype &a) { const static gentype fnbare("ahav(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"ahav"     ,&OP_ahav     ,&ahav     ,NULL   ,&ahav     ,NULL     ,&ahavOR    ); }
gentype &OP_aexcosec (gentype &a) { const static gentype fnbare("aexcosec(x)");  return OP_elementwiseDefaultCallA(a,fnbare,"aexcosec" ,&OP_aexcosec ,&aexcosec ,NULL   ,&aexcosec ,NULL     ,&aexcosecOR); }
gentype &OP_aexsec   (gentype &a) { const static gentype fnbare("aexsec(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"aexsec"   ,&OP_aexsec   ,&aexsec   ,NULL   ,&aexsec   ,NULL     ,&aexsecOR  ); }
gentype &OP_Avers    (gentype &a) { const static gentype fnbare("Avers(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Avers"    ,&OP_Avers    ,&Avers    ,NULL   ,&avers    ,NULL     ,&aversOR   ); }
gentype &OP_Acovers  (gentype &a) { const static gentype fnbare("Acovers(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"Acovers"  ,&OP_Acovers  ,&Acovers  ,NULL   ,&acovers  ,NULL     ,&acoversOR ); }
gentype &OP_Ahav     (gentype &a) { const static gentype fnbare("Ahav(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"Ahav"     ,&OP_Ahav     ,&Ahav     ,NULL   ,&ahav     ,NULL     ,&ahavOR    ); }
gentype &OP_Aexcosec (gentype &a) { const static gentype fnbare("Aexcosec(x)");  return OP_elementwiseDefaultCallA(a,fnbare,"Aexcosec" ,&OP_Aexcosec ,&Aexcosec ,NULL   ,&aexcosec ,NULL     ,&aexcosecOR); }
gentype &OP_Aexsec   (gentype &a) { const static gentype fnbare("Aexsec(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"Aexsec"   ,&OP_Aexsec   ,&Aexsec   ,NULL   ,&aexsec   ,NULL     ,&aexsecOR  ); }
gentype &OP_sinc     (gentype &a) { const static gentype fnbare("sinc(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"sinc"     ,&OP_sinc     ,&sinc     ,NULL   ,&sinc     ,NULL     ,&falseOR   ); }
gentype &OP_cosc     (gentype &a) { const static gentype fnbare("cosc(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"cosc"     ,&OP_cosc     ,&cosc     ,NULL   ,&cosc     ,NULL     ,&falseOR   ); }
gentype &OP_tanc     (gentype &a) { const static gentype fnbare("tanc(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"tanc"     ,&OP_tanc     ,&tanh     ,NULL   ,&tanc     ,NULL     ,&falseOR   ); }
gentype &OP_sinh     (gentype &a) { const static gentype fnbare("sinh(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"sinh"     ,&OP_sinh     ,&sinh     ,NULL   ,&sinh     ,NULL     ,&falseOR   ); }
gentype &OP_cosh     (gentype &a) { const static gentype fnbare("cosh(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"cosh"     ,&OP_cosh     ,&cosh     ,NULL   ,&cosh     ,NULL     ,&falseOR   ); }
gentype &OP_tanh     (gentype &a) { const static gentype fnbare("tanh(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"tanh"     ,&OP_tanh     ,&tanh     ,NULL   ,&tanh     ,NULL     ,&falseOR   ); }
gentype &OP_cosech   (gentype &a) { const static gentype fnbare("cosech(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"cosech"   ,&OP_cosech   ,&cosech   ,NULL   ,&cosech   ,NULL     ,&falseOR   ); }
gentype &OP_sech     (gentype &a) { const static gentype fnbare("sech(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"sech"     ,&OP_sech     ,&sech     ,NULL   ,&sech     ,NULL     ,&falseOR   ); }
gentype &OP_coth     (gentype &a) { const static gentype fnbare("coth(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"coth"     ,&OP_coth     ,&coth     ,NULL   ,&coth     ,NULL     ,&falseOR   ); }
gentype &OP_versh    (gentype &a) { const static gentype fnbare("versh(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"versh"    ,&OP_versh    ,&versh    ,NULL   ,&versh    ,NULL     ,&falseOR   ); }
gentype &OP_coversh  (gentype &a) { const static gentype fnbare("coversh(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"coversh"  ,&OP_coversh  ,&coversh  ,NULL   ,&coversh  ,NULL     ,&falseOR   ); }
gentype &OP_havh     (gentype &a) { const static gentype fnbare("havh(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"havh"     ,&OP_havh     ,&havh     ,NULL   ,&havh     ,NULL     ,&falseOR   ); }
gentype &OP_excosech (gentype &a) { const static gentype fnbare("excosech(x)");  return OP_elementwiseDefaultCallA(a,fnbare,"excosech" ,&OP_excosech ,&excosech ,NULL   ,&excosech ,NULL     ,&falseOR   ); }
gentype &OP_exsech   (gentype &a) { const static gentype fnbare("exsech(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"exsech"   ,&OP_exsech   ,&exsech   ,NULL   ,&exsech   ,NULL     ,&falseOR   ); }
gentype &OP_asinh    (gentype &a) { const static gentype fnbare("asinh(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"asinh"    ,&OP_asinh    ,&asinh    ,NULL   ,&asinh    ,NULL     ,&falseOR   ); }
gentype &OP_acosh    (gentype &a) { const static gentype fnbare("acosh(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"acosh"    ,&OP_acosh    ,&acosh    ,NULL   ,&acosh    ,NULL     ,&acoshOR   ); }
gentype &OP_atanh    (gentype &a) { const static gentype fnbare("atanh(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"atanh"    ,&OP_atanh    ,&atanh    ,NULL   ,&atanh    ,NULL     ,&atanhOR   ); }
gentype &OP_Acosh    (gentype &a) { const static gentype fnbare("Acosh(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Acosh"    ,&OP_Acosh    ,&Acosh    ,NULL   ,&acosh    ,NULL     ,&acoshOR   ); }
gentype &OP_Atanh    (gentype &a) { const static gentype fnbare("Atanh(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Atanh"    ,&OP_Atanh    ,&Atanh    ,NULL   ,&atanh    ,NULL     ,&atanhOR   ); }
gentype &OP_acosech  (gentype &a) { const static gentype fnbare("acosech(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"acosech"  ,&OP_acosech  ,&acosech  ,NULL   ,&acosech  ,NULL     ,&falseOR   ); }
gentype &OP_asech    (gentype &a) { const static gentype fnbare("asech(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"asech"    ,&OP_asech    ,&asech    ,NULL   ,&asech    ,NULL     ,&asechOR   ); }
gentype &OP_acoth    (gentype &a) { const static gentype fnbare("acoth(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"acoth"    ,&OP_acoth    ,&acoth    ,NULL   ,&acoth    ,NULL     ,&acothOR   ); }
gentype &OP_aversh   (gentype &a) { const static gentype fnbare("aversh(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"aversh"   ,&OP_aversh   ,&aversh   ,NULL   ,&aversh   ,NULL     ,&avershOR  ); }
gentype &OP_acovrsh  (gentype &a) { const static gentype fnbare("acovrsh(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"acovrsh"  ,&OP_acovrsh  ,&acovrsh  ,NULL   ,&acovrsh  ,NULL     ,&falseOR   ); }
gentype &OP_ahavh    (gentype &a) { const static gentype fnbare("ahavh(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"ahavh"    ,&OP_ahavh    ,&ahavh    ,NULL   ,&ahavh    ,NULL     ,&ahavhOR   ); }
gentype &OP_aexcosech(gentype &a) { const static gentype fnbare("aexcosech(x)"); return OP_elementwiseDefaultCallA(a,fnbare,"aexcosech",&OP_aexcosech,&aexcosech,NULL   ,&aexcosech,NULL     ,&falseOR   ); }
gentype &OP_aexsech  (gentype &a) { const static gentype fnbare("aexsech(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"aexsech"  ,&OP_aexsech  ,&aexsech  ,NULL   ,&aexsech  ,NULL     ,&aexsechOR ); }
gentype &OP_Asech    (gentype &a) { const static gentype fnbare("Asech(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Asech"    ,&OP_Asech    ,&Asech    ,NULL   ,&asech    ,NULL     ,&asechOR   ); }
gentype &OP_Acoth    (gentype &a) { const static gentype fnbare("Acoth(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Acoth"    ,&OP_Acoth    ,&Acoth    ,NULL   ,&acoth    ,NULL     ,&acothOR   ); }
gentype &OP_Aversh   (gentype &a) { const static gentype fnbare("Aversh(x)");    return OP_elementwiseDefaultCallA(a,fnbare,"Aversh"   ,&OP_Aversh   ,&Aversh   ,NULL   ,&aversh   ,NULL     ,&avershOR  ); }
gentype &OP_Ahavh    (gentype &a) { const static gentype fnbare("Ahavh(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Ahavh"    ,&OP_Ahavh    ,&Ahavh    ,NULL   ,&ahavh    ,NULL     ,&ahavhOR   ); }
gentype &OP_Aexsech  (gentype &a) { const static gentype fnbare("Aexsech(x)");   return OP_elementwiseDefaultCallA(a,fnbare,"Aexsech"  ,&OP_Aexsech  ,&Aexsech  ,NULL   ,&aexsech  ,NULL     ,&aexsechOR ); }
gentype &OP_sinhc    (gentype &a) { const static gentype fnbare("sinhc(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"sinhc"    ,&OP_sinhc    ,&sinhc    ,NULL   ,&sinhc    ,NULL     ,&falseOR   ); }
gentype &OP_coshc    (gentype &a) { const static gentype fnbare("coshc(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"coshc"    ,&OP_coshc    ,&coshc    ,NULL   ,&coshc    ,NULL     ,&falseOR   ); }
gentype &OP_tanhc    (gentype &a) { const static gentype fnbare("tanhc(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"tanhc"    ,&OP_tanhc    ,&tanhc    ,NULL   ,&tanhc    ,NULL     ,&falseOR   ); }
gentype &OP_sigm     (gentype &a) { const static gentype fnbare("sigm(x)");      return OP_elementwiseDefaultCallA(a,fnbare,"sigm"     ,&OP_sigm     ,&sigm     ,NULL   ,&sigm     ,NULL     ,&falseOR   ); }
gentype &OP_gd       (gentype &a) { const static gentype fnbare("gd(x)");        return OP_elementwiseDefaultCallA(a,fnbare,"gd"       ,&OP_gd       ,&gd       ,NULL   ,&gd       ,NULL     ,&falseOR   ); }
gentype &OP_asigm    (gentype &a) { const static gentype fnbare("asigm(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"asigm"    ,&OP_asigm    ,&asigm    ,NULL   ,&asigm    ,NULL     ,&asigmOR   ); }
gentype &OP_agd      (gentype &a) { const static gentype fnbare("agd(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"agd"      ,&OP_agd      ,&agd      ,NULL   ,&agd      ,NULL     ,&agdOR     ); }
gentype &OP_Asigm    (gentype &a) { const static gentype fnbare("Asigm(x)");     return OP_elementwiseDefaultCallA(a,fnbare,"Asigm"    ,&OP_Asigm    ,&Asigm    ,NULL   ,&asigm    ,NULL     ,&asigmOR   ); }
gentype &OP_Agd      (gentype &a) { const static gentype fnbare("Agd(x)");       return OP_elementwiseDefaultCallA(a,fnbare,"Agd"      ,&OP_Agd      ,&Agd      ,NULL   ,&agd      ,NULL     ,&agdOR     ); }













gentype &raiseto(gentype &a, int b)
{
    if ( a.isValInteger() )
    {
        if ( b > 0 )
        {
            int i;
            int res = 1;

            for ( i = 0 ; i < b ; i++ )
            {
                res *= a.cast_int(0);
            }

            a = res;
        }

        else if ( b < 0 )
        {
            int i;
            double res = 1;

            for ( i = 0 ; i < -b ; i++ )
            {
                res /= a.cast_int(0);
            }

            a = res;
        }

        else
        {
            a = 1;
        }
    }

    else if ( a.isValReal() )
    {
        if ( b > 0 )
        {
            int i;
            double res = 1;

            for ( i = 0 ; i < b ; i++ )
            {
                res *= a.cast_double(0);
            }

            a = res;
        }

        else if ( b < 0 )
        {
            int i;
            double res = 1;

            for ( i = 0 ; i < -b ; i++ )
            {
                res /= a.cast_double(0);
            }

            a = res;
        }

        else
        {
            a = 1.0;
        }
    }

    else if ( a.isValVector() )
    {
        gentype btemp(b);

        a = epow(a,btemp);
    }

    else
    {
        gentype btemp(b);

        a = pow(a,btemp);
    }

    return a;
}








Vector<gentype> &assign(Vector<gentype> &dest, const Vector<double > &src)
{
    if ( dest.size() != src.size() )
    {
        dest.resize(src.size());
    }

    if ( src.size() )
    {
        int i;

        for ( i = 0 ; i < src.size() ; i++ )
        {
            dest("&",i) = src(i);
        }
    }

    return dest;
}

Vector<double > &assign(Vector<double > &dest, const Vector<gentype> &src)
{
    if ( dest.size() != src.size() )
    {
        dest.resize(src.size());
    }

    if ( src.size() )
    {
        int i;

        for ( i = 0 ; i < src.size() ; i++ )
        {
            dest("&",i) = src(i).cast_double(0);
        }
    }

    return dest;
}

gentype &postProInnerProd(gentype &a)
{
    if ( a.scalarfn_isscalarfn() && ( a.scalarfn_i().size() == 1 ) )
    {
        gentype res;
 
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numpts = a.scalarfn_numpts();

        int i;

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa;

        for ( i = 0 ; i < numpts ; i++ )
        {
            xa("&",a.scalarfn_i()(zeroint()))("&",a.scalarfn_j()(zeroint())) = (((double) i)+0.5)/((double) numpts);

            aa = a(xa);
            aa.finalise();
            aa /= ((double) numpts);

            res += aa;
        }

        a = res;

        a.scalarfn_setisscalarfn(0);
    }

    if ( a.scalarfn_isscalarfn() )
    {
        gentype res;
 
        NiceAssert( a.scalarfn_i().size() == a.scalarfn_j().size() );

        int numvar = a.scalarfn_i().size();
        int numpts = a.scalarfn_numpts();
        int numtot = (int) pow(numpts,numvar);

        int i,j;
        Vector<int> k(numvar);

        k = zeroint();

        res.zero();

        SparseVector<SparseVector<gentype> > xa;

        gentype aa;

        for ( i = 0 ; i < numtot ; i++ )
        {
            for ( j = 0 ; j < numvar ; j++ )
            {
                xa("&",a.scalarfn_i()(j))("&",a.scalarfn_j()(j)) = (((double) k(j))+0.5)/((double) numpts);
            }

            aa = a(xa);
            aa.finalise();
            aa /= ((double) numtot);

            res += aa;

            for ( j = 0 ; j < numvar ; j++ )
            {
                k("&",j)++;

                if ( k(j) >= numpts )
                {
                    k("&",j) = 0;
                }

                else
                {
                    break;
                }
            }
        }

        a = res;

        a.scalarfn_setisscalarfn(0);
    }

    return a;
}


void gentype::scalarfn_setisscalarfn(int nv)
{
    varid_isscalar = nv;

    //NB use of short-circuit: if !eqnargs the *eqnargs never evaluated
    if ( !nv && eqnargs && (*eqnargs).size() )
    {
	int i;

        for ( i = 0 ; i < (*eqnargs).size() ; i++ )
        {
            ((*eqnargs)("&",i)).scalarfn_setisscalarfn(nv);
	}
    }

    return;
}























// Calculator

void intercalc(std::ostream &output, std::istream &input)
{
    std::string buffer;
    int runagain = 1;

    while ( runagain )
    {
        output << "? ";
        std::getline(input,buffer);

        if ( ( buffer == "?" ) || ( buffer == "help" ) )
        {
            output << "Usage: <exp> - evaluates expression.\n";
            output << "       exit  - quit calculator.\n";
        }

        else if ( ( buffer == "quit" ) || ( buffer == "exit" ) )
        {
            runagain = 0;
        }

        else
        {
            try
            {
                gentype expeval(buffer);
                expeval.finalise();
                output << expeval << "\n";
            }

            catch ( ... )
            {
                output << "Error during evaluation.\n";
            }
        }
    }

    return;
}
























// Sparsevector specialisations

template <>
void SparseVector<gentype>::makealtcontent(void)
{
    if ( !altcontent )
    {
        int issimple = ( nearnonsparse() && isnofaroffindpresent() ) ? 1 : 0;

        if ( issimple && size() )
        {
            int i;

            for ( i = 0 ; issimple && ( i < size() ) ; i++ )
            {
                if ( !(((*this)(i)).isCastableToRealWithoutLoss()) )
                {
                    issimple = 0;
                }
            }

            if ( issimple )
            {
                MEMNEWARRAY(altcontent,double,size());

                for ( i = 0 ; issimple && ( i < size() ) ; i++ )
                {
                    altcontent[i] = (double) (*this)(i);
                }
            }
        }
    }

    return;
}

template <> gentype &oneProduct(gentype &gres, const SparseVector<gentype> &a)
{
    if ( a.altcontent )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i]);
        }
    }

    else
    {
        oneProductPrelude(gres,a);
    }

    return gres;
}

template <> gentype &oneProductScaled(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &scale)
{
    if ( a.altcontent && scale.altcontent )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( scale.size() < dim ) ? scale.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])/((scale.altcontent)[i]);
        }
    }

    else
    {
        oneProductScaledPrelude(gres,a,scale);
    }

    return gres;
}

template <> double &oneProductAssumeReal(double &res, const SparseVector<gentype> &a)
{
    if ( a.altcontent )
    {
        int i,dim = a.size();

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i]);
        }
    }

    else
    {
        oneProductAssumeRealPrelude(res,a);
    }

    return res;
}

template <> gentype &twoProduct(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int afaroff, int bfaroff)
{
    if ( a.altcontent && b.altcontent && !afaroff && !bfaroff )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i]);
        }
    }

    else
    {
        twoProductPrelude(gres,a,b,afaroff,bfaroff);
    }

    return gres;
}

template <> gentype &twoProductScaled(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &scale, int afaroff, int bfaroff)
{
    if ( a.altcontent && b.altcontent && scale.altcontent && !afaroff && !bfaroff )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( scale.size() < dim ) ? scale.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])/(((scale.altcontent)[i])*((scale.altcontent)[i]));
        }
    }

    else
    {
        twoProductScaledPrelude(gres,a,b,scale,afaroff,bfaroff);
    }

    return gres;
}

template <> gentype &twoProductNoConj(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int afaroff, int bfaroff)
{
    if ( a.altcontent && b.altcontent && !afaroff && !bfaroff )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i]);
        }
    }

    else
    {
        twoProductNoConjPrelude(gres,a,b,afaroff,bfaroff);
    }

    return gres;
}

template <> gentype &twoProductScaledNoConj(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &scale, int afaroff, int bfaroff)
{
    if ( a.altcontent && b.altcontent && scale.altcontent && !afaroff && !bfaroff )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( scale.size() < dim ) ? scale.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])/(((scale.altcontent)[i])*((scale.altcontent)[i]));
        }
    }

    else
    {
        twoProductScaledNoConjPrelude(gres,a,b,scale,afaroff,bfaroff);
    }

    return gres;
}

template <> gentype &twoProductRevConj(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, int afaroff, int bfaroff)
{
    if ( a.altcontent && b.altcontent && !afaroff && !bfaroff )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i]);
        }
    }

    else
    {
        twoProductRevConjPrelude(gres,a,b,afaroff,bfaroff);
    }

    return gres;
}

template <> gentype &twoProductScaledRevConj(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &scale, int afaroff, int bfaroff)
{
    if ( a.altcontent && b.altcontent && scale.altcontent && !afaroff && !bfaroff )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( scale.size() < dim ) ? scale.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])/(((scale.altcontent)[i])*((scale.altcontent)[i]));
        }
    }

    else
    {
        twoProductScaledRevConjPrelude(gres,a,b,scale,afaroff,bfaroff);
    }

    return gres;
}

template <> double &twoProductAssumeReal(double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b)
{
    if ( a.altcontent && b.altcontent )
    {
        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i]);
        }
    }

    else
    {
        twoProductAssumeRealPrelude(res,a,b);
    }

    return res;
}

template <> gentype &threeProduct(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c)
{
    if ( a.altcontent && b.altcontent && c.altcontent )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( c.size() < dim ) ? c.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])*((c.altcontent)[i]);
        }
    }

    else
    {
        threeProductPrelude(gres,a,b,c);
    }

    return gres;
}

template <> gentype &threeProductScaled(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &scale)
{
    if ( a.altcontent && b.altcontent && c.altcontent && scale.altcontent )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( c.size() < dim ) ? c.size() : dim;
        dim = ( scale.size() < dim ) ? scale.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])*((c.altcontent)[i])/(((scale.altcontent)[i])*((scale.altcontent)[i])*((scale.altcontent)[i]));
        }
    }

    else
    {
        threeProductScaledPrelude(gres,a,b,c,scale);
    }

    return gres;
}

template <> double &threeProductAssumeReal(double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c)
{
    if ( a.altcontent && b.altcontent && c.altcontent )
    {
        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( c.size() < dim ) ? c.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])*((c.altcontent)[i]);
        }
    }

    else
    {
        threeProductAssumeRealPrelude(res,a,b,c);
    }

    return res;
}

template <> gentype &fourProduct(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d)
{
    if ( a.altcontent && b.altcontent && c.altcontent && d.altcontent )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( c.size() < dim ) ? c.size() : dim;
        dim = ( d.size() < dim ) ? d.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])*((c.altcontent)[i])*((d.altcontent)[i]);
        }
    }

    else
    {
        fourProductPrelude(gres,a,b,c,d);
    }

    return gres;
}

template <> gentype &fourProductScaled(gentype &gres, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d, const SparseVector<gentype> &scale)
{
    if ( a.altcontent && b.altcontent && c.altcontent && d.altcontent && scale.altcontent )
    {
        double &res = gres.force_double();

        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( c.size() < dim ) ? c.size() : dim;
        dim = ( d.size() < dim ) ? d.size() : dim;
        dim = ( scale.size() < dim ) ? scale.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])*((c.altcontent)[i])*((d.altcontent)[i])/(((scale.altcontent)[i])*((scale.altcontent)[i])*((scale.altcontent)[i])*((scale.altcontent)[i]));
        }
    }

    else
    {
        fourProductScaledPrelude(gres,a,b,c,d,scale);
    }

    return gres;
}

template <> double &fourProductAssumeReal(double &res, const SparseVector<gentype> &a, const SparseVector<gentype> &b, const SparseVector<gentype> &c, const SparseVector<gentype> &d)
{
    if ( a.altcontent && b.altcontent && c.altcontent && d.altcontent )
    {
        int i,dim = a.size();

        dim = ( b.size() < dim ) ? b.size() : dim;
        dim = ( c.size() < dim ) ? c.size() : dim;
        dim = ( d.size() < dim ) ? d.size() : dim;

        res = 0.0;

        for ( i = 0 ; i < dim ; i++ )
        {
            res += ((a.altcontent)[i])*((b.altcontent)[i])*((c.altcontent)[i])*((d.altcontent)[i]);
        }
    }

    else
    {
        fourProductAssumeRealPrelude(res,a,b,c,d);
    }

    return res;
}

template <> double absinf(const SparseVector<gentype> &a)
{
    double res = 0;

    if ( a.altcontent )
    {
        int i,dim = a.size();
        double locval;

        if ( dim )
        {
            for ( i = 0 ; i < dim ; i++ )
            {
                locval = absinf((a.altcontent)[i]);

                if ( locval > res )
                {
                    res = locval;
                }
            }
        }
    }

    else
    {
        res = absinfPrelude(a);
    }

    return res;
}

template <> double abs0(const SparseVector<gentype> &a)
{
    double res = 0;

    if ( a.altcontent )
    {
        int i,dim = a.size();
        double locval;

        if ( dim )
        {
            for ( i = 0 ; i < dim ; i++ )
            {
                locval = absinf((a.altcontent)[i]);

                if ( locval < res )
                {
                    res = locval;
                }
            }
        }
    }

    else
    {
        res = abs0Prelude(a);
    }

    return res;
}

template <> double norm1(const SparseVector<gentype> &a)
{
    double res = 0;

    if ( a.altcontent )
    {
        int i,dim = a.size();

        if ( dim )
        {
            for ( i = 0 ; i < dim ; i++ )
            {
                res += norm1((a.altcontent)[i]);
            }
        }
    }

    else
    {
        res = norm1Prelude(a);
    }

    return res;
}

template <> double norm2(const SparseVector<gentype> &a)
{
    double res = 0;

    if ( a.altcontent )
    {
        int i,dim = a.size();

        if ( dim )
        {
            for ( i = 0 ; i < dim ; i++ )
            {
                res += norm2((a.altcontent)[i]);
            }
        }
    }

    else
    {
        res = norm2Prelude(a);
    }

    return res;
}

template <> double normp(const SparseVector<gentype> &a, double p)
{
    double res = 0;

    if ( a.altcontent )
    {
        int i,dim = a.size();

        if ( dim )
        {
            for ( i = 0 ; i < dim ; i++ )
            {
                res += normp((a.altcontent)[i],p);
            }
        }
    }

    else
    {
        res = normpPrelude(a,p);
    }

    return res;
}

template <> SparseVector<gentype> &operator*=(SparseVector<gentype> &left_op, const SparseVector<gentype> &right_op)
{
    if ( left_op.altcontent && right_op.altcontent && ( left_op.size() == right_op.size() ) )
    {
        int i = 0;
        int dim = left_op.size();

        if ( dim )
        {
            for ( i = 0 ; i < dim ; i++ )
            {
                (left_op.altcontent)[i] *= (right_op.altcontent)[i];
                (*(left_op.content))("&",i).force_double() = (left_op.altcontent)[i];
            }

            left_op.resetvecID();
        }
    }

    else if ( left_op.altcontent && right_op.altcontent && ( left_op.size() < right_op.size() ) )
    {
        int i = 0;
        int dim = left_op.size();

        if ( dim )
        {
            for ( i = 0 ; i < dim ; i++ )
            {
                (left_op.altcontent)[i] *= (right_op.altcontent)[i];
                (*(left_op.content))("&",i).force_double() = (left_op.altcontent)[i];
            }

            left_op.resetvecID();
        }
    }

    else if ( left_op.altcontent && right_op.altcontent && ( left_op.size() > right_op.size() ) )
    {
        int i = 0;

        if ( right_op.size() )
        {
            for ( i = 0 ; i < right_op.size() ; i++ )
            {
                (left_op.altcontent)[i] *= (right_op.altcontent)[i];
                (*(left_op.content))("&",i).force_double() = (left_op.altcontent)[i];
            }

            left_op.resetvecID();
        }

        // Design decision: rather than wasting time re-sizing, just zero-out what's left

        for ( i = right_op.size() ; i < left_op.size() ; i++ )
        {
            (left_op.altcontent)[i] *= 0.0;
            (*(left_op.content))("&",i).force_double() = (left_op.altcontent)[i];
        }
    }

    else
    {
        multass(left_op,right_op);
    }

    return left_op;
}

int testisvnan(const gentype &x)
{
    int res = 0;

         if ( x.isValDgraph()  ) { res = testisvnan(x.cast_dgraph(0)); }
    else if ( x.isValSet()     ) { res = testisvnan(x.cast_set(0));    }
    else if ( x.isValMatrix()  ) { res = testisvnan(x.cast_matrix(0)); }
    else if ( x.isValVector()  ) { res = testisvnan(x.cast_vector(0)); }
    else if ( x.isValAnion()   ) { res = testisvnan(x.cast_anion(0));  }
    else if ( x.isValReal()    ) { res = testisvnan(x.cast_double(0)); }
    else                         { res = 0;                            }

    return res;
}

int testisinf(const gentype &x)
{
    int res = 0;

         if ( x.isValDgraph()  ) { res = testisinf(x.cast_dgraph(0)); }
    else if ( x.isValSet()     ) { res = testisinf(x.cast_set(0));    }
    else if ( x.isValMatrix()  ) { res = testisinf(x.cast_matrix(0)); }
    else if ( x.isValVector()  ) { res = testisinf(x.cast_vector(0)); }
    else if ( x.isValAnion()   ) { res = testisinf(x.cast_anion(0));  }
    else if ( x.isValReal()    ) { res = testisinf(x.cast_double(0)); }
    else                         { res = 0;                           }

    return res;
}

int testispinf(const gentype &x)
{
    int res = 0;

         if ( x.isValDgraph()  ) { res = testispinf(x.cast_dgraph(0)); }
    else if ( x.isValSet()     ) { res = testispinf(x.cast_set(0));    }
    else if ( x.isValMatrix()  ) { res = testispinf(x.cast_matrix(0)); }
    else if ( x.isValVector()  ) { res = testispinf(x.cast_vector(0)); }
    else if ( x.isValAnion()   ) { res = testispinf(x.cast_anion(0));  }
    else if ( x.isValReal()    ) { res = testispinf(x.cast_double(0)); }
    else                         { res = 0;                            }

    return res;
}

int testisninf(const gentype &x)
{
    int res = 0;

         if ( x.isValDgraph()  ) { res = testisninf(x.cast_dgraph(0)); }
    else if ( x.isValSet()     ) { res = testisninf(x.cast_set(0));    }
    else if ( x.isValMatrix()  ) { res = testisninf(x.cast_matrix(0)); }
    else if ( x.isValVector()  ) { res = testisninf(x.cast_vector(0)); }
    else if ( x.isValAnion()   ) { res = testisninf(x.cast_anion(0));  }
    else if ( x.isValReal()    ) { res = testisninf(x.cast_double(0)); }
    else                         { res = 0;                            }

    return res;
}



template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<double> &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = NULL;
    }

    if ( src.imoverhere )
    {
        throw("No");
    }

    else if ( src.infsize() )
    {
        throw("No");
    }

    else
    {
        {
	    int srcsize = src.size();
	    int i;

            if ( !nbase )
	    {
	        resize(srcsize);

                if ( !(src.base()) && content && src.contentalloced() )
                {
                    if ( src.contentarray_hold() )
                    {
                        // Design decision: preallocation is duplicated

                        content->prealloc(src.contentarray_alloc());
                    }
                }
	    }

            NiceAssert( dsize == srcsize );

    	    if ( dsize )
	    {
	        for ( i = 0 ; i < dsize ; i++ )
	        {
                    (*this)("&",i) = src(i);
                }
	    }
	}
    }

    return *this;
}


template <>
template <>
Vector<double> &Vector<double>::castassign(const Vector<gentype> &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = NULL;
    }

    if ( src.imoverhere )
    {
        throw("No");
    }

    else if ( src.infsize() )
    {
        throw("No");
    }

    else
    {
        {
	    int srcsize = src.size();
	    int i;

            if ( !nbase )
	    {
	        resize(srcsize);

                if ( !(src.base()) && content && src.contentalloced() )
                {
                    if ( src.contentarray_hold() )
                    {
                        // Design decision: preallocation is duplicated

                        content->prealloc(src.contentarray_alloc());
                    }
                }
	    }

            NiceAssert( dsize == srcsize );

    	    if ( dsize )
	    {
	        for ( i = 0 ; i < dsize ; i++ )
	        {
                    (*this)("&",i) = (double) src(i);
                }
	    }
	}
    }

    return *this;
}


template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<int> &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = NULL;
    }

    if ( src.imoverhere )
    {
        throw("No");
    }

    else if ( src.infsize() )
    {
        throw("No");
    }

    else
    {
        {
	    int srcsize = src.size();
	    int i;

            if ( !nbase )
	    {
	        resize(srcsize);

                if ( !(src.base()) && content && src.contentalloced() )
                {
                    if ( src.contentarray_hold() )
                    {
                        // Design decision: preallocation is duplicated

                        content->prealloc(src.contentarray_alloc());
                    }
                }
	    }

            NiceAssert( dsize == srcsize );

    	    if ( dsize )
	    {
	        for ( i = 0 ; i < dsize ; i++ )
	        {
                    (*this)("&",i) = src(i);
                }
	    }
	}
    }

    return *this;
}



template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<gentype> &src)
{
    return assign(src);
}

template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<Vector<int> > &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = NULL;
    }

    if ( src.imoverhere )
    {
        throw("No");
    }

    else if ( src.infsize() )
    {
        throw("No");
    }

    else
    {
        {
	    int srcsize = src.size();
	    int i;

            if ( !nbase )
	    {
	        resize(srcsize);

                if ( !(src.base()) && content && src.contentalloced() )
                {
                    if ( src.contentarray_hold() )
                    {
                        // Design decision: preallocation is duplicated

                        content->prealloc(src.contentarray_alloc());
                    }
                }
	    }

            NiceAssert( dsize == srcsize );

    	    if ( dsize )
	    {
	        for ( i = 0 ; i < dsize ; i++ )
	        {
                    (*this)("&",i) = src(i);
                }
	    }
	}
    }

    return *this;
}

template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<SparseVector<gentype> > &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = NULL;
    }

    if ( src.imoverhere )
    {
        throw("No");
    }

    else if ( src.infsize() )
    {
        throw("No");
    }

    else
    {
        {
	    int srcsize = src.size();
	    int i;

            if ( !nbase )
	    {
	        resize(srcsize);

                if ( !(src.base()) && content && src.contentalloced() )
                {
                    if ( src.contentarray_hold() )
                    {
                        // Design decision: preallocation is duplicated

                        content->prealloc(src.contentarray_alloc());
                    }
                }
	    }

            NiceAssert( dsize == srcsize );

    	    if ( dsize )
	    {

                retVector<gentype> tmpa;

	        for ( i = 0 ; i < dsize ; i++ )
	        {
                    (*this)("&",i) = src(i)(tmpa);
                }
	    }
	}
    }

    return *this;
}

template <>
template <>
Vector<gentype> &Vector<gentype>::castassign(const Vector<Vector<gentype> > &src)
{
    NiceAssert( !infsize() );

    if ( imoverhere )
    {
        MEMDEL(imoverhere);
        imoverhere = NULL;
    }

    if ( src.imoverhere )
    {
        throw("No");
    }

    else if ( src.infsize() )
    {
        throw("No");
    }

    else
    {
        {
	    int srcsize = src.size();
	    int i;

            if ( !nbase )
	    {
	        resize(srcsize);

                if ( !(src.base()) && content && src.contentalloced() )
                {
                    if ( src.contentarray_hold() )
                    {
                        // Design decision: preallocation is duplicated

                        content->prealloc(src.contentarray_alloc());
                    }
                }
	    }

            NiceAssert( dsize == srcsize );

    	    if ( dsize )
	    {
	        for ( i = 0 ; i < dsize ; i++ )
	        {
                    (*this)("&",i) = src(i);
                }
	    }
	}
    }

    return *this;
}


template <>
template <>
Vector<SparseVector<gentype> >& Vector<SparseVector<gentype> >::castassign(const Vector<SparseVector<gentype> > &src)
{
    return assign(src);
}

