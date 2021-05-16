lsv:

    virtual const Vector<gentype> &gamma(void) const { return dalpha; }
    virtual const gentype         &delta(void) const { return dbias;  }

ssv:

    virtual const Vector<gentype>                &beta  (void) const { return zbeta;                                }
    virtual const gentype                        &b     (void) const { return zb;                                   }

svm:

    virtual const gentype                 &bias       (void)        const {                                                                                                                         return dbias;       }
    virtual const Vector<gentype>         &alpha      (void)        const {                                                                                                                         return dalpha;      }

gpr:

    virtual const Vector<gentype> &muWeight(void) const { return getQconst().gamma(); }
    virtual const gentype         &muBias  (void) const { return getQconst().delta(); }



/*

Variance in model
=================

g(x) = sum_i alpha_i K(x,x_i) + b

E g(x) = sum_i E( alpha_i K(x,x_i) ) + E(b)
       = sum_i E(alpha_i) E(K(x,x_i)) + E(b)

var g(x) = E ( ( g(x) - E(g(x)) )^2 )
         = E ( (   sum_i alpha_i K(x,x_i)
                 + b
                 - sum_i E(alpha_i) E(K(x,x_i))
                 - E(b) )^2 )
         = E ( (   sum_i alpha_i K(x,x_i)
                 + (b-E(b))
                 - sum_i E(alpha_i) E(K(x,x_i)) )^2 )
         = E ( (     sum_ij alpha_i.alpha_j.K(x,x_i).K(x,x_j)
                 +   (b-E(b))^2
                 +   sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
                 + 2.sum_i alpha_i.K(x,x_i).(b-E(b))
                 - 2.sum_ij alpha_i.E(alpha_j).K(x,x_i).E(K(x,x_j)) 
                 - 2.sum_j (b-E(b)).E(alpha_j).E(K(x,x_j)) )
         =     sum_ij E(alpha_i.alpha_j.K(x,x_i).K(x,x_j))
           +   E((b-E(b))^2)
           +   sum_ij E(E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j)))
           + 2.sum_i E(alpha_i.K(x,x_i).(b-E(b)))
           - 2.sum_ij E(alpha_i.E(alpha_j).K(x,x_i).E(K(x,x_j)))
           - 2.sum_j E((b-E(b)).E(alpha_j).E(K(x,x_j)))
         =     sum_ij E(alpha_i.alpha_j).E(K(x,x_i).K(x,x_j))
           +   var(b)
           +   sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           + 2.sum_i E(K(x,x_i)).E(alpha_i.(b-E(b)))
           - 2.sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           - 2.sum_j E(b-E(b)).E(alpha_j).E(K(x,x_j))
         =     sum_ij E(  alpha_i).E(alpha_j).E(  K(x,x_i)).E(K(x,x_j))
           +   sum_ij cov(alpha_i,   alpha_j).E(  K(x,x_i)).E(K(x,x_j))
           +   sum_ij E(  alpha_i).E(alpha_j).cov(K(x,x_i),   K(x,x_j))
           +   sum_ij cov(alpha_i,   alpha_j).cov(K(x,x_i),   K(x,x_j))
           +   var(b)
           +   sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           + 2.sum_i E(K(x,x_i)).E(  alpha_i).E(b-E(b))
           + 2.sum_i E(K(x,x_i)).cov(alpha_i,   b-E(b))
           - 2.sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           - 2.sum_j E(b-E(b)).E(alpha_j).E(K(x,x_j))
         =     sum_ij E(  alpha_i).E(alpha_j).E(  K(x,x_i)).E(K(x,x_j))
           +   sum_ij E(  alpha_i).E(alpha_j).E(  K(x,x_i)).E(K(x,x_j))
           - 2.sum_ij E(  alpha_i).E(alpha_j).E(  K(x,x_i)).E(K(x,x_j))
           +   sum_ij cov(alpha_i,   alpha_j).E(  K(x,x_i)).E(K(x,x_j))
           +   sum_ij E(  alpha_i).E(alpha_j).cov(K(x,x_i),   K(x,x_j))
           +   sum_ij cov(alpha_i,   alpha_j).cov(K(x,x_i),   K(x,x_j))
           +   var(b)
           + 2.sum_i E(K(x,x_i)).E(alpha_i).E(b-E(b))
           - 2.sum_i E(K(x,x_i)).E(alpha_i).E(b-E(b))
           + 2.sum_i E(K(x,x_i)).cov(alpha_i,b-E(b))
         =     sum_ij cov(alpha_i,   alpha_j).E(  K(x,x_i)).E(K(x,x_j))
           +   sum_ij E(  alpha_i).E(alpha_j).cov(K(x,x_i),   K(x,x_j))
           +   sum_ij cov(alpha_i,   alpha_j).cov(K(x,x_i),   K(x,x_j))
           +   var(b)
           + 2.sum_i E(K(x,x_i)).cov(alpha_i,b-E(b))

So:

g(x)       := g(x)
sigma^2(x) := sigma^2(x) + e^2(x)

where, on rhs:

g(x)       inheritted from model
sigma^2(x) inheritted from model

e^2(x) =   sum_ij cov(alpha_i,   alpha_j).E(  K(x,x_i)).E(K(x,x_j))
       +   sum_ij E(  alpha_i).E(alpha_j).cov(K(x,x_i),   K(x,x_j))
       +   sum_ij cov(alpha_i,   alpha_j).cov(K(x,x_i),   K(x,x_j))
       +   var(b)
       + 2.sum_i E(K(x,x_i)).cov(alpha_i,b)

calculate:

- E(alpha_i), cov(alpha_i,alpha_j), cov(alpha_i,b), var(b) using bootstrap.

inherit:

- cov(K(x,x_i),K(x,x_j)) from previous model if 800-series kernel used

Assumption: K(x,x_i),K(x,x_j) independent (should be true: it's an inner
            project with one side = function of x_i or x_j.

so:

e^2(x) =   sum_ij cov(alpha_i,alpha_j).E(K(x,x_i)).E(K(x,x_j))
       +   sum_i E(alpha_i)^2.var(K(x,x_i))
       +   sum_i var(alpha_i).var(K(x,x_i))
       +   var(b)
       + 2.sum_i E(K(x,x_i)).cov(alpha_i,b)

---------------------------------------------------------------------------

Variance in inheritted kernel (801)
===================================

K(x,y) = sum_ij alpha_i.alpha_j.K(x_i,x_j,x,y)

E(K(x,y)) =   sum_ij E(alpha_i).E(alpha_j).E(K(x_i,x_j,x,y))
            + sum_ij cov(alpha_i, alpha_j).E(K(x_i,x_j,x,y))

var(K(x,y)) = E( ( K(x,y) - E(K(x,y)) )^2 )
            = E( ( sum_ij   alpha_i.   alpha_j.   K(x_i,x_j,x,y)
                 - sum_ij E(  alpha_i).E(alpha_j).E(K(x_i,x_j,x,y))
                 - sum_ij cov(alpha_i,   alpha_j).E(K(x_i,x_j,x,y)) )^2 )
            = E(     sum_ijkl     alpha_i.   alpha_j.     alpha_k.   alpha_l.   K(x_i,x_j,x,y).   K(x_k,x_l,x,y)
                 +   sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl     alpha_i.   alpha_j. E(  alpha_k).E(alpha_l).  K(x_i,x_j,x,y). E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl     alpha_i.   alpha_j. cov(alpha_k,   alpha_l).  K(x_i,x_j,x,y). E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )
            =        sum_ijkl E(  alpha_i.   alpha_j.     alpha_k.   alpha_l).E(K(x_i,x_j,x,y).   K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i.   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i.   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )
(assuming no kernel covariance)
            =        sum_ijkl E(  alpha_i.   alpha_j.     alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i.   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i.   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )
            =        sum_ijkl E(  alpha_i.   alpha_j).E(  alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i.   alpha_j ,    alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl cov(alpha_i,   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )
            =        sum_ijkl cov(alpha_i.   alpha_j ,    alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl cov(alpha_i,   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )
            =        sum_ijkl cov(alpha_i.   alpha_j ,    alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl cov(alpha_i,   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )
                 +   sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl cov(alpha_i,   alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
            =        sum_ijkl cov(alpha_i.   alpha_j ,    alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 -   sum_ijkl cov(alpha_i,   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 3.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )

            =        sum_ijkl cov(alpha_i.   alpha_k).cov(alpha_j.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i.   alpha_k).E(  alpha_j).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl E(  alpha_i).E(alpha_k).cov(alpha_j.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 -   sum_ijkl cov(alpha_i,   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 3.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )
then errors
            =        sum_ijkl cov(alpha_i.   alpha_j).cov(alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 -   sum_ijkl cov(alpha_i,   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 +   sum_ijkl cov(alpha_i.   alpha_j).E(  alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 3.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )
                 +   sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
            =        sum_ijkl cov(alpha_i.   alpha_j).cov(alpha_k.   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                 - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,   alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )


so:

var(K(x,y)) =   sum_ijkl cov(alpha_i.   alpha_j).cov(alpha_k,alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
            - 2.sum_ijkl E(  alpha_i).E(alpha_j).cov(alpha_k,alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y)) )


Summary:

E(K(x,y))   = sum_ij ( E(alpha_i).E(alpha_j) + cov(alpha_i,alpha_j) ).E(K(x_i,x_j,x,y))

var(K(x,y)) = sum_ijkl ( cov(alpha_i.alpha_j) - 2.E(alpha_i).E(alpha_j) ).cov(alpha_k,alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Simplify: cov(alpha_i,alpha_j) = delta_ij var(alpha_i)

Variance in model
=================

g(x) = sum_i alpha_i K(x,x_i) + b

E g(x) = sum_i E( alpha_i K(x,x_i) ) + E(b)
       = sum_i E(alpha_i) E(K(x,x_i)) + E(b)

var g(x) = E ( ( g(x) - E(g(x)) )^2 )
         = E ( (   sum_i alpha_i K(x,x_i)
                 + b
                 - sum_i E(alpha_i) E(K(x,x_i))
                 - E(b) )^2 )
         = E ( (   sum_i alpha_i K(x,x_i)
                 + (b-E(b))
                 - sum_i E(alpha_i) E(K(x,x_i)) )^2 )
         = E ( (     sum_ij alpha_i.alpha_j.K(x,x_i).K(x,x_j)
                 +   (b-E(b))^2
                 +   sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
                 + 2.sum_i alpha_i.K(x,x_i).(b-E(b))
                 - 2.sum_ij alpha_i.E(alpha_j).K(x,x_i).E(K(x,x_j)) 
                 - 2.sum_j (b-E(b)).E(alpha_j).E(K(x,x_j)) )
         =     sum_ij E(alpha_i.alpha_j.K(x,x_i).K(x,x_j))
           +   E((b-E(b))^2)
           +   sum_ij E(E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j)))
           + 2.sum_i E(alpha_i.K(x,x_i).(b-E(b)))
           - 2.sum_ij E(alpha_i.E(alpha_j).K(x,x_i).E(K(x,x_j)))
           - 2.sum_j E((b-E(b)).E(alpha_j).E(K(x,x_j)))
         =     sum_ij E(alpha_i.alpha_j).E(K(x,x_i).K(x,x_j))
           +   var(b)
           +   sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           + 2.sum_i E(alpha_i.(b-E(b))).E(K(x,x_i))
           - 2.sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           - 2.sum_j E((b-E(b)).E(alpha_j)).E(K(x,x_j))
         =     sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           +   sum_i E(alpha_i)^2.var(K(x,x_i))
           +   sum_i var(alpha_i).E(K(x,x_i))^2
           +   sum_i var(alpha_i).var(K(x,x_i))
           +   var(b)
           +   sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           + 2.sum_i E(alpha_i).E(b-E(b)).E(K(x,x_i))
           - 2.sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           - 2.sum_j E(b-E(b)).E(alpha_j)).E(K(x,x_j))
         =     sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           +   sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           - 2.sum_ij E(alpha_i).E(alpha_j).E(K(x,x_i)).E(K(x,x_j))
           +   sum_i E(alpha_i)^2.var(K(x,x_i))
           +   sum_i var(alpha_i).E(K(x,x_i))^2
           +   sum_i var(alpha_i).var(K(x,x_i))
           +   var(b)
           + 2.sum_i E(alpha_i).E(b-E(b)).E(K(x,x_i))
           - 2.sum_i E(alpha_i).E(b-E(b)).E(K(x,x_i))
         =     sum_i E(alpha_i)^2.var(K(x,x_i))
           +   sum_i var(alpha_i).E(K(x,x_i))^2
           +   sum_i var(alpha_i).var(K(x,x_i))
           +   var(b)

So:

g(x)       := g(x)
sigma^2(x) := sigma^2(x) + e^2(x)

where, on rhs:

g(x)       inheritted from model
sigma^2(x) inheritted from model

e^2(x) = sum_i var(alpha_i).E(K(x,x_i))^2
       + sum_i E(alpha_i)^2.var(K(x,x_i))
       + sum_i var(alpha_i).var(K(x,x_i))
       + var(b)

calculate:

- E(alpha_i), var(alpha_i), var(b) using bootstrap.

inherit:

- var(K(x,x_i)) from previous model if 800-series kernel used




---------------------------------------------------------------------------

Variance in inheritted kernel (801)
===================================

K(x,y) = sum_ij alpha_i.alpha_j.K(x_i,x_j,x,y)

E(K(x,y)) =   sum_ij E(alpha_i).E(alpha_j).E(K(x_i,x_j,x,y))
            + sum_i var(alpha_i).E(K(x_i,x_i,x,y))

var(K(x,y)) = E( ( K(x,y) - E(K(x,y)) )^2 )
            = E( (   sum_ij   alpha_i .  alpha_j .  K(x_i,x_j,x,y)
                   - sum_ij E(alpha_i).E(alpha_j).E(K(x_i,x_j,x,y))
                   - sum_i var(alpha_i).E(K(x_i,x_i,x,y))           )^2 )

            = E( (     sum_ijkl   alpha_i .  alpha_j .  alpha_k .  alpha_l .  K(x_i,x_j,x,y) .  K(x_k,x_l,x,y)
                   +   sum_ijkl E(alpha_i).E(alpha_j).E(alpha_k).E(alpha_l).E(K(x_i,x_j,x,y)).E(K(x_k,x_l,x,y))
                   +   sum_ij var(alpha_i).var(alpha_j).E(K(x_i,x_i,x,y)).E(K(x_j,x_j,x,y))
                   - 2.sum_ijkl   alpha_i .  alpha_j .E(alpha_k).E(alpha_l).  K(x_i,x_j,x,y) .E(K(x_k,x_l,x,y))
                   - 2.sum_ijk   alpha_i .  alpha_j .var(alpha_k).  K(x_i,x_j,x,y) .E(K(x_k,x_k,x,y))
                   + 2.sum_ijk E(alpha_i).E(alpha_j).var(alpha_k).E(K(x_i,x_j,x,y)).E(K(x_k,x_k,x,y))






            = E( (     sum_ijkl   a_i .  a_j .  a_k .  a_l .  Kij .  Kkl
                   +   sum_ijkl E(a_i).E(a_j).E(a_k).E(a_l).E(Kij).E(Kkl)
                   +   sum_ij   v(a_i).v(a_j).E(Kii).E(Kjj)
                   - 2.sum_ijkl   a_i .  a_j .E(a_k).E(a_l).  Kij .E(Kkl)
                   - 2.sum_ijk    a_i .  a_j .v(a_k).  Kij .E(Kkk)
                   + 2.sum_ijk  E(a_i).E(a_j).v(a_k).E(Kij).E(Kkk)

            =     sum_ijkl E(a_i .  a_j .  a_k .  a_l).E(Kij .  Kkl)
              +   sum_ijkl E(a_i).E(a_j).E(a_k).E(a_l).E(Kij).E(Kkl)
              +   sum_ij   v(a_i).v(a_j).E(Kii).E(Kjj)
              - 2.sum_ijkl E(a_i .  a_j).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijk  E(a_i .  a_j).v(a_k).E(Kij).E(Kkk)
              + 2.sum_ijk  E(a_i).E(a_j).v(a_k).E(Kij).E(Kkk)

            =     sum_ijkl E(a_i   .  a_j   .  a_k .  a_l).E(Kij).E(Kkl)
              +   sum_ij   E(a_i^2 .  a_j^2).              v(Kij)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              +   sum_ij   v(a_i  ).v(a_j  ).              E(Kii).E(Kjj)
              - 2.sum_ijkl E(a_i   .  a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijk  E(a_i   .  a_j  ).v(a_k).       E(Kij).E(Kkk)
              + 2.sum_ijk  E(a_i  ).E(a_j  ).v(a_k).       E(Kij).E(Kkk)

            =     sum_ijkl E(a_i   .  a_j   .  a_k .  a_l).E(Kij).E(Kkl)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijkl E(a_i   .  a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijk  E(a_i   .  a_j  ).v(a_k).       E(Kij).E(Kkk)
              + 2.sum_ijk  E(a_i  ).E(a_j  ).v(a_k).       E(Kij).E(Kkk)
              +   sum_ij   E(a_i^2 .  a_j^2).              v(Kij)
              +   sum_ij   v(a_i  ).v(a_j  ).              E(Kii).E(Kjj)

            =     sum_ijkl E(a_i   .  a_j   .  a_k .  a_l).E(Kij).E(Kkl)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijkl E(a_i   .  a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijk  E(a_i   .  a_j  ).v(a_k).       E(Kij).E(Kkk)
              + 2.sum_ijk  E(a_i  ).E(a_j  ).v(a_k).       E(Kij).E(Kkk)
              +   sum_ij   E(a_i^2).E(a_j^2).              v(Kij)
              +   sum_ij   v(a_i  ).v(a_j  ).              E(Kii).E(Kjj)
              +   sum_i    v(a_i^2).                       v(Kii)

            =     sum_ijkl E(a_i   .  a_j   .  a_k .  a_l).E(Kij).E(Kkl)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ikl  v(a_i  )         .E(a_k).E(a_l).E(Kii).E(Kkl)
              - 2.sum_ijk  E(a_i   .  a_j  ).v(a_k).       E(Kij).E(Kkk)
              + 2.sum_ijk  E(a_i  ).E(a_j  ).v(a_k).       E(Kij).E(Kkk)
              +   sum_ij   E(a_i^2).E(a_j^2).              v(Kij)
              +   sum_ij   v(a_i  ).v(a_j  ).              E(Kii).E(Kjj)
              +   sum_i    v(a_i^2).                       v(Kii)

            =     sum_ijkl E(a_i   .  a_j   .  a_k .  a_l).E(Kij).E(Kkl)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ikl  v(a_i  )         .E(a_k).E(a_l).E(Kii).E(Kkl)
              - 2.sum_ijk  E(a_i  ).E(a_j  ).v(a_k).       E(Kij).E(Kkk)
              - 2.sum_ik   v(a_i  )         .v(a_k).       E(Kii).E(Kkk)
              + 2.sum_ijk  E(a_i  ).E(a_j  ).v(a_k).       E(Kij).E(Kkk)
              +   sum_ij   E(a_i^2).E(a_j^2).              v(Kij)
              +   sum_ij   v(a_i  ).v(a_j  ).              E(Kii).E(Kjj)
              +   sum_i    v(a_i^2).                       v(Kii)

            =     sum_ijkl E(a_i   .  a_j   .  a_k .  a_l).E(Kij).E(Kkl)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)

              - 2.sum_ijk  v(a_i  )         .E(a_j).E(a_k).E(Kii).E(Kjk)
              - 2.sum_ijk  E(a_j  ).E(a_k  ).v(a_i).       E(Kjk).E(Kii)
              + 2.sum_ijk  E(a_j  ).E(a_k  ).v(a_i).       E(Kjk).E(Kii)

              - 2.sum_ij   v(a_i  )         .v(a_j).       E(Kii).E(Kjj)
              +   sum_ij   E(a_i^2).E(a_j^2).              v(Kij)
              +   sum_ij   v(a_i  ).v(a_j  ).              E(Kii).E(Kjj)
              +   sum_i    v(a_i^2).                       v(Kii)

            =     sum_ijkl E(a_i   .  a_j   .  a_k .  a_l).E(Kij).E(Kkl)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)

              - 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              - 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              + 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)

              - 2.sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij E(a_i^2).E(a_j^2).v(Kij)
              +   sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)

              +   sum_i v(a_i^2).v(Kii)

            =     sum_ijkl E(a_i   .  a_j  ).E(a_k .  a_l).E(Kij).E(Kkl)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)

              - 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              - 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              + 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)

              - 2.sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij E(a_i^2).E(a_j^2).v(Kij)
              +   sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij v(a_i .  a_j  ).E(Kij).E(Kij)

              +   sum_i v(a_i^2).v(Kii)

            =     sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              +   sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)
              - 2.sum_ijkl E(a_i  ).E(a_j  ).E(a_k).E(a_l).E(Kij).E(Kkl)

              - 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              - 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              + 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              +   sum_ijk E(a_i).E(a_j).v(a_k).E(Kij).E(Kkk)
              +   sum_ikl v(a_i).E(a_k).E(a_l).E(Kii).E(Kkl)

              - 2.sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij E(a_i^2).E(a_j^2).v(Kij)
              +   sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij v(a_i .  a_j  ).E(Kij).E(Kij)
              +   sum_ik v(a_i).v(a_k).E(Kii).E(Kkk)

              +   sum_i v(a_i^2).v(Kii)

            = - 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              - 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              + 2.sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              +   sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)
              +   sum_ijk v(a_i).E(a_j).E(a_k).E(Kii).E(Kjk)

              - 2.sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij E(a_i^2).E(a_j^2).v(Kij)
              +   sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij v(a_i .  a_j  ).E(Kij).E(Kij)
              +   sum_ik v(a_i).v(a_k).E(Kii).E(Kkk)

              +   sum_i v(a_i^2).v(Kii)

            = - 2.sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij v(a_i  ).v(a_j  ).E(Kii).E(Kjj)
              +   sum_ij E(a_i^2).E(a_j^2).v(Kij)
              +   sum_ij v(a_i .  a_j  ).E(Kij).E(Kij)

              +   sum_i v(a_i^2).v(Kii)

            =    sum_ij E(a_i^2).E(a_j^2).v(Kij)
              +  sum_ij v(a_i.a_j).E(Kij).E(Kij)
              +  sum_i v(a_i^2).v(Kii)

so:

E(K(x,y)) =   sum_ij E(alpha_i).E(alpha_j).E(K(x_i,x_j,x,y))
            + sum_i var(alpha_i).E(K(x_i,x_i,x,y))

var(K(x,y)) =    sum_ij E(alpha_i^2).E(alpha_j^2).var(Kij)
              +  sum_ij var(alpha_i.alpha_j).E(Kij).E(Kij)
              +  sum_i var(alpha_i^2).var(Kii)


*/
