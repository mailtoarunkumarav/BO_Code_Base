#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "gentype.h"
#include "vector.h"
#include "sparsevector.h"




int main()
{
    gentype f;
    double xmin;
    double xmax;
    int xits;
    Vector<double> yvals;

    std::cerr << "f: "; std::cin >> f;
    std::cerr << "x min: "; std::cin >> xmin;
    std::cerr << "x max: "; std::cin >> xmax;
    std::cerr << "x iterations: "; std::cin >> xits;
    std::cerr << "y vector: "; std::cin >> yvals;

    int i,j,k;
    gentype x;
    gentype y;
    Vector<gentype> fres(yvals.size());

    for ( i = 0 ; i <= xits ; i++ )
    {
        x =  xmin;
        x += i*((xmax-xmin)/xits);

        for ( j = 0 ; j < yvals.size() ; j++ )
        {
            y = yvals(j);

            fres("&",j) = f(x,y);
        }

        max(fres,k);

        for ( j = 0 ; j < yvals.size() ; j++ )
        {
            if ( j == k )
            {
                std::cout << "*";
            }

            std::cout << fres(j) << "\t";
        }

        std::cout << "\n";
    }

    return 0;
}
