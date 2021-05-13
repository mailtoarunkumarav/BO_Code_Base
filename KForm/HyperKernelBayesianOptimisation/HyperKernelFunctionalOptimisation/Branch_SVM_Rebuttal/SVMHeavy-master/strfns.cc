
//
// Parenthesis-aware streaming function
//
// Version: 6
// Date: 05/12/2014
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctype.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <string>
#include "strfns.h"
#include "numbase.h"
#include "vecstack.h"


int readParenString(std::istream &input, std::string &dest)
{
    Stack<char> parenStack;
    int isQuote = 0;
    char tt = 'x';

    dest = "";

    while ( !(input.eof()) && !( !(parenStack.size()) && ( isspace(input.peek()) || ( input.peek() == ';' ) || ( input.peek() == ',' ) || ( input.peek() == ']' ) ) ) )
    {
        input.get(tt);

        if ( !isQuote )
        {
            if ( tt == '\"' )
            {
                isQuote = 1;
                parenStack.push(tt);
            }

            else if ( ( tt == '(' ) || ( tt == '[' ) || ( tt == '{' ) )
            {
                parenStack.push(tt);
            }

            else if ( ( tt == ')' ) || ( tt == ']' ) || ( tt == '}' ) )
            {
                if ( parenStack.size() )
                {
                    if ( ( parenStack.accessTop() == '(' ) && ( tt == ')' ) )
                    {
                        parenStack.pop();
                    }

                    else if ( ( parenStack.accessTop() == '[' ) && ( tt == ']' ) )
                    {
                        parenStack.pop();
                    }

                    else if ( ( parenStack.accessTop() == '{' ) && ( tt == '}' ) )
                    {
                        parenStack.pop();
                    }

                    else
                    {
                        dest += tt;

                        return 1;
                    }
                }

                else
                {
                    return 0;
                }
            }
        }

        else if ( tt == '\"' )
        {
            NiceAssert( parenStack.size() );
            isQuote = 0;
            parenStack.pop();
        }

        dest += tt;

        input.peek();
    }

    if ( parenStack.size() )
    {
        return 1;
    }

    return 0;
}

