
//
// Parenthesis-aware streaming function
//
// Version: 6
// Date: 23/06/2015
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

#ifndef _strfns_h
#define _strfns_h

#include <iostream>
#include <string>

// Read parenthesised string
//
// That is, read string until white-space not included in parenthesis is
// encountered.  Parenthesis are [], (), {} and "".  The first three of these
// are nested in the normal way, while the third differs insofar as
// parenthesise inside it are ignorned.  Good examples:
//
// 12
// [ 1 2 { 3 } "hello world {]]]" ]
// [ { 1 2 } 3 [ 4 5 ] ]
//
// Bad examples:
//
// [ 1 2 }
// "Hello world
//
// Strings end at either whitespace not contained in parenthesis, eof() or
// at close of parenthesis ])} not paired with initial parenthesis.  For
// example:
//
//     stream    | output string
// --------------+---------------
//  "[ 1 2 3 ]"  | "[ 1 2 3 ]"
//  "[ 1 2 3 ] " | "[ 1 2 3 ]"
//  "[ 1 2 3 ])" | "[ 1 2 3 ]"
//
// return 0 on success
// return 1 for unpaired brackets
// return 2 for other error (currently none)

int readParenString(std::istream &input, std::string &dest);

#endif
