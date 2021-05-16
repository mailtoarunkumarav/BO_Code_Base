
//
// Stack class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// What is it: a vector with push and pop operations added on top

#ifndef _vecstack_h
#define _vecstack_h

#include "vector.h"

template <class T>
class Stack;

// Stream operators

template <class T> std::ostream &operator<<(std::ostream &output, const Stack<T> &src );
template <class T> std::istream &operator>>(std::istream &input,        Stack<T> &dest);

// Swap function

template <class T> void qswap(Stack<T> &a, Stack<T> &b);

// The class itself

template <class T>
class Stack : public Vector<T>
{
public:

    // Constructors and Destructors

    svm_explicit Stack() : Vector<T>()
    {
        return;
    }

    Stack(const Stack<T> &src) : Vector<T>(static_cast<Vector<T> &>(src))
    {
        return;
    }

    svm_explicit Stack(const Vector<T> &src) : Vector<T>(src)
    {
        return;
    }

    // Assignment

    Stack<T> &operator=(const Stack<T> &src)
    {
        static_cast<Vector<T> &>(*this) = static_cast<const Vector<T> &>(src);

        return *this;
    }

    Stack<T> &operator=(const Vector<T> &src)
    {
        static_cast<Vector<T> &>(*this) = src;

        return *this;
    }

    // Stack operations
    //
    // accessTop: returns reference to element on top of stack
    // push: push element onto stack
    // pop: pop element from stack
    // safepop: like pop, but returns 0 if pop successful, 1 if stack empty
    // isempty: return 1 if stack empty, 0 otherwise

    void push(const T &src)
    {
        (*this).add((*this).size());
	accessTop() = src;

        return;
    }

    int pop(T &dest)
    {
	if ( !isempty() )
	{
	   dest = accessTop();
           (*this).remove((*this).size()-1);

	   return 0;
	}

        return 1;
    }

    int pop(void)
    {
	if ( !isempty() )
	{
           (*this).remove((*this).size()-1);

	   return 0;
	}

        return 1;
    }

    T &accessTop(void)
    {
        return (*this)("&",(*this).size()-1);
    }

    int isempty(void)
    {
        return ( (*this).size() == 0 );
    }
};

template <class T>
void qswap(Stack<T> &a, Stack<T> &b)
{
    qswap(static_cast<Vector<T> &>(a),static_cast<Vector<T> &>(b));

    return;
}

template <class T>
std::ostream &operator<<(std::ostream &output, const Stack<T> &src)
{
    output << static_cast<const Vector<T> &>(src);

    return output;
}

template <class T>
std::istream &operator>>(std::istream &input, Stack<T> &dest)
{
    input >> static_cast<Vector<T> &>(dest);

    return input;
}

#endif
