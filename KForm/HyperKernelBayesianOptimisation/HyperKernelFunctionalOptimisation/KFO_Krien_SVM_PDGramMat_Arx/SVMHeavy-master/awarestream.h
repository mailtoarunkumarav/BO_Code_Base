
//
// Very slightly smarter stream class
//
// Version: 7
// Date: 08/04/2016
// Written by: Alistair Shilton (AlShilton@gmail.com)
// Copyright: all rights reserved
//

// What is it: a vector with push and pop operations added on top
// and some other stuff.  It also works as istream/ostream.

#ifndef _awarestream_h
#define _awarestream_h

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <string>
#include <math.h>
#include <streambuf>
#include "basefn.h"




// Rudimentary string fifo buffer

class stringfifo;
class stringfifo
{
public:
    svm_explicit stringfifo(void)
    {
        next     = NULL;
        nodeval  = NULL;
        killflag = 0;

        return;
    }

    ~stringfifo(void)
    {
        if ( nodeval )
        {
            MEMDELARRAY(nodeval);
            nodeval = NULL;
        }

        if ( next )
        {
            MEMDEL(next);
            next = NULL;
        }

        return;
    }

    // Push string onto fifo.  Note that this ignores the kill flag
    //
    // Success return: killflag (0 or 1)
    // Fail return: -killflag-2
    //              (-2 means empty fifo, so can just wait for it to fill)
    //
    // The return value is used, so don't redefine behaviour

    int push(const std::string &src) svmvolatile
    {
        int ires = killflag;

        std::stringstream ss(src);
        std::string partsrc;

        while ( ss.good() && !ires )
        {
            ss >> partsrc;

            if ( partsrc.length() )
            {
                ires |= push(partsrc.c_str(),(unsigned int) partsrc.length());
            }
        }

        return ires;
    }

    // Pop string off fifo.  Note that this ignores the kill flag
    //
    // Success return: killflag (0 or 1)
    // Fail return: -killflag-2
    //
    // The return value is used, so don't redefine behaviour

    int pop(std::string &res) svmvolatile
    {
        int ires = killflag;

        if ( !next )
        {
            ires = -ires-2;
        }

        else if ( !(next->next) )
        {
            NiceAssert(nodeval);

            // nodeval is typically volatile char * volatile
            // this makes it tricky to cast away volatility for assignment.
            // note however that this code is only accessed by one thread at
            // a time, so the following hack should work fine as nodeval
            // can't change midway through.
            res = (char *) ((void *) nodeval);

            MEMDELARRAY(nodeval);
            nodeval = NULL;
            MEMDEL(next);
            next = NULL;
        }

        else
        {
            ires = next->pop(res);
        }

        return ires;
    }

    // Returns: -1: kill flag is set
    //           0: fifo is empty
    //          >1: number of strings on fifo

    int size(void) svmvolatile
    {
        int res = 0;

        if ( killflag )
        {
            return -1;
        }

        else if ( next )
        {
            res = (next->size())+1;
        }

        return res;
    }

    // Set/unset "kill flag"

    void setkillflag(void) svmvolatile
    {
        killflag = 1;

        if ( next )
        {
            next->setkillflag();
        }

        return;
    }

    void unsetkillflag(void) svmvolatile
    {
        killflag = 0;

        if ( next )
        {
            next->unsetkillflag();
        }

        return;
    }

private:
    svmvolatile char *nodeval; // assumed empty if next = NULL;
    svmvolatile stringfifo *next;
    svmvolatile int killflag;

    svm_explicit stringfifo(int _killflag)
    {
        next     = NULL;
        nodeval  = NULL;
        killflag = _killflag;

        return;
    }

    int push(const char *src, unsigned int len) svmvolatile
    {
        int ires = killflag;

        stringfifo *newnext; MEMNEW(newnext,stringfifo(killflag)); // need to pass killflag

        newnext->next    = next;
        newnext->nodeval = nodeval;

        next    = newnext;
        MEMNEWARRAY(nodeval,char,len+1);

        if ( !nodeval )
        {
            ires = -ires-2;
        }

        else
        {
            if ( len )
            {
                unsigned int i;

                for ( i = 0 ; i < len ; i++ )
                {
                    nodeval[i] = src[i];
                }
            }

            nodeval[len] = '\0';
        }

        return ires;
    }
};

// Rudimentary sparse vector of string fifos

class fifolist;
class fifolist
{
public:
    svm_explicit fifolist(void)
    {
        fifonum = 0;
        next = NULL;

        return;
    }

    ~fifolist(void)
    {
        if ( next )
        {
            MEMDEL(next);
            next = NULL;
        }

        return;
    }

    // push string onto given fifo

    int push(unsigned int num, const std::string &src) svmvolatile
    {
        return getnum(num).push(src);
    }

    // pop string off given fifo

    int pop(unsigned int num, std::string &res) svmvolatile
    {
        return getnum(num).pop(res);
    }

    // return size of given fifo

    int size(unsigned int num) svmvolatile
    {
        return getnum(num).size();
    }

    // set kill flag for given fifo

    void setkillflag(unsigned int num) svmvolatile
    {
        getnum(num).setkillflag();

        return;
    }

    // set kill flag for all

    void setkillflag(void) svmvolatile
    {
        thisfifo.setkillflag();

        if ( next )
        {
            next->setkillflag();
        }

        return;
    }

    // return number of fifos in sparse vector

    int indsize(void) svmvolatile
    {
        int ires = 0;

        if ( next )
        {
            ires += next->indsize();
        }

        return ires;
    }

private:
    svmvolatile unsigned int fifonum;
    svmvolatile stringfifo thisfifo;
    svmvolatile fifolist *next;

    // This constructor is only used internally

    svm_explicit fifolist(unsigned int num)
    {
        fifonum = num;
        next = NULL;

        return;
    }

    // Retrieve element in fifo, assigning first if need be

    svmvolatile stringfifo &getnum(unsigned int num) svmvolatile
    {
        NiceAssert( num >= fifonum );

        svmvolatile stringfifo *res = NULL;

        if ( fifonum == num )
        {
            res = &thisfifo;
        }

        else
        {
            if ( !next )
            {
                MEMNEW(next,fifolist(num));
            }

            else if ( next->fifonum > num )
            {
                svmvolatile fifolist *afternext = next;
                MEMNEW(next,fifolist(num));
                next->next = afternext;
            }

            res = &(next->getnum(num));
        }

        return *res;
    }

};













class awarestream;

// This function opens a Unix socket server.  The filename (socketname) is
// generated and put in sockname, and a pointer to the socket returned.  If
// fixname then sockname is instead given by the caller and *must* be used!
//
// dellisten = 0 connect now
//             1 connect when used
// fixname   = 0 generate socket name that is not in use
//           = 1 use sockname given
// _sercli   = 0 for client
//             1 for server
//
// TCPIP version sockname == "" means use "127.0.0.1"

awarestream *makeUnixSocket(std::string &sockname, int dellisten = 0, int fixname = 0, int _sercli = 1);
void delUnixSocket(awarestream *sock);

awarestream *makeTCPIPSocket(const std::string &server_url, int port, int dellisten = 0, int _sercli = 1);
void delTCPIPSocket(awarestream *sock);

// Examples of stream binding:
//
// awarestream sbuf(...);
//
// std::istream sin(&sbuf);
// std::ostream sout(&sbuf);
//
// std::string a;
// gentype b;
//
// sout << a;
// sin >> b;

#define FIFO_BASESLEEP         100000
#define FIFO_ADDSLEEP_RAND     10000

// Note that we inherit from streambuf to make this trivially streamable

class awarestream : public std::streambuf
{
public:

    // Constructor for standard istream type
    //
    // _str       = stream that data will be coming from
    // _deletable = 0 if stream is to be left open always (eg std::cin)
    //              1 if stream should be closed once done (eg filestream)
    //
    // (_std default was std::cin)

    awarestream(std::istream *_str = NULL, int _deletable = 0)
    {
        if ( _str == NULL )
        {
            srcsel         = 0;
            str            = NULL;
            ostr           = NULL;
            deletable      = 0;
            odeletable     = 0;
            port           = 0;
            sock           = 0;
            feedback       = 0;
            serverorclient = -1; // 1 = server, 0 = client
            server_url     = "";
            socktype       = -1;
            streamtype     = SVM_AF_INET;
            sun_path       = "";
            fifoind        = -1;
            dellisten      = 0;
        }

        else
        {
            srcsel         = 1;
            str            = _str;
            ostr           = NULL;
            deletable      = _deletable;
            odeletable     = 0;
            port           = 0;
            sock           = 0;
            feedback       = 0;
            serverorclient = -1; // 1 = server, 0 = client
            server_url     = "";
            socktype       = -1;
            streamtype     = SVM_AF_INET;
            sun_path       = "";
            fifoind        = -1;
            dellisten      = 0;
        }
    }

    awarestream(const char *dummy, std::ostream *_ostr = NULL, int _odeletable = 0)
    {
        if ( !_ostr )
        {
            _ostr = &outstream();
        }

        (void) dummy;

        srcsel         = 1;
        str            = NULL;
        ostr           = _ostr;
        deletable      = 0;
        odeletable     = _odeletable;
        port           = 0;
        sock           = 0;
        feedback       = 1;
        serverorclient = -1; // 1 = server, 0 = client
        server_url     = "";
        socktype       = -1;
        streamtype     = SVM_AF_INET;
        sun_path       = "";
        fifoind        = -1;
        dellisten      = 0;
    }

    awarestream(std::istream *_istr, std::ostream *_ostr, int _ideletable, int _odeletable)
    {
        srcsel         = 1;
        str            = _istr;
        ostr           = _ostr;
        deletable      = _ideletable;
        odeletable     = _odeletable;
        port           = 0;
        sock           = 0;
        feedback       = 1;
        serverorclient = -1; // 1 = server, 0 = client
        server_url     = "";
        socktype       = -1;
        streamtype     = SVM_AF_INET;
        sun_path       = "";
        fifoind        = -1;
        dellisten      = 0;
    }

    // Constructor for UDP/TCP control.
    //
    // _port     = port to listen to
    // _socktype = SVM_SOCK_STREAM if connecting over TCP (default)
    //           = SVM_SOCK_DGRAM if connecting over UDP
    // _dellisten = 0 wait for client to connect here
    //              1 delay wait until other server call occurs (servers only)
    //
    // If socket cannot be bound then this will revert to no stream and throw an error
    //
    // The first version sets up as a server and waits for connection from a client,
    // the second is a client that attempts to connect to a server at address _server_url.
    //
    // If server_url == "" then it is set "127.0.0.1" (loopback)

    awarestream(int _port, int _socktype = SVM_SOCK_STREAM, int _feedback = 0, int _dellisten = 0)
    {
        srcsel         = 2;
        str            = NULL;
        ostr           = NULL;
        deletable      = 1;
        odeletable     = 1;
        port           = _port;
        sock           = 0;
        feedback       = _feedback;
        serverorclient = 1; // 1 = server, 0 = client
        server_url     = "";
        socktype       = _socktype;
        streamtype     = SVM_AF_INET;
        sun_path       = "";
        fifoind        = -1;
        dellisten      = serverorclient ?_dellisten : 0;

        if ( connectTCPUDP() )
        {
            throw("Server setup fail.");
        }

        return;
    }

    awarestream(const std::string &_server_url, int _port, int _socktype = SVM_SOCK_STREAM, int _feedback = 0, int _dellisten = 0)
    {
        std::string loc_server_url(_server_url);

        if ( loc_server_url == "" )
        {
            loc_server_url = "127.0.0.1";
        }

        srcsel         = 2;
        str            = NULL;
        ostr           = NULL;
        deletable      = 1;
        odeletable     = 1;
        port           = _port;
        sock           = 0;
        feedback       = _feedback;
        serverorclient = 0; // 1 = server, 0 = client
        server_url     = loc_server_url;
        socktype       = _socktype;
        streamtype     = SVM_AF_INET;
        sun_path       = "";
        fifoind        = -1;
        dellisten      = serverorclient ?_dellisten : 0;

        if ( connectTCPUDP() )
        {
            throw("Client setup fail.");
        }

        return;
    }

    // Constructor for unix socket control.
    //
    // _addrpath  = path/filename of unix socket control
    // _socktype  = SVM_SOCK_STREAM if connecting over TCP
    //              SVM_SOCK_DGRAM if connecting over UDP (default)
    // _feedback  = 0 uni-directional
    //              1 bi-directional
    // _sercli    = 0 for client
    //              1 for server
    // _dellisten = 0 wait for client to connect here
    //              1 delay wait until other server call occurs (servers only)
    //
    // If socket cannot be bound then this will revert to no stream and throw an error

    awarestream(const char *dummy, std::string _addrpath, int _socktype = SVM_SOCK_DGRAM, int _feedback = 0, int _sercli = 1, int _dellisten = 0)
    {
	(void) dummy;

        srcsel         = 3;
        str            = NULL;
        ostr           = NULL;
        deletable      = 1;
        odeletable     = 1;
        port           = 0;
        sock           = 0;
        feedback       = _feedback;
        serverorclient = _sercli; // 1 = server, 0 = client
        server_url     = "";
        socktype       = _socktype;
        streamtype     = SVM_AF_UNIX;
        sun_path       = _addrpath;
        fifoind        = -1;
        dellisten      = serverorclient ?_dellisten : 0;

        if ( connectTCPUDP() )
        {
            throw("Server setup fail.");
        }

        return;
    }

    // Constructor for shared string fifo (cross-thread comms)
    //
    // _fifoind = which fifo is used

    awarestream(const char *dummy, const char *dummyb, int _fifoind)
    {
	(void) dummy;
	(void) dummyb;

        NiceAssert( _fifoind >= 0 );

        srcsel         = 4;
        str            = NULL;
        ostr           = NULL;
        deletable      = 0;
        odeletable     = 0;
        port           = 0;
        sock           = 0;
        feedback       = 0;
        serverorclient = -1; // 1 = server, 0 = client
        server_url     = "";
        socktype       = -1;
        streamtype     = SVM_AF_INET;
        sun_path       = "";
        fifoind        = _fifoind;
        dellisten      = 0;

        return;
    }

    // Copy constructor.  This will always throw as there is no defined concept of copy here

//    awarestream(const awarestream &src)
//    {
//        *this = src;
//
//        return;
//    }

    // Destructor

    ~awarestream()
    {
        if ( deletable && ( srcsel == 1 ) )
        {
            MEMDEL(str);
        }

        if ( odeletable && ( srcsel == 1 ) )
        {
            MEMDEL(ostr);
        }

        if ( ( srcsel == 2 ) || ( srcsel == 3 ) )
        {
            disconnectTCPUDP();
        }

        return;
    }

    // Assignment operator - will always throw as assignment ill-defined

    awarestream &operator=(const awarestream &src)
    {
        (void) src;

        throw "Cannot duplicate awarestream";

        return *this;
    }

    // vogon pipes a string to the output stream.
    //
    // Success return: 0
    // Fail return: 2 for udp fail
    //              3 for tcp fail
    //              1 for successful push onto killed fifo
    //             -1 for failed push onto fifo
    //             -2 for failed push onto killed fifo

    int vogon(const std::string &src)
    {
        int res = 0;

        // srcsel == 0: acts like /dev/null

        if ( dellisten )
        {
            if ( srcsel != 3 )
            {
                svm_listen(sock,1024);
                clilen = sizeof(*cli_addr);
            }

            if ( ( sock = svm_accept(sock,cli_addr,&clilen) ) < 0 )
            {
                srcsel = 0;
                return 1;
            }

            dellisten = 0;
        }

        if ( ( srcsel == 1 ) && ( ostr != NULL ) )
        {
            *ostr << src;
            res = 0;
        }

        else if ( ( ( srcsel == 2 ) || ( srcsel == 3 ) ) && feedback )
        {
            if ( svm_send(sock,src.c_str(),strlen(src.c_str()),0) < 0 )
            {
                // Close socket, revert type and return an error

                disconnectTCPUDP();
                res = srcsel;
            }

            else
            {
                res = 0;
            }
        }

        else if ( srcsel == 4 )
        {
            svm_mutex_lock(fifolock);

            res = strfifo.push(fifoind,src);

            svm_mutex_unlock(fifolock);
        }

        return res;
    }

    // Skim pipes a string off the input stream.
    //
    // Success return: 0
    // Fail return: 2 for udp fail
    //              3 for tcp fail
    //              4 for attempted read from output only stream
    //              1 for successful pop from killed fifo
    //             -1 for failed pop from fifo
    //             -2 for failed pop from killed fifo

    int skim(std::string &dest)
    {
        int res = 0;

        if ( dellisten )
        {
            if ( srcsel != 3 )
            {
                svm_listen(sock,1024);
                clilen = sizeof(*cli_addr);
            }

            if ( ( sock = svm_accept(sock,cli_addr,&clilen) ) < 0 )
            {
                srcsel = 0;
                return 1;
            }

            dellisten = 0;
        }

        if ( srcsel == 0 )
        {
            dest = "";
            res = 4;
        }

        else if ( srcsel == 1 )
        {
            *str >> dest;
            res = 0;
        }

        else if ( ( srcsel == 2 ) || ( srcsel == 3 ) )
        {
            // If buffer is empty then listen until we receive something.  We then shove that into the buffer and continue

            dest = "";

            while ( dest.length() == 0 )
            {
                buffer >> dest;

                if ( !(buffer.good()) && ( dest.length() == 0 ) )
                {
                    buffer.clear();

                    int bytes_read = 0;
                    char recv_data[SVM_UDPBUFFERLEN];
                    struct svm_sockaddr_in client_addr;
                    svm_socklen_t addr_len = sizeof(struct svm_sockaddr);

                    // Wait until we get something (

                    while ( bytes_read == 0 )
                    {
                        if ( ( bytes_read = svm_recvfrom(sock,recv_data,SVM_UDPBUFFERLEN,0,(struct svm_sockaddr *) &client_addr,&addr_len) ) < 0 )
                        {
                            // Close socket, revert type and return an error

                            dest = "";
                            disconnectTCPUDP();
                            res = srcsel;
                            goto exitpoint;
                        }
                    }

                    // Null terminate the response and shove it into the buffer

                    recv_data[bytes_read] = '\0';

                    buffer << recv_data;
                }
            }

            res = 0;
        }

        else if ( srcsel == 4 )
        {
            int isdone = 0;

            while ( !isdone )
            {
                svm_mutex_lock(fifolock);

                int fifosizeval = strfifo.size(fifoind);

                if ( fifosizeval > 0 )
                {
                    // There are strings in the fifo, so pop one of them.

                    res = strfifo.pop(fifoind,dest);

                    svm_mutex_unlock(fifolock);

                    isdone = 1;
                }

                else if ( fifosizeval == 0 )
                {
                    // There are no strings in the fifo, but fifo live
                    // Unlock fifo, wait a random interval to give other
                    // threads a chance to push onto fifo or kill it, then
                    // retry.

                    svm_mutex_unlock(fifolock);

                    svm_usleep(FIFO_BASESLEEP+(svm_rand()%FIFO_ADDSLEEP_RAND));
                }

                else
                {
                    // fifo has kill-flag set, so treat it like a dead
                    // socket.

                    svm_mutex_unlock(fifolock);

                    dest = "";
                    res = fifosizeval;

                    isdone = 1;
                }
            }
        }

exitpoint:
        return res;
    }

    // Returns 1 if there (might be) more data in the buffer.  May not be
    // reliable for sockets

    int good(void)
    {
        int res = 0;

        if ( dellisten )
        {
            if ( srcsel != 3 )
            {
                svm_listen(sock,1024);
                clilen = sizeof(*cli_addr);
            }

            if ( ( sock = svm_accept(sock,cli_addr,&clilen) ) < 0 )
            {
                srcsel = 0;
                return 1;
            }

            dellisten = 0;
        }

        if ( srcsel == 0 )
        {
            res = 0;
        }

        else if ( srcsel == 1 )
        {
            res = str->good();
        }

        else if ( ( srcsel == 2 ) || ( srcsel == 3 ) )
        {
            res = 1;
        }

        else if ( srcsel == 4 )
        {
            svm_mutex_lock(fifolock);

            res = ( strfifo.size(fifoind) >= 0 ) ? 1 : 0;

            svm_mutex_unlock(fifolock);
        }

        return res;
    }

    // Streaming stuff

    virtual std::streambuf::int_type underflow()
    {
        if ( gptr() == egptr() )
        {
            std::string tmpbuf;

            if ( skim(tmpbuf) )
            {
                return std::streambuf::traits_type::eof();
            }

            tmpbuf += "\n"; // Make sure the buffer gets flushed!

            std::streamsize size = tmpbuf.size();

            strncpy(inputbuffer,tmpbuf.c_str(),tmpbuf.size());
            setg(inputbuffer,inputbuffer,inputbuffer+size);
        }

        return ( gptr() == egptr() ) ? std::streambuf::traits_type::eof() : std::streambuf::traits_type::to_int_type(*gptr());
    }

    virtual std::streambuf::int_type overflow(std::streambuf::int_type c)
    {
        int sizeis = pptr()-pbase();
        int i;

        for ( i = 0 ; i < sizeis ; i++ )
        {
            outputbuffer += pbase()[i];
        }

        // Newline or \0 flushes the buffer to the socket/stream

        if ( ( c == '\n' ) || ( c == '\0' ) )
        {
            if ( outputbuffer.length() )
            {
                outputbuffer += '\n';

                if ( vogon(outputbuffer) )
                {
                    return std::streambuf::traits_type::eof();
                }

                outputbuffer = "";
            }
        }

        else
        {
            outputbuffer += c;
        }

        dummybuffer[0] = std::streambuf::traits_type::to_char_type(c);
        setp(dummybuffer,dummybuffer); // Keep this empty!

        return traits_type::not_eof(c);
    }

    // Kill one or all string fifos.  Will affect all threads using fifos.

    void killfifo(int num)
    {
        svm_mutex_lock(fifolock);

        strfifo.setkillflag(num);

        svm_mutex_unlock(fifolock);

        return;
    }

    void killfifo(void)
    {
        svm_mutex_lock(fifolock);

        strfifo.setkillflag();

        svm_mutex_unlock(fifolock);

        // Sleep long enough to ensure that all threads waiting for
        // data on fifo receive kill signal

        svm_usleep(5*(FIFO_BASESLEEP+FIFO_ADDSLEEP_RAND));

        return;
    }

private:

    int srcsel;    // 0 none, 1 stream, 2 UDP, 3 TCP, 4 shared fifo
    int feedback;  // means no passing back upstream for UDP/TCP, 1 means allow feedback upstream
    int dellisten; // 0 normal, 1 means that server has not yet had a connection established to it so need to listen etc.
    std::istream *str;
    std::ostream *ostr;
    int deletable;
    int odeletable;
    int port;
    int sock;
    int serverorclient; // 1 = server, 0 = client
    std::string server_url;
    int socktype;
    std::stringstream buffer;
    int streamtype;
    std::string sun_path;
    int fifoind;
    svmvolatile static svm_mutex fifolock;
    svmvolatile static fifolist strfifo;

    char inputbuffer[2048];
    char dummybuffer[2];
    std::string outputbuffer;

    int connectTCPUDP(void)
    {
        return baseconnectTCPUDP();
    }

    // global because dellisten

    svm_socklen_t clilen;
    struct svm_sockaddr *cli_addr;

    int baseconnectTCPUDP(void)
    {
	struct svm_sockaddr_in server_addr_ip;
	struct svm_sockaddr_un server_addr_un;
	struct svm_sockaddr *server_addr;
        int server_addr_size;

	// Create a socket - domain SVM_AF_INET (IP) or SVM_AF_UNIX (unix sockets)
	//                 - type of service SVM_SOCK_DGRAM (UDP) or SVM_SOCK_STREAM (TCP)
	//                 - protocol 0 (not used)

	if ( ( sock = svm_socket(streamtype,socktype,0) ) < 0 )
	{
	    srcsel = 0;
	    return 1;
	}

	// Identify (name) the socket and bind it

	memset((char *) &server_addr,0,sizeof(server_addr));

	server_addr_ip.sin_family = streamtype;                                     // Address family used
	server_addr_ip.sin_port   = svm_htons(port);                                // Port to listen to
	server_addr_un.sun_family = streamtype;                                     // Address family used
        strncpy_safe(server_addr_un.sun_path,sun_path.c_str(),SVM_UNIX_PATH_MAX-1); // path for unix stream

	server_addr      = ( streamtype == SVM_AF_INET ) ? ( (struct svm_sockaddr *) &server_addr_ip ) : ( (struct svm_sockaddr *) &server_addr_un );
	server_addr_size = ( streamtype == SVM_AF_INET ) ? sizeof(struct svm_sockaddr_in) : sizeof(struct svm_sockaddr_un);

        if ( serverorclient )
        {
            // server

#ifndef VISUAL_STU
            server_addr_ip.sin_addr.s_addr = svm_htonl(SVM_INADDR_ANY); // Allow connection from all
#endif

            if ( svm_bind(sock,server_addr,server_addr_size) )
            {
                srcsel = 0;
                return 1;
            }

            if ( socktype == SVM_SOCK_STREAM )
            {
                //svm_socklen_t clilen;
                struct svm_sockaddr_in cli_addr_ip;
                struct svm_sockaddr_un cli_addr_un;

		//struct svm_sockaddr *cli_addr = ( streamtype == SVM_AF_INET ) ? ( (struct svm_sockaddr *) &cli_addr_ip ) : ( (struct svm_sockaddr *) &cli_addr_un );
		cli_addr = ( streamtype == SVM_AF_INET ) ? ( (struct svm_sockaddr *) &cli_addr_ip ) : ( (struct svm_sockaddr *) &cli_addr_un );


                if ( srcsel == 3 )
                {
                    svm_listen(sock,1024);
                    clilen = sizeof(*cli_addr);
                }

                if ( !dellisten )
                {
                    if ( srcsel != 3 )
                    {
                        svm_listen(sock,1024);
                        clilen = sizeof(*cli_addr);
                    }

                    if ( ( sock = svm_accept(sock,cli_addr,&clilen) ) < 0 )
                    {
                        srcsel = 0;
                        return 1;
                    }
                }
            }
        }

        else
        {
            // client

#ifndef VISUAL_STU
            server_addr_ip.sin_addr.s_addr = svm_inet_addr(server_url.c_str()); // Specify connection to server
#endif

            if ( socktype == SVM_SOCK_STREAM )
            {
                if ( svm_connect(sock,server_addr,server_addr_size) < 0 )
                {
                    srcsel = 0;
                    return 1;
                }
            }
        }

        return 0;
    }

    void disconnectTCPUDP(void)
    {
        try { svm_shutdown(sock,SVM_SHUT_RDWR); } catch (...) { ; }
        try { svm_close(sock);                  } catch (...) { ; }

        srcsel = 0;

        return;
    }
};

inline awarestream *&setident (awarestream *&a) { throw("Whatever"); return a; }
inline awarestream *&setzero  (awarestream *&a) { return a = NULL; }
inline awarestream *&setposate(awarestream *&a) { return a; }
inline awarestream *&setnegate(awarestream *&a) { throw("I reject your reality and substitute my own"); return a; }
inline awarestream *&setconj  (awarestream *&a) { throw("Mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"); return a; }
inline awarestream *&setrand  (awarestream *&a) { throw("Blippity Blappity Blue"); return a; }
inline awarestream *&postProInnerProd(awarestream *&a) { return a; }


inline void qswap(awarestream *&a, awarestream *&b);
inline void qswap(awarestream *&a, awarestream *&b)
{
    awarestream *x;

    x = a; a = b; b = x;

    return;
}


#endif
