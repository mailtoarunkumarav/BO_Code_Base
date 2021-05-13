# -*- coding: utf-8 -*-
import socket
import os, os.path
import time
import sys
import numpy as np
import math
import shlex
import datetime

class Custom_Print(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self) :
        for f in self.files:
            f.flush()

class SVM_object:

    # get string from client.  Strings are separated by \n
    def getstring(self, client):
        res = ''
        while True:
            x = client.recv(1)
            if x == b'\n':
                break
            else:
                xx = str(x.decode('utf-8'))
                res += xx
        return res

    # send string to client.  Strings are separated by \n

    def sendstring(self, client, msg):
        rs += '\n'
        res = client.send(rs.encode('utf-8'))
        return res

    # convert string to vector.  note that dimension of result
    # may not necessarily match dim if vector is sparse.
    #
    # non-sparse format: [ x1 ; x2 ; ... ]
    # sparse format: [ i1:x1 ; i2:x2 ; ... ]
    # partially sparse format: [ i1:x1 ; x2 ; ... ; in:xn ; ... ]

    def strtovec(self, xstr, dim):
        res = np.zeros(dim)
        i = 0
        pos = 0
        # print("x (",dim,") = ",xstr)
        while i < dim:
            # skip whitespace/start
            while xstr[pos] == ' ' or xstr[pos] == '[':
                pos += 1
            # exit if vector finished
            if xstr[pos] == ']':
                break
            # grab string, possibly update position
            xnxt = ''
            while xstr[pos] != ' ':
                a = xstr[pos]
                pos += 1
                if a == ':':
                    i = int(xnxt)
                    xnxt = ''
                elif xstr[pos] != ' ':
                    xnxt += a
            # Convert, store
            if i >= dim:
                # print("resize here from ",dim," to ",i+1)
                res = np.resize(res, (i + 1))
                dim = i + 1
            res[i] = float(xnxt)
            # next...
            i += 1
        return res

    # NOTE: Kernel calculation function here
    # NOTE: If vectors are non-sparse then you can simplify this
    #      by making assumptions about vector dimensions

    # quadratic kernel
    #def evalK(self, xa, xb, ia, ib):
    #    xabprod = 0.0
    #    if xa.size < xb.size:
    #        xabprod = np.inner(xa, xb[0:xa.size])
    #    else:
    #        xabprod = np.inner(xa[0:xb.size], xb)
    #    res = pow(1 + xabprod, 2)
    #    return res

    ##SE kernel
    def evalK(xa,xb,ia,ib):
      res = 1
      if ia != ib:
        xanorm = np.inner(xa,xa)
        xbnorm = np.inner(xb,xb)
        xabprod = 0.0
        if xa.size < xb.size:
          xabprod = np.inner(xa,xb[0:xa.size])
        else:
          xabprod = np.inner(xa[0:xb.size],xb)
        res = math.exp(-(xanorm+xbnorm-(2*xabprod))/200)
      return res

    def start_svm(self):
        servername = 'kern2.sock'
        resfilename = 'resfile.txt'

        # delete socket if exists
        if os.path.exists(servername):
            os.remove(servername)

        # Construct kernel server
        print("Create socket server", servername, "...")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(servername)

        # Call svmheavy and leave it hanging:
        #  -Zdawkins is required as python messes with svmheavy's keypress detection code
        #  -c 1 sets c constant
        #  -kt 903 -kd 2 sets kernel to "ask on socket kern2.sock"
        #  -AA xor.txt tells svmheavy to load datafile xor.txt (SVM classification, target at start by default)
        #  -tr evaluates performance using recall
        print("Starting svmheavy")
        os.system("./svmheavyv7.exe -Zdawkins -c 1 -kt 903 -kd 2 -AA xor.txt -tx > resfile.txt &")

        # Listen for server and connect
        server.listen()
        client, addr = server.accept()
        print("Client is awake...")

        while True:
            # query incoming
            m = self.getstring(client)
            # print("m = ",m," (should be 2 or stop, if anything else something has gone wrong)")

            if m == 'stop':
                print("stop received, exiting")
                client.close()
                break;

            else:
                t = self.getstring(client)
                # print("t = ",t," (should be 9xx)")
                ns = self.getstring(client)
                n = int(ns)
                # print("n = ",n," (dimension)")
                d = self.getstring(client)
                # print("d = ",d," (ignore)")
                r = self.getstring(client)
                # print("r = ",r," (ignore)")
                i = self.getstring(client)
                # print("i = ",i," (ignore)")
                ia = self.getstring(client)
                # print("ia = ",ia," (first argument index)")
                ib = self.getstring(client)
                # print("ib = ",ib," (second argument index)")
                xsa = self.getstring(client)
                xa = self.strtovec(xsa, n)
                # print("xa = ",xa," (first argument vector)")
                xsb = self.getstring(client)
                xb = self.strtovec(xsb, n)
                # print("xb = ",xb," (second argument vector)")

                # Calculate K(xa,xb), send to server
                rs = str(self.evalK(xa, xb, ia, ib))
                client.send(rs.encode('utf-8'))

        # get result
        resfile = open(resfilename, "r")
        res = resfile.readline()
        print("result: ", res)
        resfile.close()

        return res

if __name__ == '__main__':

    timenow = datetime.datetime.now()
    stamp = timenow.strftime("%H%M%S_%d%m%Y")
    f = open('console_output_' + str(stamp) + '.txt', 'w')
    original = sys.stdout
    sys.stdout = Custom_Print(sys.stdout, f)
    svm_obj = SVM_object()
    svm_acc = svm_obj.start_svm()
    print("this is SVM :",svm_acc)





