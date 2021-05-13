#!/bin/bash

n=1
echo 'mex    -v COMPFLAGS="$COMPFLAGS -O2" ' ${!n} ' *.obj' >> mexmake.m
