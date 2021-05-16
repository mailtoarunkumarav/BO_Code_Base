#!/bin/bash

n=1
echo 'mex -c -v COMPFLAGS="$COMPFLAGS -O2" ' ${!n} >> mexmake.m
