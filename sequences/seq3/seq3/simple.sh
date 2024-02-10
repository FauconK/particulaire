#!/bin/bash

j=134

for i in *.png ;
 do
   echo $i
    mv $i im$j.png
   j=`expr $j + 1`

 done
