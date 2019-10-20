#!/bin/bash
expName=exp2;
startIter=250;
saveIter=5;
entIter=500;
genName=checkpoints/${expName}_${saveIter}_net_G.t7;
disName=checkpoints/${expName}_${saveIter}_net_D.t7;

mkdir output/${expName}
rm -r /src/cache/*

luarocks install optnet

for ((a=$startIter; a <= $entIter ; a+=$saveIter));
do
    if (($a==0))
    then
        echo "DATA_ROOT=myimages dataset=folder display=0 niter=$saveIter saveIter=$saveIter name=${expName} th main.lua";
        DATA_ROOT=myimages dataset=folder display=0 niter=$saveIter saveIter=$saveIter name=${expName} batchSize=128 th main.lua
    elif (($a<$entIter))
    then
        echo "net=${genName} name=output/${expName}/${a} display=0 th generate.lua";
        net=${genName} name=output/${expName}/${a} display=0 th generate.lua
        
        echo "DATA_ROOT=myimages dataset=folder netD=${disName} netG=${genName} display=0 niter=$saveIter saveIter=$saveIter name=$expName th main.lua";
        DATA_ROOT=myimages dataset=folder netD=${disName} netG=${genName} display=0 niter=$saveIter saveIter=$saveIter name=$expName batchSize=128 th main.lua
    else
        echo "net=${genName} name=output/${expName}/${a} display=0 th generate.lua";
        net=${genName} name=output/${expName}/${a} display=0 th generate.lua
    fi
    
done
