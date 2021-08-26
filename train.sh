b="$@" 
if [ $1 == '-l' ] ; then
    python3 train.py -l;
else 
    for v in ${b[@]} 
        
        do 
            python3 train.py -p $v;
        done
fi