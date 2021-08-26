b="$@" 
for v in ${b[@]} 
    do 
        python3 train.py -p $v;
    done
