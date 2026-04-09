set -e
# strip first argument
# shift

files=$(echo ./output/*)

echo "level problem compilation_failure correctness_failure correctness_passed total_runs"
# Loop through each argument
for arg in $files; do
    level=$(basename $arg | awk -F "_" '{print $2, $3}')
    a=$(cat $arg/auto-comp* | grep "Kernel evaluation failed" | wc -l)
    b=$(cat $arg/auto-comp* | grep "Kernel had compilation failure" | wc -l)
    c=$(cat $arg/auto-comp* | grep "timed out" | wc -l)
    d=$(cat $arg/auto-comp* | grep "Kernel did not pass correctness" | wc -l)
    e=$(cat $arg/auto-comp* | grep "Kernel passed correctness" | wc -l)
    f=$(cat $arg/auto-comp* | grep "Running command on GPU" | wc -l)
    echo "$level $a $b $c $d $e $f"
done