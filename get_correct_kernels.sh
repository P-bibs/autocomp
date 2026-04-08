cat $1 | grep -B 1 "passed correctness" | grep -v "\-\-" | grep -v "correctness" | awk '{print $14}' | awk -F "=" '{print $2}'
