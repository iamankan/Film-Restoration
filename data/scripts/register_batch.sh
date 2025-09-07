#!/bin/bash
# save as gen_numbers.sh and make it executable: chmod +x gen_numbers.sh

for i in $(seq -f "%03g" 1 15); do
    ./register.sh --id ${i}
done
