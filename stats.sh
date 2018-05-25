#!/bin/bash

set -e

THREADS=1

OUT="prox_to_graph"
IN="stats"

rm -f out

while IFS='' read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *"n_threads"* ]] && \
      THREADS="$(echo $line | cut -d' ' -f 5 | xargs)" \
      && continue
    ITER="$(echo $line | cut -d' ' -f 1)"
    TIME="$(echo $line | cut -d' ' -f 2)"
    OBJT="$(echo $line | cut -d' ' -f 3)"
    echo "$THREADS $ITER $TIME $OBJT"
    echo "$THREADS,$ITER,$TIME,$OBJT" >> $OUT
done < "$IN"
