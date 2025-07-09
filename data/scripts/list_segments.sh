#!/bin/bash

for item in ./*; do
  if [ -d "$item" ]; then
    for subfolder in "$item"/paths/*; do
      if [ -d "$subfolder" ] && [[ "$(basename "$subfolder")" == *frangi_gaussian_dijkstraP* ]]; then 
        echo "$(basename "$subfolder")"
      fi
    done
  fi
done