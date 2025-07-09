#!/bin/bash

for item in ./*; do
  if [ -d "$item" ]; then
    ls $item/volumes
  fi
done