#!/usr/bin/env bash
set -e

DATA_DIR="data/backgrounds"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading backgrounds..."
echo "  - COCO val2017"
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip
echo "  - DTD"
wget -c https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xzf dtd-r1.0.1.tar.gz

echo "Done."
echo "Datasets saved in:"
echo "  $DATA_DIR/val2017"
echo "  $DATA_DIR/dtd"