# Copies over the necessary driver files into ./drivers. 
# During docker build, files in ./drivers are copied into the container.
# This is needed to run visii in evaluate.py 
cd drivers/
cp /usr/lib/x86_64-linux-gnu/libnvoptix.so.1 .
cp /usr/lib/x86_64-linux-gnu/*nvidia* .
