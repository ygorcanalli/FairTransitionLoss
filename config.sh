docker build -t aif360-image -f aif360/Dockerfile .
docker run -it --gpus all \
-p 8888:8888 -p 6006:6006 -d --rm  \
-v $(pwd)/notebooks:/notebooks --name \
aif360-container aif360-image
