
docker rmi -f $(docker images -qf "dangling=true")
docker build -t testlander -f Dockerfile.test .


time docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:rw --privileged -v /home/daqi/v3/week11/hw/data/videos:/tmp/videos testlander
