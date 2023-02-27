# docker run -it --rm --runtime=nvidia --volume="$PWD:/workspace" --ipc=host --privileged --name haznitrama-summtwice haznitrama/summtwice bash
docker run -it --rm --runtime=nvidia --volume="$PWD:/workspace" --ipc=host --privileged --name haznitrama-summtwice pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel bash
