# About
This is a minimal example for getting Detectron2 (mainly instance segmentation) working on a M1 Mac via Docker.

# Docker Image
Get the Docker image by either:
1. pulling pre-built image:
    ```shell
    docker pull eight0153/detectron2:cpu
    ```
2. building the image from scratch:
    ```shell
    docker buildx build --platform linux/amd64 -t eight0153/detectron2:cpu .
    ```
   The option `--platform linux/amd64` is needed to get everything working on a M1 Mac.

# Running the Test Script
```shell
docker run -v $(pwd):/app -t eight0153/detectron2:cpu detectron_test.py --input_image sample.jpg --output_folder /app
```
This above command will output a binary mask called `sample_mask.png`.
