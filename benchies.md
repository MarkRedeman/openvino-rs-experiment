# Benchmark scratch notes

```
openvino-rs-experiment main ? ✗ ./standalone/run-inference.sh infer --model ./models/card-classification/model.xml   --weights ./models/card-classification/model.bin   --image ./images/diamond-card.jpg   --task benchmark   --width 224   --height 224   --benchmark-warmup 20   --benchmark-duration 10 --device=cpu

openvino-rs-experiment main ? ✗ ./standalone/run-inference.sh infer --model ./models/card-classification/model.xml   --weights ./models/card-classification/model.bin   --image ./images/diamond-card.jpg   --task benchmark   --width 224   --height 224   --benchmark-warmup 20   --benchmark-duration 10 --device=gpu

```

```
./standalone/run-inference.sh \
  infer \
  --model models/fish-detection/model.xml \
  --weights models/fish-detection/model.bin \
  --image images/fish.png \
  --task benchmark \
  --detection-format geti \
  --threshold 0.3 \
  --width 992 \
  --height 800 \
  --preprocess-backend openvino \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --benchmark-stage-iters 300 \
  --benchmark-stage-read-each-iter \
  --device=cpu
  
  
./standalone/run-inference.sh \
  infer \
  --model models/fish-detection/model.xml \
  --weights models/fish-detection/model.bin \
  --image images/fish.png \
  --task benchmark \
  --detection-format geti \
  --threshold 0.3 \
  --width 992 \
  --height 800 \
  --preprocess-backend openvino \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --benchmark-stage-iters 300 \
  --benchmark-stage-read-each-iter \
  --device=gpu
```



./standalone/run-inference.sh \
  infer \
  --model models/card-classification/model.xml \
  --weights models/card-classification/model.bin \
  --image images/diamond-card.jpg \
  --task classify \
  --top-k 4 \
  --device=npu

./standalone/run-inference.sh \
  infer \
  --task act \
  --model models/act-openvino/act.xml \
  --weights models/act-openvino/act.bin \
  --episode-dir episodes/ep_000_dc7198bd \
  --output-json output/act-output-standalone.json
  
  
  ./standalone/run-inference.sh \
  infer \
  --task act \
  --model models/act-openvino/act.xml \
  --weights models/act-openvino/act.bin \
  --episode-dir episodes/ep_000_dc7198bd \
  --output-json output/act-output-standalone.json
  
  
  ```bash
  docker compose run --rm inference-npu \
  infer \
  --model /models/fish-detection-fp16/model.xml \
  --weights /models/fish-detection-fp16/model.bin \
  --image /images/fish.png \
  --task benchmark \
  --detection-format geti \
  --threshold 0.3 \
  --width 992 \
  --height 800 \
  --preprocess-backend openvino \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --benchmark-stage-iters 300 \
  --benchmark-stage-read-each-iter \
  --device 'HETERO:NPU,CPU'
  ```

  ```bash
docker compose run --rm inference-npu \
  infer \
  --task act \
  --model /models/act-openvino/act.xml \
  --weights /models/act-openvino/act.bin \
  --episode-dir /episodes/ep_000_dc7198bd \
  --device 'HETERO:NPU,CPU' \
  --output-json /output/act-output-standalone.json
  ```

  ./standalone/run-inference.sh \
  infer \
  --task act \
  --model models/act-openvino/act.xml \
  --weights models/act-openvino/act.bin \
  --episode-dir episodes/ep_000_dc7198bd \
  --output-json output/act-output-standalone.json


```
openvino-rs-experiment main  ? ❯ docker compose run --rm inference-npu infer --task act   --model /models/act-openvino/act.xml   --weights /models/act-openvino/act.bin   --episode-dir /episodes/ep_000_dc7198bd   --device 'NPU'
Loading ACT model: /models/act-openvino/act.xml (device: NPU)
ACT output: chunk_size=100, action_dim=6
step[000] = [1.6992188, -67.125, 74.875, 71.1875, 58.3125, 0.64160156]
step[001] = [1.5820313, -67.1875, 75.625, 71.125, 58.40625, 0.7475586]
step[002] = [1.6298828, -67.0625, 74.8125, 71.0625, 58.375, 0.64160156]
step[003] = [1.5585938, -67.125, 74.5, 71.0625, 58.3125, 0.6870117]
step[004] = [1.7832031, -67.125, 74.875, 71.125, 58.3125, 0.671875]
step[005] = [1.4003906, -66.8125, 74.625, 71.0625, 58.28125, 0.6567383]
step[006] = [1.3632813, -67.0, 74.625, 71.0625, 58.25, 0.64160156]
step[007] = [1.5878906, -67.25, 74.875, 71.0625, 58.375, 0.6113281]
step[008] = [1.6503906, -67.0, 75.0625, 71.1875, 58.34375, 0.77783203]
step[009] = [1.9189453, -67.0, 74.8125, 71.125, 58.40625, 0.671875]
... (90 more actions)
```


./standalone/run-inference.sh \
  infer \
  --task act-benchmark \
  --model models/act-openvino/act.xml \
  --weights models/act-openvino/act.bin \
  --episode-dir episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --output-json output/act-benchmark.json

```
docker compose run --rm inference-npu \
  infer \
  --task act-benchmark \
  --model /models/act-openvino/act.xml \
  --weights /models/act-openvino/act.bin \
  --episode-dir /episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --device 'NPU'
```

```
docker compose run --rm inference-gpu \
  infer \
  --task act-benchmark \
  --model /models/act-openvino/act.xml \
  --weights /models/act-openvino/act.bin \
  --episode-dir /episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --device 'GPU'
```


## Act with FP16

```
CPU:
docker compose run --rm inference \
  infer \
  --task act-benchmark \
  --model /models/act-openvino-fp16/act.xml \
  --weights /models/act-openvino-fp16/act.bin \
  --episode-dir /episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --device 'CPU'
GPU:
docker compose run --rm inference-gpu \
  infer \
  --task act-benchmark \
  --model /models/act-openvino-fp16/act.xml \
  --weights /models/act-openvino-fp16/act.bin \
  --episode-dir /episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --device 'GPU'
NPU:
docker compose run --rm inference-npu \
  infer \
  --task act-benchmark \
  --model /models/act-openvino-fp16/act.xml \
  --weights /models/act-openvino-fp16/act.bin \
  --episode-dir /episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --device 'NPU'
```
