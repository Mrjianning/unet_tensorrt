# unet_tensorrt
unet 使用 TensorRT 加速部署

## 环境
- ubuntu20.04
- Tesorrt8.x
- cuda12.0
- opencv4.8.0

## 编译
```bash
mkdir build && cd build
cmake ..
make -j8
```

- 编译生成的文件在worksoace/bin

## 运行
进入worksoace/bin执行可执行文件

