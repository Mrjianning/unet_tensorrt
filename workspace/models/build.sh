trtexec --onnx=./seg_xray_96_96_2.0.onnx   --fp16 --minShapes=input:1x3x96x96 --optShapes=input:256x3x96x96 --maxShapes=input:256x3x96x96 --saveEngine=./seg_xray_96_96.engine  --workspace=1024 