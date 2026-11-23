# Results

| run | model | test set | n | CER ↓ | WER ↓ | word acc ↑ | sentence acc ↑ | EM@1 / @3 / @5 ↑ | p50 ms | p95 ms | RSS MB | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control-tagger-t4-r1 | tagger:ds-tagger | doc_split/test | 200 | 0.0109 | 0.0457 | 0.9543 | 0.5650 | – / – / – | 14.17 | 26.67 | 290.6 | cpu |
| control-tagger-t4-r2 | tagger:ds-tagger | doc_split/test | 200 | 0.0109 | 0.0457 | 0.9543 | 0.5650 | – / – / – | 14.03 | 24.76 | 290.6 | cpu |
| control-tagger-t4-r3 | tagger:ds-tagger | doc_split/test | 200 | 0.0109 | 0.0457 | 0.9543 | 0.5650 | – / – / – | 10.97 | 21.85 | 289.9 | cpu |
| kd-int8-t1-r1 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 15.47 | 46.51 | 267.2 | cpu |
| kd-int8-t1-r2 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 18.34 | 54.33 | 267.3 | cpu |
| kd-int8-t1-r3 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 16.16 | 52.92 | 266.2 | cpu |
| kd-int8-t2-r1 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 11.77 | 21.39 | 268.3 | cpu |
| kd-int8-t2-r2 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 9.34 | 24.84 | 267.9 | cpu |
| kd-int8-t2-r3 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 17.56 | 36.71 | 267.6 | cpu |
| kd-int8-t4-r1 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 8.68 | 15.49 | 268.7 | cpu |
| kd-int8-t4-r2 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 6.33 | 10.68 | 270.9 | cpu |
| kd-int8-t4-r3 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 10.52 | 17.88 | 269.2 | cpu |
| kd-int8-t8-r1 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 7.88 | 12.13 | 268.9 | cpu |
| kd-int8-t8-r2 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 6.46 | 9.66 | 269.8 | cpu |
| kd-int8-t8-r3 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 8.0 | 12.01 | 269.5 | cpu |
| large-int8-t1-r1 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 39.4 | 103.89 | 303.1 | cpu |
| large-int8-t1-r2 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 38.42 | 106.66 | 305.4 | cpu |
| large-int8-t1-r3 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 28.31 | 49.05 | 302.3 | cpu |
| large-int8-t2-r1 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 29.0 | 58.26 | 303.8 | cpu |
| large-int8-t2-r2 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 30.47 | 67.81 | 305.0 | cpu |
| large-int8-t2-r3 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 22.0 | 34.25 | 304.8 | cpu |
| large-int8-t4-r1 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 20.21 | 36.99 | 304.1 | cpu |
| large-int8-t4-r2 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 14.19 | 22.79 | 305.2 | cpu |
| large-int8-t4-r3 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 22.85 | 39.37 | 304.5 | cpu |
| large-int8-t8-r1 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 15.5 | 23.17 | 305.5 | cpu |
| large-int8-t8-r2 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 13.69 | 21.01 | 305.7 | cpu |
| large-int8-t8-r3 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 14.98 | 22.46 | 304.8 | cpu |
