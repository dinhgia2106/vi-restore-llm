# Results

| run | model | test set | n | CER ↓ | WER ↓ | word acc ↑ | sentence acc ↑ | EM@1 / @3 / @5 ↑ | p50 ms | p95 ms | RSS MB | device |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control-tagger-t4-r1 | tagger:ds-tagger | doc_split/test | 200 | 0.0109 | 0.0457 | 0.9543 | 0.5650 | – / – / – | 9.99 | 18.03 | 289.9 | cpu |
| control-tagger-t4-r2 | tagger:ds-tagger | doc_split/test | 200 | 0.0109 | 0.0457 | 0.9543 | 0.5650 | – / – / – | 10.11 | 19.74 | 290.1 | cpu |
| control-tagger-t4-r3 | tagger:ds-tagger | doc_split/test | 200 | 0.0109 | 0.0457 | 0.9543 | 0.5650 | – / – / – | 10.72 | 18.22 | 289.5 | cpu |
| kd-int8-t4-r1 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 6.04 | 10.0 | 269.3 | cpu |
| kd-int8-t4-r2 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 6.14 | 10.0 | 269.4 | cpu |
| kd-int8-t4-r3 | tagger-onnx:ds-tagger-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0107 | 0.0442 | 0.9558 | 0.5650 | – / – / – | 6.36 | 10.06 | 268.2 | cpu |
| kd-onnx-t4-r1 | tagger-onnx:ds-tagger-kd/tagger.onnx | doc_split/test | 200 | 0.0108 | 0.0444 | 0.9556 | 0.5650 | – / – / – | 17.29 | 28.66 | 318.9 | cpu |
| kd-onnx-t4-r2 | tagger-onnx:ds-tagger-kd/tagger.onnx | doc_split/test | 200 | 0.0108 | 0.0444 | 0.9556 | 0.5650 | – / – / – | 17.24 | 29.68 | 318.6 | cpu |
| kd-onnx-t4-r3 | tagger-onnx:ds-tagger-kd/tagger.onnx | doc_split/test | 200 | 0.0108 | 0.0444 | 0.9556 | 0.5650 | – / – / – | 17.56 | 29.66 | 318.3 | cpu |
| kd-pt-t4-r1 | tagger:ds-tagger-kd | doc_split/test | 200 | 0.0108 | 0.0444 | 0.9556 | 0.5650 | – / – / – | 10.01 | 19.57 | 289.9 | cpu |
| kd-pt-t4-r2 | tagger:ds-tagger-kd | doc_split/test | 200 | 0.0108 | 0.0444 | 0.9556 | 0.5650 | – / – / – | 10.04 | 17.03 | 290.5 | cpu |
| kd-pt-t4-r3 | tagger:ds-tagger-kd | doc_split/test | 200 | 0.0108 | 0.0444 | 0.9556 | 0.5650 | – / – / – | 9.97 | 20.27 | 290.0 | cpu |
| large-int8-t4-r1 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 13.93 | 22.83 | 304.2 | cpu |
| large-int8-t4-r2 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 13.72 | 21.53 | 304.4 | cpu |
| large-int8-t4-r3 | tagger-onnx:ds-tagger-large/tagger.int8.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6000 | – / – / – | 13.88 | 23.01 | 304.6 | cpu |
| large-onnx-t4-r1 | tagger-onnx:ds-tagger-large/tagger.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6050 | – / – / – | 42.17 | 67.54 | 413.7 | cpu |
| large-onnx-t4-r2 | tagger-onnx:ds-tagger-large/tagger.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6050 | – / – / – | 43.62 | 70.74 | 413.5 | cpu |
| large-onnx-t4-r3 | tagger-onnx:ds-tagger-large/tagger.onnx | doc_split/test | 200 | 0.0087 | 0.0362 | 0.9638 | 0.6050 | – / – / – | 44.86 | 72.12 | 413.5 | cpu |
| s-kd-int8-t4-r1 | tagger-onnx:ds-tagger-s-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0137 | 0.0570 | 0.9430 | 0.5100 | – / – / – | 3.11 | 4.6 | 256.1 | cpu |
| s-kd-int8-t4-r2 | tagger-onnx:ds-tagger-s-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0137 | 0.0570 | 0.9430 | 0.5100 | – / – / – | 3.12 | 4.77 | 255.7 | cpu |
| s-kd-int8-t4-r3 | tagger-onnx:ds-tagger-s-kd/tagger.int8.onnx | doc_split/test | 200 | 0.0137 | 0.0570 | 0.9430 | 0.5100 | – / – / – | 3.04 | 4.7 | 256.0 | cpu |
| s-kd-onnx-t4-r1 | tagger-onnx:ds-tagger-s-kd/tagger.onnx | doc_split/test | 200 | 0.0137 | 0.0570 | 0.9430 | 0.5050 | – / – / – | 6.66 | 11.1 | 269.9 | cpu |
| s-kd-onnx-t4-r2 | tagger-onnx:ds-tagger-s-kd/tagger.onnx | doc_split/test | 200 | 0.0137 | 0.0570 | 0.9430 | 0.5050 | – / – / – | 6.8 | 11.13 | 269.8 | cpu |
| s-kd-onnx-t4-r3 | tagger-onnx:ds-tagger-s-kd/tagger.onnx | doc_split/test | 200 | 0.0137 | 0.0570 | 0.9430 | 0.5050 | – / – / – | 6.78 | 10.77 | 270.3 | cpu |
