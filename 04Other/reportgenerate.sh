cd /app/Deeploy/DeeployTest
# core 1 SB
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_8 --cores=1 --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_8_SB_1_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_16 --cores=1 --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_16_SB_1_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_32 --cores=1 --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_32_SB_1_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_64 --cores=1 --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_64_SB_1_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_128 --cores=1 --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_128_SB_1_report.txt

python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_8_SB_1_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_8_SB_1_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_16_SB_1_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_16_SB_1_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_32_SB_1_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_32_SB_1_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_64_SB_1_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_64_SB_1_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_128_SB_1_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_128_SB_1_report.csv

# core 8 SB
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_8 --cores=8  --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_8_SB_8_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_16 --cores=8  --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_16_SB_8_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_32 --cores=8 --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_32_SB_8_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_64 --cores=8 --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_64_SB_8_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_128 --cores=8 --defaultMemLevel L3 --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_128_SB_8_report.txt

# core 8 DB
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_8 --cores=8 --defaultMemLevel L3 --doublebuffer  --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_8_DB_8_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_16 --cores=8 --defaultMemLevel L3 --doublebuffer  --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_16_DB_8_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_32 --cores=8 --defaultMemLevel L3 --doublebuffer --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_32_DB_8_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_64 --cores=8 --defaultMemLevel L3 --doublebuffer --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_64_DB_8_report.txt
python testRunner_tiled_siracusa.py -t Tests/testTrainCCT/CCT_Classifier_Training/CCT_1_16_16_128 --cores=8 --defaultMemLevel L3 --doublebuffer --profileTiling  > /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_128_DB_8_report.txt
 
# csv generation

python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_8_SB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_8_SB_8_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_16_SB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_16_SB_8_report.csv    
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_32_SB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_32_SB_8_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_64_SB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_64_SB_8_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_128_SB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_128_SB_8_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_8_DB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_8_DB_8_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_16_DB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_16_DB_8_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_32_DB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_32_DB_8_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_64_DB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_64_DB_8_report.csv
python /app/Onnx4Deeploy/Report/script/benchmark_parser.py /app/Onnx4Deeploy/Report/CCT_GEMM_FPU_1_16_16_128_DB_8_report.txt /app/Onnx4Deeploy/Report/csv/CCT_GEMM_FPU_1_16_16_128_DB_8_report.csv
