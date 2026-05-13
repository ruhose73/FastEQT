import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from EQTransformer.utils.associator import run_associator

input_dir = 'detections_hdfs'
output_dir = 'geo-files-association'

input_dir_exist = os.path.exists(input_dir)
output_dir_exist = os.path.exists(output_dir)

start_time = '2024-01-01 00:00:00.000'
end_time = '2024-02-01 00:00:00.000'

if input_dir_exist and output_dir_exist:
    print("Запуск ассоциативного модуля")
    run_associator(input_dir=input_dir, output_dir=output_dir, start_time=start_time, end_time=end_time, consider_combination=False, pair_n=3)
else: 
    print("INPUT", input_dir)
    print("OUTPUT", output_dir)