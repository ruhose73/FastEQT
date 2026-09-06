import os
import sys

sys.path.insert(0, os.getcwd())
from EQTransformer.utils.associator import run_associator_v2

if os.path.exists("phase_dataset"):
    os.remove("phase_dataset")

run_associator_v2(
    input_dir="memory-25/assoc_input_filt",
    output_dir="memory-25/assoc_output_filt",
    start_time="2025-01-01 00:00:00.000",
    end_time="2025-04-01 00:00:00.000",
    moving_window=50,
    consider_combination=False,
    pair_n=6,
    coherence_tolerance=3.0,
)

if os.path.exists("phase_dataset"):
    os.remove("phase_dataset")

run_associator_v2(
    input_dir="memory-25/assoc_input_cpu_2_filt",
    output_dir="memory-25/assoc_output_cpu_2_filt",
    start_time="2025-04-01 00:00:00.000",
    end_time="2025-07-01 00:00:00.000",
    moving_window=50,
    consider_combination=False,
    pair_n=6,
    coherence_tolerance=3.0,
)

print("Готово: memory-25/assoc_output_filt и memory-25/assoc_output_cpu_2_filt")
