import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from EQTransformer.core.predictor import predictor

predictor(input_dir= 'geo-files_processed_hdfs', input_model='ModelsAndSampleData/EqT_original_model.h5', output_dir='detections_hdfs', detection_threshold=0.8, P_threshold=0.6, S_threshold=0.5, number_of_plots=1000, plot_mode='time', use_multiprocessing=False,gpuid=0, output_probabilities=True, estimate_uncertainty=True, batch_size=500)