from EQTransformer.core.mseed_predictor import mseed_predictor
from EQTransformer.utils.downloader import downloadMseeds
from EQTransformer.utils.downloader import makeStationList
from EQTransformer.utils.hdf5_maker import preprocessor
from EQTransformer.core.predictor import predictor
import os

json_basepath = os.path.join(os.getcwd(),"json/station_list.json")

# makeStationList(json_path=json_basepath2, client_list=["SCEDC"], min_lat=35.50, max_lat=35.60, min_lon=-117.80, max_lon=-117.40, start_time="2019-09-01 00:00:00.00", end_time="2019-09-03 00:00:00.00", channel_list=["HH[ZNE]", "HH[Z21]", "BH[ZNE]"], filter_network=["SY"], filter_station=[])
# downloadMseeds(client_list=["SCEDC", "IRIS"], stations_json=json_basepath2, output_dir="downloads_mseeds", min_lat=35.50, max_lat=35.60, min_lon=-117.80, max_lon=-117.40, start_time="2019-09-01 00:00:00.00", end_time="2019-09-03 00:00:00.00", chunk_size=1, channel_list=[], n_processor=2)

# Количество логических процессоров (потоков)
num_threads = os.cpu_count()
print(f"Количество потоков: {num_threads}")

preprocessor(preproc_dir="preproc",
             mseed_dir='geo-files', 
             stations_json=json_basepath, 
             overlap=0.3, 
             n_processor=num_threads)