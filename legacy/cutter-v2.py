import os
import csv
from obspy import read
from multiprocessing import Pool, cpu_count
from EQTransformer.utils.hdf5_maker import preprocessorV3

LOG_QUEUE = []  # global log queue

"""
segment_item: tuple (Stream, seg_name, preproc_dir, stations_json)

Processes a single Stream segment and returns a tuple:
(segment name, status, error message)
"""
def worker(segment_item):

    from traceback import format_exc

    segment, seg_name, preproc_dir, stations_json = segment_item
    status = "success"
    message = ""

    try:
        # Passing the Stream directly to preprocessorV3
        preprocessorV3(
            preproc_dir=preproc_dir,
            stream_list=[(segment, seg_name)],
            stations_json=stations_json
        )
    except Exception as e:
        status = "error"
        message = format_exc()  # Detailed error trace
        print(f"Processing error {seg_name}: {message}")

    return (seg_name, status, message)


# Split into 5-minute chunks
def geofile_splitter(base_directory, target_month=1, target_year=2024):
    files = os.listdir(base_directory)
    segments = []

    for file_path in files:
        full_path = os.path.join(base_directory, file_path)
        try:
            st = read(full_path)
        except Exception as e:
            print(f"Reading error {full_path}: {e}")
            continue

        tr = st[0]
        start_time = tr.stats.starttime
        end_time = tr.stats.endtime

        if start_time.month != target_month or start_time.year != target_year:
            continue

        t = start_time
        while t < end_time:
            t_next = t + 5 * 60
            segment = st.slice(t, t_next)
            if len(segment) == 0 or segment[0].stats.npts == 0:
                t = t_next
                continue
            seg_name = f"{file_path}__{t.strftime('%Y%m%dT%H%M%SZ')}__{min(t_next, end_time).strftime('%Y%m%dT%H%M%SZ')}"
            segments.append((segment, seg_name))
            t = t_next

    print(f"Total segments to process: {len(segments)}")
    return segments


# Parallel Processing
def preproc_parallel(segments, preproc_dir, stations_json):
    os.makedirs(preproc_dir, exist_ok=True)
    num_threads = max(1, cpu_count() - 1)

    # We form an input list for the worker
    task_list = [(seg, name, preproc_dir, stations_json) for seg, name in segments]

    with Pool(num_threads) as pool:
        results = pool.map(worker, task_list)

    # Save logs
    with open("./processing_log.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "status", "message"])
        writer.writerows(results)

    print(f"The processing log has been saved.: ./processing_log.csv")
    print("Processing completed.")


# main func
def main(base_directory, preproc_dir, stations_json):
    segments = geofile_splitter(base_directory)
    if segments:
        preproc_parallel(segments, preproc_dir, stations_json)


if __name__ == "__main__":
    base_directory = "./data-v2/input/ANN"
    preproc_dir = "./data-v2/preproc"
    stations_json = "./json/station_ANN.json"
    main(base_directory, preproc_dir, stations_json)
