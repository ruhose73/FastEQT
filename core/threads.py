import os
import math

def get_threads_to_use(percent): 
    value = percent / 100
    num_threads = os.cpu_count()
    used_threads = math.ceil(num_threads * value)
    print('-' * 100)
    print("Всего потоков:", num_threads)
    print("Используется потоков:", used_threads)
    print('-' * 100)
    return used_threads

print(get_threads_to_use(80))