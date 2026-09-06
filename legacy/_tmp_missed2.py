import openpyxl, math

wb = openpyxl.load_workbook('Каталог Сочи 300 км.xlsx', read_only=True, data_only=True)
ws = wb.active

# Remaining missed events (NOT "insufficient stations")
targets = [
    (2024, 1,  1,  4,  5),
    (2024, 1, 17,  3, 52),
    (2024, 1, 23, 21, 30),
    (2024, 1, 23, 22, 31),
    (2024, 1, 24, 11, 19),
    (2024, 1, 25, 10, 33),
    (2024, 1, 25, 14,  2),
    (2024, 1, 25, 17, 35),
    (2024, 1, 25, 23, 39),
] 

reasons = {
    (1,  4,  5): 'NEUR poor quality (depth 49km)',
    (1, 17, 52): 'NEUR + distance 282km',
    (1, 23, 30): 'Swarm: 18s after 21:29',
    (1, 23, 31): 'Swarm/associator',
    (1, 24, 19): 'SPGR data gap',
    (1, 25, 33): 'DOMR nearest, no pair',
    (1, 25,  2): 'ZEI nearest, 301km',
    (1, 25, 35): 'NEUR poor quality',
    (1, 25, 39): 'No pair with GUZR',
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2-lat1); dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R*2*math.asin(math.sqrt(a))

stations = {
    'SOC':  (43.570, 39.763),
    'VSLR': (43.461, 40.032),
    'MRNR': (43.937, 39.479),
    'GUZR': (43.996, 40.118),
    'GOYR': (44.247, 39.377),
    'GRYR': (44.117, 41.094),
    'LABN': (44.641, 40.724),
    'DOMR': (43.292, 41.624),
    'SPGR': (44.742, 38.073),
    'SHA1': (43.738, 42.657),
    'BEYR': (44.012, 42.818),
    'NEUR': (43.263, 42.716),
    'ZEI':  (42.788, 43.901),
    'PYA1': (44.062, 43.096),
    'NCK':  (43.495, 43.596),
    'GOFR': (45.084, 43.048),
    'SRGR': (45.421, 39.169),
    'GLDR': (44.983, 37.721),
    'SUKR': (44.799, 37.429),
    'TMNR': (45.155, 36.785),
    'ANN':  (44.881, 37.314),
}
in_net = {'SOC','VSLR','GUZR','BEYR','SHA1','MRNR','SPGR','DOMR','ZEI','LABN','PYA1'}

for i, row in enumerate(ws.iter_rows(values_only=True)):
    if i == 0: continue
    if row[0] is None: continue
    t = row[0]
    if not hasattr(t, 'year'): continue
    if t.year != 2024 or t.month != 1: continue
    for (y,mo,d,h,mi) in targets:
        if t.day==d and t.hour==h and abs(t.minute-mi)<=2:
            lat, lon = float(row[1]), float(row[2])
            ms = float(row[4]) if row[4] else None
            depth = float(row[3]) if row[3] else 10.0
            key = (mo, h, mi)
            reason = reasons.get((mo, h, mi), reasons.get((mo, h, t.minute), '?'))
            print(f'\n=== {t}  Ms={ms:.2f}  depth={depth}km  [{reason}] ===')
            print(f'  Coords: {lat:.3f}N  {lon:.3f}E')
            dists = [(st, haversine(lat,lon,slat,slon)) for st,(slat,slon) in stations.items()]
            dists.sort(key=lambda x: x[1])
            print('  Nearest stations (* = in network incl. PYA1):')
            for st, d2 in dists[:8]:
                mark = '*' if st in in_net else ' '
                print(f'    {mark} {st:5s}: {d2:5.0f} km')

wb.close()
