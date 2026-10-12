import json
import os
import struct
import zipfile
import glob
import numpy as np
import pandas as pd
import rasterio


def read_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_points(path, label_field="flooded"):
    rows = []
    skipped = 0
    for ft in read_geojson(path)["features"]:
        geom = ft.get("geometry")
        props = ft.get("properties") or {}
        if geom is None or geom.get("type") != "Point" or props.get(label_field) is None:
            skipped += 1
            continue
        x, y = geom["coordinates"][:2]
        rows.append({
            "fid": props.get("fid"),
            "lon": x,
            "lat": y,
            "label": int(props[label_field]),
        })
    return pd.DataFrame(rows), skipped

def boundary_geoms(path):
    data = read_geojson(path)
    return [ft["geometry"] for ft in data["features"] if ft.get("geometry")]

def read_tif(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
        epsg = src.crs.to_epsg() if src.crs is not None else None
        return arr, transform, epsg, src.nodata

def find_input(path, pattern, dest):
    if path and path.lower().endswith(".zip") and os.path.exists(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        path = dest

    if path and os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, "**", pattern), recursive=True))
        return hits[0] if hits else None

    return path if path and os.path.exists(path) else None

def read_dbf(path):
    with open(path, "rb") as f:
        h = f.read(32)
        n = struct.unpack("<I", h[4:8])[0]
        hl, rl = struct.unpack("<HH", h[8:12])
        fields = []
        while True:
            d = f.read(32)
            if d[0] == 0x0D:
                break
            fields.append((d[:11].split(b"\0")[0].decode(), chr(d[11]), d[16]))
        f.seek(hl)
        recs = []
        for _ in range(n):
            r = f.read(rl)
            pos = 1
            rec = {}
            for name, t, ln in fields:
                rec[name] = r[pos:pos + ln].decode("utf8", "replace").strip()
                pos += ln
            recs.append(rec)
    return fields, recs

def read_shp(path):
    data = open(path, "rb").read()
    pos = 100
    shapes = []
    while pos < len(data):
        num, clen = struct.unpack(">ii", data[pos:pos+8])
        pos += 8
        chunk = data[pos:pos + clen * 2]
        pos += clen * 2

        if struct.unpack("<i", chunk[:4])[0] == 0:
            shapes.append(None)
            continue

        bbox = struct.unpack("<4d", chunk[4:36])
        npart, npt = struct.unpack("<ii", chunk[36:44])
        parts = list(struct.unpack(f"<{npart}i", chunk[44:44 + 4 * npart]))
        pts = np.frombuffer(chunk[44 + 4 * npart:44 + 4 * npart + 16 * npt], dtype="<f8").reshape(-1, 2)
        shapes.append((bbox, [pts[parts[i]:(parts[i + 1] if i + 1 < npart else npt)] for i in range(npart)]))
    return shapes

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_csv(path, df):
    df.to_csv(path, index=False)
