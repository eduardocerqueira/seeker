#date: 2025-08-08T16:51:44Z
#url: https://api.github.com/gists/269ec86255cb8c6434929630db83c115
#owner: https://api.github.com/users/datavudeja

import pandas as pd
import lxml.etree as etree
import pyarrow.parquet as pq
import h5py
import yaml
import json
import fastavro
from chardet import detect
from io import BytesIO

def ingest_data(filename):
    if filename.endswith('.csv'):
        return read_csv_with_encoding(filename)
    elif filename.endswith('.json'):
        return read_json_with_encoding(filename)
    elif filename.endswith('.xml'):
        return read_xml(filename)
    elif filename.endswith('.parquet'):
        return read_parquet(filename)
    elif filename.endswith('.xlsx') or filename.endswith('.xls'):
        return read_excel(filename)
    elif filename.endswith('.h5'):
        return read_hdf5(filename)
    elif filename.endswith('.avro'):
        return read_avro(filename)
    elif filename.endswith('.yaml') or filename.endswith('.yml'):
        return read_yaml(filename)
    else:
        raise ValueError(f"Unsupported file type for {filename}")

def read_csv_with_encoding(filename, chunk_size=10000):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = detect(raw_data)
        detected_encoding = result['encoding']
        confidence = result['confidence']
        print(f"Detected encoding: {detected_encoding} with confidence {confidence}")

    if confidence > 0.7:
        return pd.read_csv(filename, encoding=detected_encoding, chunksize=chunk_size)
    else:
        # Attempt common encodings
        for encoding in ['utf-8', 'utf-8-sig', 'utf-16', 'windows-1252', 'iso-8859-1']:
            try:
                return pd.read_csv(filename, encoding=encoding, chunksize=chunk_size)
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not determine the correct encoding for the file.")

def read_json_with_encoding(filename, chunk_size=10000):
    with open(filename, 'rb') as file:
        raw_data = file.read()
        result = detect(raw_data)
        detected_encoding = result['encoding']
        confidence = result['confidence']
        print(f"Detected encoding: {detected_encoding} with confidence {confidence}")

    if confidence > 0.7:
        with open(filename, 'r', encoding=detected_encoding) as file:
            data = json.load(file)
        return pd.DataFrame(data)
    else:
        # Attempt common encodings
        for encoding in ['utf-8', 'utf-8-sig', 'utf-16', 'windows-1252', 'iso-8859-1']:
            try:
                with open(filename, 'r', encoding=encoding) as file:
                    data = json.load(file)
                return pd.DataFrame(data)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        raise ValueError("Could not determine the correct encoding for the JSON file.")

def read_xml(filename):
    tree = etree.parse(filename)
    root = tree.getroot()
    data = []
    for child in root:
        row = {}
        for subchild in child.iterchildren():
            if subchild.text:
                row[subchild.tag] = subchild.text.strip()
        data.append(row)
    return pd.DataFrame(data)

def read_parquet(filename):
    return pd.read_parquet(filename)

def read_excel(filename):
    return pd.read_excel(filename)

def read_hdf5(filename):
    data_frames = {}
    with h5py.File(filename, 'r') as hdf:
        for key in hdf.keys():
            data_frames[key] = pd.read_hdf(filename, key=key)
    return data_frames

def read_avro(filename):
    with open(filename, 'rb') as fo:
        avro_reader = fastavro.reader(fo)
        records = list(avro_reader)
    return pd.DataFrame.from_records(records)

def read_yaml(filename):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
    json_data = json.dumps(data)
    return pd.read_json(json_data, orient='records')