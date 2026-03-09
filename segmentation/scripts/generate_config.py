import json
import argparse
from pathlib import Path

def correction(start, end, values, corr):
    for i in range(start, end+1):
        corr[i] = {
            'b':values[0],
            'window':values[1],
            'search_radius':values[2]
        }
    return corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=Path, help="File name for json with path.")
    args = parser.parse_args()

    file = args.file
    
    corr = {}
    corr = correction(start=1041, end=1044, values=(5,5,5), corr=corr)
    corr = correction(start=1097, end=1100, values=(20,10,20), corr=corr)
    corr = correction(start=1117, end=1118, values=(10,5,10), corr=corr)
    corr = correction(start=1138, end=1140, values=(10,5,10), corr=corr)
    corr = correction(start=1148, end=1148, values=(10,5,10), corr=corr)
    corr = correction(start=1159, end=1161, values=(10,5,10), corr=corr)
    corr = correction(start=1184, end=1184, values=(10,5,10), corr=corr)
    corr = correction(start=1186, end=1187, values=(10,5,10), corr=corr)
    corr = correction(start=1191, end=1194, values=(10,5,10), corr=corr)
    corr = correction(start=1210, end=1213, values=(10,5,10), corr=corr)
    corr = correction(start=1216, end=1218, values=(10,5,10), corr=corr)
    corr = correction(start=1236, end=1241, values=(20,10,20), corr=corr)
    corr = correction(start=1247, end=1253, values=(20,10,20), corr=corr)
    corr = correction(start=1255, end=1256, values=(20,10,20), corr=corr)
    corr = correction(start=1262, end=1266, values=(20,10,20), corr=corr)
    corr = correction(start=1275, end=1280, values=(20,10,20), corr=corr)
    print(corr)

    corr_keys = list(corr.keys())

    
    config = {}
    for i in range(1000, 3000, 1):
        if i in corr_keys:
            config[i] = corr[i]
        else:
            config[i]={
                'b':50,
                'window':50,
                'search_radius':20
            }

    with open(file=file, mode='w') as f:
        json.dump(obj=config, fp=f)

if __name__ == "__main__":
    main()