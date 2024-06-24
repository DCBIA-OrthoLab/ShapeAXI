import argparse

import pandas as pd

from shapeaxi.saxi_dataset import SaxiDataset 


def main(args):
    
    
    df = pd.read_csv(args.csv)
    
    ds = SaxiDataset(df, mount_point=args.mount_point, batch_size=1, num_workers=0, surf_column="surf_path", CN=False)

    v_num = []
    f_num = []
    for V, F in ds:
        v_num.append(V.shape[0])
        f_num.append(F.shape[0])

    df["num_verts"] = v_num
    df["num_faces"] = f_num
    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ShapeAxis training')
    parser.add_argument('--csv', type=str, help='path to the csv file')
    parser.add_argument('--mount_point', type=str, help='path to the mount point', default="./")
    parser.add_argument('--out', type=str, help='path to the output', default="out.csv")
    args = parser.parse_args()
    main(args)