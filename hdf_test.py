import pandas as pd

try:
    print("Trying hdf")
    # f = h5.File("/mnt/nvme0n1/Datasets/SingleCellFromNathan_
    # 17122021/TransformerFeats/1B2.h5", 'r')
    df = pd.read_hdf(
        "/mnt/nvme0n1/Datasets/SingleCellFromNathan_17122021/" "TransformerFeats/1B2.h5"
    )
    print("Worked")
except Exception as e:
    print(e)
    print("fuck")

print(df)
