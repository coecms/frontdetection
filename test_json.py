import json
import pandas as pd

with open(f"test_zeropoints.json", "r") as testfile:
    jdump = json.load(testfile)
df = pd.DataFrame(jdump["along_lat"], columns=["lat", "lon"])
print(df)

with open(f"test_zeropoints2.json", "r") as testfile:
    jdump_oldvers = json.load(testfile)
df_oldvers = pd.DataFrame(jdump_oldvers["along_lat"], columns=["lat", "lon"])
sort_oldvers = df_oldvers.sort_values(by=["lat", "lon"], ascending=[False, True])
print(sort_oldvers)

print(sort_oldvers.equals(df))
