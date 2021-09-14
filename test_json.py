import json
import pandas as pd

with open(f"test_zeropoints.json", "r") as testfile:
    jdump = json.load(testfile)
with open(f"test_zeropoints2.json", "r") as testfile:
    jdump_oldvers = json.load(testfile)


def df_peroutput(newdump, olddump, out_type):
    """
    newdump: json output from the new zeropoints version
    olddump: json output from the old zeropoints version
    out_type: ["along_lat" | "along_lon"]
    """

    df = pd.DataFrame(newdump[out_type], columns=["lat", "lon"])

    df_oldvers = pd.DataFrame(olddump[out_type], columns=["lat", "lon"])
    sort_oldvers = df_oldvers.sort_values(by=["lat", "lon"], ascending=[False, True])

    return (df, sort_oldvers)


new_along_lat, old_along_lat = df_peroutput(jdump, jdump_oldvers, "along_lat")
new_along_lon, old_along_lon = df_peroutput(jdump, jdump_oldvers, "along_lon")

print(new_along_lon)
print(old_along_lon)
print(f"along_lat comparison: ", old_along_lat.equals(new_along_lat))
print(f"along_lon comparison: ", old_along_lon.equals(new_along_lon))
