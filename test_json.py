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

    col3 = "true_lat"
    sort_by = ["lon", "true_lat"]
    ascending_by = [True, False]  # Allows to have additional values at the end
    if out_type == "along_lat":
        col3 = "true_lon"
        sort_by = ["lat", "true_lon"]
        ascending_by = [False, True]  # Allows to have additional values at the end

    df = pd.DataFrame(newdump[out_type], columns=["lat", "lon", col3])
    sort_df = df.sort_values(by=sort_by, ascending=ascending_by)

    df_oldvers = pd.DataFrame(olddump[out_type], columns=["lat", "lon", col3])
    sort_oldvers = df_oldvers.sort_values(by=sort_by, ascending=ascending_by)

    sort_df.reset_index(drop=True, inplace=True),
    sort_oldvers.reset_index(drop=True, inplace=True),
    return (sort_df, sort_oldvers)


new_along_lat, old_along_lat = df_peroutput(jdump, jdump_oldvers, "along_lat")
new_along_lon, old_along_lon = df_peroutput(jdump, jdump_oldvers, "along_lon")

print(new_along_lon)
print(old_along_lon)
print(f"along_lat comparison: ", old_along_lat.equals(new_along_lat))
print(f"along_lon comparison: ", old_along_lon.equals(new_along_lon))

# Difference of along_lon
# All values are equal except for the additional values
# found by new version.
diff = new_along_lon - old_along_lon
diff_nozeros = diff.loc[~(diff == 0).all(axis=1)]
print(diff_nozeros)

# Difference of along_lat
# All values are equal except for the additional values
# found by new version.
diff = new_along_lat - old_along_lat
diff_nozeros = diff.loc[~(diff == 0).all(axis=1)]
print(diff_nozeros)
