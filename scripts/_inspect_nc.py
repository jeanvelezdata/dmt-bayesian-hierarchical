import xarray as xr
import sys
nc = sys.argv[1] if len(sys.argv) > 1 else "results/idata.nc"
ds = xr.open_dataset(nc, group="posterior")
print("Variables:")
for var in ds.data_vars:
    print(f"  {var}: shape={ds[var].shape}, dims={ds[var].dims}")
print("Coords:")
for k, v in ds.coords.items():
    print(f"  {k}: size={len(v)}, values={list(v.values)}")
ds.close()
