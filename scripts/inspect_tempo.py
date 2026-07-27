import netCDF4 as nc
from pathlib import Path

# 1. Setup path to the first sample file
BASE_DIR = Path(__file__).resolve().parent.parent
tempo_dir = BASE_DIR / 'data' / 'raw' / 'tempo' / 'NO2_L3_V03'

sample_file = sorted(list(tempo_dir.rglob('*.nc4')))[0]
print(f"📡 Deep Inspecting Groups in: {sample_file.name}\n" + "="*60)

def print_group_structure(group, path="/"):
    print(f"\n📁 GROUP: {path}")
    print("-" * 40)
    
    # Print variables in this group
    if not group.variables:
        print("  (No variables in this group)")
    else:
        for var_name, var in group.variables.items():
            long_name = getattr(var, 'long_name', getattr(var, 'standard_name', 'No description'))
            units = getattr(var, 'units', 'no units')
            print(f"  📊 [{var_name}] -> {long_name} ({units}) | Dims: {var.dimensions}")
            
    # Recursively check sub-groups
    for subgroup_name, subgroup in group.groups.items():
        new_path = f"{path}{subgroup_name}/" if path == "/" else f"{path}/{subgroup_name}/"
        print_group_structure(subgroup, new_path)

# Open and scan the NetCDF4 file
with nc.Dataset(sample_file, 'r') as root:
    print_group_structure(root)