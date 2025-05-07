import iris
import numpy as np
from iris.analysis import Linear

def print_info(cube, label=""):
    print(f"\n{label} cube:\n{cube}")
    print(f"  coords: {[coord.name() for coord in cube.coords()]}")
    print(f"  attributes: {cube.attributes}")
    print(f"  time: {cube.coord('time')}")
    print(f"  forecast_reference_time: {cube.coord('forecast_reference_time')}")
    print(f"  forecast_period: {cube.coord('forecast_period')}")
    print(f"  time units: {cube.coord('time').units}")
    print("-" * 60)

def harmonize_coord_metadata(cube1, cube2, coord_name):
    """Remove trailing '_0' from var_name on each cube's coordinate if present."""
    coord1 = cube1.coord(coord_name)
    coord2 = cube2.coord(coord_name)

    if coord1.var_name and coord1.var_name.endswith('_0'):
        coord1.var_name = coord1.var_name[:-2]

    if coord2.var_name and coord2.var_name.endswith('_0'):
        coord2.var_name = coord2.var_name[:-2]

def compare_and_concatenate_cubes(cubes_1, cubes_2):
    if len(cubes_1) != len(cubes_2):
        raise ValueError("Cubelists have different lengths.")

    updated_cubes_1 = []
    updated_cubes_2 = []
    concatenated = []

    for i, (cube1, cube2) in enumerate(zip(cubes_1, cubes_2)):
        print(f"\n--- Comparing cube {i} ---")

        # Print initial info
        print_info(cube1, label="Cube 1")
        print_info(cube2, label="Cube 2")

        # Harmonize coordinates if needed
        harmonize_coord_metadata(cube1, cube2, "time")
        harmonize_coord_metadata(cube1, cube2, "forecast_reference_time")
        harmonize_coord_metadata(cube1, cube2, "forecast_period")

        # Update lists
        updated_cubes_1.append(cube1)
        updated_cubes_2.append(cube2)

    # Try to concatenate
    cube_list = iris.cube.CubeList(updated_cubes_1 + updated_cubes_2)
    concatenated = cube_list.concatenate()

    return updated_cubes_1, updated_cubes_2, concatenated

cubes_1 = iris.load('/data/scratch/lewis.blunn/ml_mass_retrieval/temp_out_dir/20240716T1800Z/processed_20240716T1800Z_Paris_pmv_CCIv2_RAL3p2_DSMURK_SMFIX_MORUSES_em18_pa000.cutout.nc')
cubes_2 = iris.load('/data/scratch/lewis.blunn/ml_mass_retrieval/temp_out_dir/20240716T1800Z/processed_20240716T1800Z_Paris_pmv_CCIv2_RAL3p2_DSMURK_SMFIX_MORUSES_em18_pa012.cutout.nc')

updated_1, updated_2, concatenated = compare_and_concatenate_cubes(cubes_1, cubes_2)

for i, cube in enumerate(concatenated):
    print(f"\nConcatenated cube {i}:\n{cube.summary()}")



















"""
def print_info(cubes, index):
    cube = cubes[index]
    print(f"      cube:\n",cube)
    print(f"      coords:\n", [coord.name() for coord in cube.coords()])
    print(f"      attributes:\n", cube.attributes)
    print(f"      time:\n", cube.coord('time'))
    print(f"      forecast_reference_time':\n", cube.coord('forecast_reference_time'))
    print(f"      time units:\n", cube.coord('time').units)

cubes_1 = iris.load('/data/scratch/lewis.blunn/ml_mass_retrieval/temp_out_dir/20240716T1800Z/processed_20240716T1800Z_Paris_pmv_CCIv2_RAL3p2_DSMURK_SMFIX_MORUSES_em18_pa000.cutout.nc')
print("original cubes 1:\n",cubes_1)
print_info(cubes_1,1)

cubes_2 = iris.load('/data/scratch/lewis.blunn/ml_mass_retrieval/temp_out_dir/20240716T1800Z/processed_20240716T1800Z_Paris_pmv_CCIv2_RAL3p2_DSMURK_SMFIX_MORUSES_em18_pa012.cutout.nc')
print("original cubes 2:\n",cubes_2)
print_info(cubes_2,1)

cubes_c = iris.cube.CubeList(cubes_1 + cubes_2).concatenate()
print("concatenated cubes:\n",cubes_c)
"""
              
