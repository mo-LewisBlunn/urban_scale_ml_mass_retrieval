import iris
import numpy as np
from iris.analysis import Linear

target_heights = [10.,100.,500.,1000., 1500.,2000.,3000.,4000.,6000.,8000.,10000.]

cubes = iris.load('/data/scratch/lewis.blunn/ml_mass_retrieval/temp_out_dir/20240716T1800Z/20240716T1800Z_Paris_pmv_CCIv2_RAL3p2_DSMURK_SMFIX_MORUSES_em00_pd000.cutout.pp')
print("original cubes:\n",cubes)

def interpolate_cube(cube, target_heights):
    return cube.interpolate([('level_height', target_heights)], Linear())

new_cubes = iris.cube.CubeList()
for i, cube in enumerate(cubes):
    print("cube:\n",cube)
    print(f"      coords={[coord.name() for coord in cube.coords()]}")
    print(f"      attributes={cube.attributes}")
    print(f"      member={cube.coord('time')}")
    print(f"      member={cube.coord('forecast_reference_time')}")
    print(f"      member={cube.coord('time').units}")
    iris.util.promote_aux_coord_to_dim_coord(cube, 'level_height')
    new_cube = interpolate_cube(cube, target_heights)
    new_cubes.append(new_cube)
    print("new_cube:\n",new_cube)
print("new_cubes:\n",new_cubes)

iris.save(new_cubes,'new_cubes.nc')

loaded_cubes = iris.load('new_cubes.nc')
print("loaded_cubes:\n",loaded_cubes)
              
