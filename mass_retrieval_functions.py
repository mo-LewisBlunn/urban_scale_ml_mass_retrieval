import iris
import numpy as np
import tempfile
import subprocess
import os
import time
import re
import sys
import getpass
from config import Config
from datetime import datetime, timedelta
from iris.analysis import Linear
from iris.coords import DimCoord, AuxCoord
from iris.cube import CubeList
from collections import defaultdict
from iris.util import unify_time_units
from iris.util import equalise_attributes


### Functions ###

def get_land_cover(sim_dt, land_covers_dt):
    """
    Helper function to determine land cover based on sim date.
    """
    for cover, (start, end) in land_covers_dt.items():
        if start <= sim_dt <= end:
            return cover
    return None



def ancil_retrieval(ancil_mass_path, ancil_suites, ancil_models, temp_out_dir):
    """
    Retrieves the ancillary files from MASS.
    """

    for suite in ancil_suites:
        for mod in ancil_models:
            print(f"\nRetrieving ancils for suite: {suite}, mod: {mod} ...")
            
            # Ensure output directory exists
            output_dir = os.path.join(temp_out_dir, f'ancils_{suite}_{mod}')
            os.makedirs(output_dir, exist_ok=True)
            orog_srtm_filename = '/qrparm.orog.srtm.121x1'

            def ancil_retrieval_command(suite,mod,test_path,add_path):
                if not os.path.exists(test_path):
                    # Build and run the command to retrieve the files
                    cmd = ['moo', 'get', ancil_mass_path+f'{suite}_Paris_{mod}_L70/{add_path}', output_dir]
                    print(f"Retrieving ancillary files:\n{' '.join(cmd)}")

                    try:
                        subprocess.run(cmd, check=True)
                        print(f"Successfully retrieved files to: {output_dir}")
                    except subprocess.CalledProcessError as e:
                        print(f"Failed to retrieve ancillary files: {e}")
                        raise

                else:
                    print(f"Files already exists locally at {output_dir}")

            ancil_retrieval_command(suite,mod,test_path=output_dir+'/qrparm.mask_cci',add_path='*')
            ancil_retrieval_command(suite,mod,test_path=output_dir+orog_srtm_filename,add_path=f'orog_srtm/{orog_srtm_filename}')



def make_target_grid(temp_out_dir, target_grid_spacing, output_dir, suite, model, tol=1e-8):
    """
    Make the target_grid from the high-res orography cube. The 
    target_grid is based on the fixed resolution portion.
    Note: the target grid will remain rotated if the high-res orography 
    grid is. This reduces the amount of the high-res data regridding and 
    means that the high-res data won't have to be interpolated (unless
    it is staggered differently e.g., winds). The low-res data will be 
    regridded (and therefore interpolated) to this high-res target grid.   
    """

    cube = iris.load_cube(f'{temp_out_dir}/ancils_{suite}_{model}/qrparm.orog.srtm.121x1')

    # Detect coordinate names
    lat_name = 'latitude' if cube.coords('latitude') else 'grid_latitude'
    lon_name = 'longitude' if cube.coords('longitude') else 'grid_longitude'

    lat_coord = cube.coord(lat_name)
    lon_coord = cube.coord(lon_name)

    lat_vals = lat_coord.points
    lon_vals = lon_coord.points

    # Compute spacing (diffs)
    dlat = np.diff(lat_vals)
    dlon = np.diff(lon_vals)

    # Find where spacing matches target
    lat_mask = np.isclose(dlat, target_grid_spacing, atol=tol)
    lon_mask = np.isclose(dlon, target_grid_spacing, atol=tol)

    # Find start/end indices of matching blocks
    def get_bounds(mask):
        indices = np.where(mask)[0]
        if indices.size == 0:
            return None
        if not np.all(np.diff(indices) == 1):
            raise ValueError("Regular grid region is not contiguous.")
        return indices[0], indices[-1] + 1  # +1 for inclusive end

    lat_bounds = get_bounds(lat_mask)
    lon_bounds = get_bounds(lon_mask)

    if lat_bounds is None or lon_bounds is None:
        raise ValueError("No regular region found matching target_grid_spacing.")

    # Translate bounds to coordinate values
    min_lat = lat_vals[lat_bounds[0]]
    max_lat = lat_vals[lat_bounds[1]]
    min_lon = lon_vals[lon_bounds[0]]
    max_lon = lon_vals[lon_bounds[1]]

    print(f"Constraining to regular region:")
    print(f"  Latitude: {min_lat:.5f} to {max_lat:.5f}")
    print(f"  Longitude: {min_lon:.5f} to {max_lon:.5f}")

    # Use iris.Constraint to extract region
    constraints = iris.Constraint(
        **{
            lat_name: lambda x: min_lat <= x <= max_lat,
            lon_name: lambda x: min_lon <= x <= max_lon,
        }
    )
    cropped_cube = cube.extract(constraints)
    
    iris.save(cropped_cube, output_dir+'target_grid.nc')

    return cropped_cube



def process_ancils(grid_dir, temp_out_dir, log_dir, ancil_suites, ancil_models, ancil_stash_codes, ancil_lc_names, ancil_lai_names):
    """
    Extracts the ancillaries we want, splits out pseudo_level into separate cubes, and regrids them.
    """
    job_ids = []
    for suite in ancil_suites:
        for mod in ancil_models:
            print(f"\nProcessing ancils for suite: {suite}, mod: {mod} ...")

            ancil_dir = os.path.join(temp_out_dir, f'ancils_{suite}_{mod}')
            all_files = os.listdir(ancil_dir)

            exclude_files = {"text_files.tar", "qrparm.orog", "qrparm.veg.frac_cci"}
            ancil_files = [
                os.path.join(ancil_dir, f) for f in all_files
                if f not in exclude_files and os.path.isfile(os.path.join(ancil_dir, f))
            ]

            cubes = iris.cube.CubeList()
            for file in ancil_files:
                try:
                    cubes.extend(iris.load(file))
                except Exception as e:
                    print(f"Failed to load {file}: {e}")

            filtered_cubes = iris.cube.CubeList()
            for cube in cubes:
                stash_attr = cube.attributes.get('STASH')
                if stash_attr and stash_attr in ancil_stash_codes:
                    filtered_cubes.append(cube)

            final_cubes = iris.cube.CubeList()
            for cube in filtered_cubes:
                if cube.coords("pseudo_level", dim_coords=True):
                    levels = cube.coord("pseudo_level")
                    for i, level_value in enumerate(levels.points):
                        single_slice = cube.extract(iris.Constraint(pseudo_level=level_value))
                        if len(levels.points) == 10:
                            new_name = ancil_lc_names[i]
                        elif len(levels.points) == 5:
                            new_name = ancil_lai_names[i]
                        else:
                            new_name = f"{cube.name()}_{i}"
                        single_slice.rename(new_name)
                        final_cubes.append(single_slice)
                else:
                    final_cubes.append(cube)

            print(f"Final cube count after splitting pseudo-levels: {len(final_cubes)}")

            output_path = os.path.join(temp_out_dir, f"ancils_{suite}_{mod}_filtered.nc")
            iris.save(final_cubes, output_path)

            regrid_script = f"regrid_{suite}_{mod}.sh"
            log_dir_ancils = os.path.join(log_dir, "ancils")
            os.makedirs(log_dir_ancils, exist_ok=True)
            log_file = os.path.join(log_dir_ancils, f"regrid_{suite}_{mod}.log")
            regridded_output_path = os.path.join(temp_out_dir, f"ancils_{suite}_{mod}.nc")

            with open(regrid_script, 'w') as f:
                f.write(f"""#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --output={log_file}
#SBATCH --time=20

module load scitools

python <<EOF
import iris
from iris.analysis import Linear
import os

input_file = "{output_path}"
target_grid = iris.load_cube("{grid_dir}/target_grid.nc")
cubes = iris.load(input_file)

regridded = iris.cube.CubeList()
for cube in cubes:
    regridded.append(cube.regrid(target_grid, Linear()))

iris.save(regridded, "{regridded_output_path}")
print("Regridding complete.")
os.remove(input_file)
EOF

rm {regrid_script}
""")

            os.chmod(regrid_script, 0o755)

            try:
                result = subprocess.run(['sbatch', regrid_script], capture_output=True, text=True, check=True)
                job_id = result.stdout.strip().split()[-1]
                job_ids.append(job_id)
                print(f"Submitted regridding job for ancils_{suite}_{mod}: Job ID {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to submit regridding job for ancils_{suite}_{mod}")
                print("STDOUT:", e.stdout)
                print("STDERR:", e.stderr)

    wait_for_jobs_to_finish(job_ids)



def retrieve_sim(log_dir, temp_out_dir, streams, members, sim, mod,
                 stash_codes_strs, cruns, suite, land_covers_dt):
    """
    Retrieve all mass files for a given simulation and model.
    """
    print(f"\nRetrieving mod: {mod}, sim: {sim} ...")
    sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")

    # Make the directories log_dir/sim and temp_out_dir/sim if they don't already exist
    sim_log_dir = os.path.join(log_dir, sim)
    sim_temp_out_dir = os.path.join(temp_out_dir, sim)
    os.makedirs(sim_log_dir, exist_ok=True)
    os.makedirs(sim_temp_out_dir, exist_ok=True)

    job_ids = []

    # Loop for retrieving individual mass files (each is a "job")
    for stream in streams:
        stash_codes_str = stash_codes_strs[stream]  # Select stash codes corresponding to the stream
        for mem in members:
            for crun in cruns[stream]:
                # Determine land cover
                lc = get_land_cover(sim_dt, land_covers_dt)
                if lc is None:
                    print(f"No land cover found for sim {sim}")
                    continue

                # Set the mass_file name
                mass_file = get_mass_file_name(sim, mod, lc, mem, stream, crun)

                # Retrieve mass_file from MASS
                job_id = mass_retrieval_job(stash_codes_str, suite, temp_out_dir, log_dir, sim, mass_file)
                if job_id:
                    job_ids.append(job_id)

    # Wait for all jobs to complete (note: completion could be due to success, failure, or timeout)
    wait_for_jobs_to_finish(job_ids)



def get_mass_file_name(sim, mod, lc, mem, stream, crun):
    """
    Build the expected mass_file name.
    """
    return f'{sim}_Paris_{mod}_{lc}_RAL3p2_DSMURK_SMFIX_MORUSES_em{mem}_{stream}{crun}.cutout.pp'



def mass_retrieval_job(stash_codes_str, suite, temp_out_dir, log_dir, sim, mass_file):
    """
    Submit a SLURM job to retrieve a mass file from MASS.
    """
    # File paths of MASS retrieval files and log files
    output_path = os.path.join(temp_out_dir, sim, mass_file)
    log_path = os.path.join(log_dir, sim, f"{mass_file}.retrieval.log")

    # Remove the MASS retrieval file if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Create a temporary file for the SLURM shell script
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as script_file:
        script_contents = f"""#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=1
#SBATCH --output={log_path}
#SBATCH --time=120

query_file=$(mktemp)
cat > $query_file <<EOF
begin
    pp_file = "{mass_file}"
    stash = ({stash_codes_str})
end
EOF

moo select $query_file moose:/devfc/{suite}/field.pp/ {temp_out_dir}{sim}
echo "finished {mass_file}"
"""
        script_file.write(script_contents)
        script_path = script_file.name

    # Make the script executable
    os.chmod(script_path, 0o755)

    # Submit the job using sbatch
    try:
        result = subprocess.run(['sbatch', script_path], capture_output=True, text=True, check=True)
        print(f"Submitted job for: {mass_file}")
        return result.stdout.strip().split()[-1]  # SLURM job ID
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job for {mass_file}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return None



def wait_for_jobs_to_finish(job_ids, poll_interval=30):
    """
    Wait until all submitted jobs have completed.
    """
    print(f"Waiting for {len(job_ids)} job(s) to finish...")
    print(job_ids)
    while True:
        try:
            result = subprocess.run(['squeue', '-u', getpass.getuser()], capture_output=True, text=True, check=True)
        except Exception as e:
            print(f"Error checking job status: {e}")
            break

        active_jobs = result.stdout
        still_running = [job_id for job_id in job_ids if job_id in active_jobs]

        if not still_running:
            print("All jobs completed.")
            break
        else:
            print(f"Jobs still running: {', '.join(still_running)}")
            time.sleep(poll_interval)



def validate_and_rerun_failed_jobs_for_sim(sim, mod, temp_out_dir, log_dir, stash_codes_strs,
                                            cruns, streams, members, suite, land_covers_dt, 
                                            max_retries=5, wait_time=2*60):
    """
    Validate log files and re-run failed jobs for a specific simulation.
    Retry up to `max_retries` times with `wait_time` seconds between attempts.
    Save a summary of failures for that simulation.
    Return True if successful, False if still failed after retries.
    """
    # find if the retrieved file exists or if the log says the job failed, and list the bad jobs
    retry_count = 0
    sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")
    summary_failures = []
    while retry_count <= max_retries:
        failed_jobs = []

        for stream in streams:
            stash_codes_str = stash_codes_strs[stream]
            for mem in members:
                for crun in cruns[stream]:
                    lc = get_land_cover(sim_dt, land_covers_dt)
                    if lc is None:
                        continue

                    mass_file = get_mass_file_name(sim, mod, lc, mem, stream, crun)
                    output_path = os.path.join(temp_out_dir, sim, mass_file)
                    log_path = os.path.join(log_dir, sim, f"{mass_file}.retrieval.log")

                    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                        failed_jobs.append((mod, stream, stash_codes_str, mem, crun, mass_file))
                        continue

                    if not os.path.exists(log_path):
                        failed_jobs.append((mod, stream, stash_codes_str, mem, crun, mass_file))
                        continue

                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        if not any(f"finished {mass_file}" in line for line in lines[-10:]):
                            failed_jobs.append((mod, stream, stash_codes_str, mem, crun, mass_file))

        if not failed_jobs:
            print(f"Log files checked -- simulation {sim} retrieval completed successfully.")
            return True

        if retry_count == max_retries:
            summary_failures = failed_jobs
            break

        if retry_count > 0:
            print(f"Retrieval problem detected. Waiting {wait_time} seconds before next retry...")
            time.sleep(wait_time)
        else:
            print(f"Retrieval problem detected.")
            
        # Make the directories log_dir/sim and temp_out_dir/sim if they don't already exist
        sim_log_dir = os.path.join(log_dir, sim)
        sim_temp_out_dir = os.path.join(temp_out_dir, sim)
        os.makedirs(sim_log_dir, exist_ok=True)
        os.makedirs(sim_temp_out_dir, exist_ok=True)
            
        print(f"Retry attempt {retry_count+1}/{max_retries}, retrying {len(failed_jobs)} failed retrieval job(s) for {sim}...")
        job_ids = []
        for mod, stream, stash_codes_str, mem, crun, mass_file in failed_jobs:
            job_id = mass_retrieval_job(stash_codes_str, suite, temp_out_dir, log_dir, sim, mass_file)
            if job_id:
                job_ids.append(job_id)
        wait_for_jobs_to_finish(job_ids)

        retry_count += 1

    # Write simulation-level summary
    summary_file = os.path.join(log_dir, sim, 'summary_failed_retrieval_jobs.log')
    with open(summary_file, 'w') as f:
        if summary_failures:
            for mod, stream, _, mem, crun, mass_file in summary_failures:
                f.write(f"{sim},{mod},{stream},{mem},{crun},{mass_file}\n")
            print(f"Simulation {sim} failed. Summary written to {summary_file}")
            return False
        else:
            f.write("No failures\n")
            print(f"Simulation {sim} completed successfully. Summary written to {summary_file}")




def process_retrieval(temp_out_dir, grid_dir, log_dir, sim, mod, streams, members, cruns, land_covers_dt, target_heights):
    """
    Submit batch jobs to process each retrieved .pp file and save as netcdf.
    Performs vertical and horizontal interpolation.
    """
    print(f"\nProcessing mod: {mod}, sim: {sim} ...")
    sim_temp_out_dir = os.path.join(temp_out_dir, sim)
    sim_log_dir = os.path.join(log_dir, sim)
    os.makedirs(sim_temp_out_dir, exist_ok=True)
    os.makedirs(sim_log_dir, exist_ok=True)

    sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")
    job_ids = []

    for stream in streams:
        for mem in members:
            for crun in cruns[stream]:
                lc = get_land_cover(sim_dt, land_covers_dt)
                if lc is None:
                    continue
                mass_file = get_mass_file_name(sim, mod, lc, mem, stream, crun)
                file_path = os.path.join(sim_temp_out_dir, mass_file)

                if not os.path.exists(file_path):
                    print(f"File not found (skipping): {file_path}")
                    continue

                log_path = os.path.join(sim_log_dir, f"{mass_file}.process.log")
                mass_file_nc = mass_file.replace(".pp", ".nc")
                output_file = os.path.join(sim_temp_out_dir, f"processed_{mass_file_nc}")

                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as script_file:
                    script_contents = f"""#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem=3G
#SBATCH --ntasks=1
#SBATCH --output={log_path}
#SBATCH --time=10

module load scitools

python <<EOF
import iris
import iris.coords as icoord
import numpy as np
from iris.analysis import Linear

target_heights = np.array({target_heights})
target_file = "{grid_dir}"+"target_grid.nc"
input_file = "{file_path}"
output_file = "{output_file}"

def interpolate_cube_vertical(cube, target_heights):
    return cube.interpolate([('level_height', target_heights)], Linear())
    
def interpolate_cube_horizontal(cube, target_grid):
    return cube.regrid(target_grid, Linear())

target_grid = iris.load_cube(target_file)  
cubes = iris.load(input_file)
new_cubes = iris.cube.CubeList()
for i, cube in enumerate(cubes):
    coords = [coord.name() for coord in cube.coords()]
    if "model_level_number" in coords and "level_height" in coords:
        iris.util.promote_aux_coord_to_dim_coord(cube, 'level_height')
        cube = interpolate_cube_vertical(cube, target_heights)
        cube = interpolate_cube_horizontal(cube, target_grid)
        new_cubes.append(cube)
    else:
        cube = interpolate_cube_horizontal(cube, target_grid)
        new_cubes.append(cube)

iris.save(new_cubes, output_file)
print(f"Saved processed cubes to {output_file}")

print(f"finished {output_file}")
EOF
"""
                    script_file.write(script_contents)
                    script_path = script_file.name

                os.chmod(script_path, 0o755)

                try:
                    result = subprocess.run(['sbatch', script_path], capture_output=True, text=True, check=True)
                    job_id = result.stdout.strip().split()[-1]
                    job_ids.append(job_id)
                    print(f"Submitted processing job for {mass_file}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed to submit processing job for {mass_file}")
                    print("STDOUT:", e.stdout)
                    print("STDERR:", e.stderr)

    wait_for_jobs_to_finish(job_ids)



def load_processed_cubelists(temp_out_dir, log_dir, sim, mod, streams, members, cruns, land_covers_dt):
    """
    Load processed cube files (saved as *.processed) and return cubelists by stream.
    """
    print(f"\nLoading processed mod: {mod}, sim: {sim} ...")
    sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")
    sim_temp_out_dir = os.path.join(temp_out_dir, sim)
    sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")
    cubelists_by_stream = {stream: [] for stream in streams}

    member_mapping = {f"{int(i):02d}": 1 if i == 0 else i - 16 for i in range(0, 35) if i == 0 or 18 <= i <= 34}

    for stream in streams:
        for mem in members:
            for crun in cruns[stream]:
                lc = get_land_cover(sim_dt, land_covers_dt)
                if lc is None:
                    continue

                mass_file = get_mass_file_name(sim, mod, lc, mem, stream, crun)
                mass_file_nc = mass_file.replace(".pp", ".nc")
                file_path = os.path.join(sim_temp_out_dir, f"processed_{mass_file_nc}")

                if os.path.exists(file_path):
                    try:
                        cubes = iris.load(file_path)

                        member_index = member_mapping.get(mem)
                        if member_index is None:
                            print(f"  Warning: member '{mem}' not found in mapping.")
                            member_index = -1

                        for cube in cubes:
                            coord = iris.coords.DimCoord(
                                member_index,
                                standard_name=None,
                                long_name='member',
                                units='1'
                            )
                            cube.add_aux_coord(coord)

                        cubelists_by_stream[stream].append(cubes)
                        print(f"Loaded {len(cubes)} cube(s) from {file_path} with member index {member_index}")
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
                else:
                    print(f"Processed file not found (skipped): {file_path}")

    print(f"Finished loading.")

    return cubelists_by_stream



def clean_and_join(cubelists_by_stream, sim, filter_start_hour, filter_end_hour):
    """
    Merge on scalar coordinate (member) and concatenate on time if it is a dimension coordinate.
    Also, cleans up differences in the time coordinates between cubes.
    Returns a dictionary: {stream: CubeList}
    """
    print("\nCleaning and joining ...")
    merged_by_stream = {}

    for stream, cubelists in cubelists_by_stream.items():
        merged_cubelist = iris.cube.CubeList()

        # Combine all cubes into one big CubeList
        for cubelist in cubelists:
            merged_cubelist.extend(cubelist)

        # merge
        merged = merged_cubelist.merge()
             
        # clean:
        # - times rounded to 5 min
        # - max wind gust moved to end of hour (and cell method deleted so the variable isn't removed in the next step)
        # - cubes with cell method are removed (since we don't want cubes with time processing)
        cleaned = iris.cube.CubeList()
        for cube in merged:
        
            # - round the time coordinate to the nearest 5 min (so they are on the hour) 
            # - move maximum wind gust within the hour from half past the hour to end of hour
            if 'time' in [coord.name() for coord in cube.coords()]:
                time_coord = cube.coord('time')
                datetimes = time_coord.units.num2date(time_coord.points)
                new_datetimes = []

                def round_time_to_nearest_five_minutes(dt):
                    discard = timedelta(minutes=dt.minute % 5,
                                        seconds=dt.second,
                                        microseconds=dt.microsecond)
                    dt -= discard
                    if discard >= timedelta(minutes=2.5):
                        dt += timedelta(minutes=5)
                    return dt

                for t in datetimes:
                    if cube.attributes.get('STASH') == 'm01s03i463':
                        t = t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                    else:
                        t = round_time_to_nearest_five_minutes(t)
                    new_datetimes.append(t)

                time_coord.points = time_coord.units.date2num(new_datetimes) 
            
                if cube.attributes.get('STASH') == 'm01s03i463':
                    cube.cell_methods = ()
            
            # only include cubes that don't have a cell method
            if not cube.cell_methods:
                cleaned.append(cube)        
            
        # if time coordinate is present, concatenate times and constrain them  
        # put the cubelists into the stream dictionary            
        first_cube = cleaned[0] # note: assumes time coordinate is the same for all cubes in the cubelist
        time_coord = first_cube.coord('time') if first_cube.coords('time') else None
        is_dim_coord = time_coord and time_coord in first_cube.dim_coords      
        try:
            if is_dim_coord: 
                for i, cube in enumerate(cleaned):
                    def harmonize_coord_metadata(cube, coord_name):
                        """Remove trailing '_<number>' from var_name on the coordinate if present."""
                        coord = cube.coord(coord_name)
                        if coord.var_name and re.search(r'_\d+$', coord.var_name):
                            coord.var_name = re.sub(r'_\d+$', '', coord.var_name)
                        return True
                    harmonize_coord_metadata(cube, "time")
                    harmonize_coord_metadata(cube, "forecast_reference_time")
                    harmonize_coord_metadata(cube, "forecast_period")          
                merged_and_concatenated = cleaned.concatenate()    
                sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")
                start_time = sim_dt + timedelta(hours=filter_start_hour) 
                end_time = sim_dt + timedelta(hours=filter_end_hour)
                time_constraint = iris.Constraint(time=lambda cell: start_time <= cell.point <= end_time) 
                merged_and_concatenated = merged_and_concatenated.extract(time_constraint)
                merged_by_stream[stream] = merged_and_concatenated
                print(f"Merged and concatenated {len(merged_cubelist)} cubes by member and time into {len(merged_and_concatenated)} cube(s) for stream '{stream}'")
            else:
                merged_by_stream[stream] = cleaned
                print(f"Merged {len(merged_cubelist)} cubes by member and time into {len(merged)} cube(s) for stream '{stream}'")
        except Exception as e:
            print(f"Failed to merge/concatenate cubelist for stream '{stream}': {e}")
            merged_by_stream[stream] = merged_cubelist  # Return raw list for manual inspection

    return merged_by_stream
