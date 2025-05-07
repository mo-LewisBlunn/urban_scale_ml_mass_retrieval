import tempfile
import subprocess
import os
import time
from datetime import datetime, timedelta
import iris
from iris.analysis import Linear
from iris.coords import DimCoord, AuxCoord
import numpy as np

### Variables ###

# Settings
MASS_retrieval = False # when set to False, if the files are all there, the retrieval will be skipped
process_retrieval = True # when set to False, the processing won't be performed

# Fixed variables
suite = 'u-dg726'
temp_out_dir = '/data/scratch/lewis.blunn/ml_mass_retrieval/temp_out_dir/'
log_dir = '/data/scratch/lewis.blunn/ml_mass_retrieval/log_dir/'
orography_ancil_mass_path = 'moose:/adhoc/users/lewis.blunn/ancils/u-da774_Paris_0p3km_L70/orog_srtm/qrparm.orog.srtm.121x1'
filter_start_hour = 4 # this plus simulation time will be the output data start time
filter_end_hour = 38 # this plus simulation time will be the output data end time
target_grid_spacing = 0.0027 # the grid spacing of the high-res data in the fixed region
target_heights = [10.,100.,500.,1000., 1500.,2000.,3000.,4000.,6000.,8000.,10000.] # the heights at which to output on vertical levels, interpolation will be performed if the level doesn't exist

# Looping variables
models = ['pmv']
simulations = ['20240716T1800Z']
streams = ['pa', 'pd']
members = ['00', '18'] # ['00', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
cruns = {
    'pa': ['000', '012', '024'],
    'pd': [f'{i:03}' for i in range(36)]
} #cruns = {'pa':['000'], 'pd':['000']}
land_covers = {
    'CCIv2': ['20240716T1800Z', '20240811T1800Z'],
    'WC': ['20240812T1800Z', '20240909T1800Z']
}
stash_codes_strs = {
    'pa': '3236, 3225, 3226, 3463, 3247, 3245, 4203, 30405, 30406, 24, 3234, 3217, 1235, 1202, 2207, 2201, 8223',
    'pd': '16004, 10, 2, 3, 150, 408'
}

# Convert land cover date ranges to datetime
land_covers_dt = {
    lc: [datetime.strptime(d, "%Y%m%dT%H%MZ") for d in dates]
    for lc, dates in land_covers.items()
}



### Functions ###

def get_land_cover(sim_dt):
    """
    Helper function to determine land cover based on sim date.
    """
    for cover, (start, end) in land_covers_dt.items():
        if start <= sim_dt <= end:
            return cover
    return None



def get_mass_file_name(sim, mod, lc, mem, stream, crun):
    """
    Build the expected mass_file name.
    """
    return f'{sim}_Paris_{mod}_{lc}_RAL3p2_DSMURK_SMFIX_MORUSES_em{mem}_{stream}{crun}.cutout.pp'



def mass_retrieval_job(stash_codes_str, suite, temp_out_dir, log_dir, sim, mass_file):
    """
    Submit a SLURM job to retrieve a mass file from MASS.
    """
    output_path = os.path.join(temp_out_dir, sim, mass_file)
    log_path = os.path.join(log_dir, sim, f"{mass_file}.log")

    # Remove the temporary output file if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Create a temporary file for the SLURM shell script
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as script_file:
        script_contents = f"""#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH --mem-per-cpu=3000
#SBATCH --ntasks=1
#SBATCH --output={log_path}
#SBATCH --time=15

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
    while True:
        try:
            result = subprocess.run(['squeue', '-u', os.getlogin()], capture_output=True, text=True, check=True)
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



def retrieve_sim(sim, mod):
    """
    Retrieve all mass files for a given simulation and model.
    """
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
                lc = get_land_cover(sim_dt)
                if lc is None:
                    print(f"No land cover found for sim {sim}")
                    continue

                # Set the mass_file name
                mass_file = get_mass_file_name(sim, mod, lc, mem, stream, crun)

                # Retrieve mass_file from MASS
                job_id = mass_retrieval_job(stash_codes_str, suite, temp_out_dir, log_dir, sim, mass_file)
                if job_id:
                    job_ids.append(job_id)

    # Wait for all jobs to complete
    wait_for_jobs_to_finish(job_ids)



def validate_and_rerun_failed_jobs_for_sim(sim, models, max_retries=3, wait_time=10*60):
    """
    Validate log files and re-run failed jobs for a specific simulation.
    Retry up to `max_retries` times with `wait_time` seconds between attempts.
    Save a summary of failures for that simulation.
    Return True if successful, False if still failed after retries.
    """
    retry_count = 0
    sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")
    summary_failures = []

    while retry_count <= max_retries:
        failed_jobs = []

        for mod in models:
            for stream in streams:
                stash_codes_str = stash_codes_strs[stream]
                for mem in members:
                    for crun in cruns[stream]:
                        lc = get_land_cover(sim_dt)
                        if lc is None:
                            continue

                        mass_file = get_mass_file_name(sim, mod, lc, mem, stream, crun)
                        log_path = os.path.join(log_dir, sim, f"{mass_file}.log")

                        if not os.path.exists(log_path):
                            failed_jobs.append((mod, stream, stash_codes_str, mem, crun, mass_file))
                            continue

                        with open(log_path, 'r') as f:
                            lines = f.readlines()
                            if not any(f"finished {mass_file}" in line for line in lines[-10:]):
                                failed_jobs.append((mod, stream, stash_codes_str, mem, crun, mass_file))

        if not failed_jobs:
            print(f"\nLog files checked -- simulation {sim} completed successfully.")
            return True

        if retry_count == max_retries:
            summary_failures = failed_jobs
            break

        print(f"\nRetrying {len(failed_jobs)} failed job(s) for {sim}... (attempt {retry_count+1}/{max_retries})\n")
        job_ids = []
        for mod, stream, stash_codes_str, mem, crun, mass_file in failed_jobs:
            job_id = mass_retrieval_job(stash_codes_str, suite, temp_out_dir, log_dir, sim, mass_file)
            if job_id:
                job_ids.append(job_id)
        wait_for_jobs_to_finish(job_ids)

        retry_count += 1
        print(f"Waiting {wait_time} seconds before next retry...")
        time.sleep(wait_time)

    # Write simulation-level summary
    if summary_failures:
        summary_file = os.path.join(log_dir, sim, 'summary_failed_jobs.log')
        with open(summary_file, 'w') as f:
            for mod, stream, _, mem, crun, mass_file in summary_failures:
                f.write(f"{sim},{mod},{stream},{mem},{crun},{mass_file}\n")
        print(f"Simulation {sim} failed. Summary written to {summary_file}")
        return False



def load_cubelists_by_stream(sim, mod):
    """
    Load .pp files retrieved for a simulation into a dictionary of cubelists, grouped by stream.
    Adds a scalar coordinate 'member' to each cube.
    Returns a dictionary: {stream: [CubeList, CubeList, ...]}
    """
    sim_temp_out_dir = os.path.join(temp_out_dir, sim)
    sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")
    cubelists_by_stream = {stream: [] for stream in streams}

    # Define member mapping: '00' => 1, '18' => 2, ..., '34' => 18
    #member_mapping = {f"{int(i):02d}": idx + 1 for idx, i in enumerate(range(0, 35)) if i == 0 or 18 <= i <= 34}
    member_mapping = {f"{int(i):02d}": 1 if i == 0 else i - 16 for i in range(0, 35) if i == 0 or 18 <= i <= 34}

    for stream in streams:
        for mem in members:
            for crun in cruns[stream]:
                lc = get_land_cover(sim_dt)
                if lc is None:
                    continue
                mass_file = get_mass_file_name(sim, mod, lc, mem, stream, crun)
                file_path = os.path.join(sim_temp_out_dir, mass_file)

                if os.path.exists(file_path):
                    try:
                        cubes = iris.load(file_path)

                        # Add scalar coord 'member' to each cube
                        member_index = member_mapping.get(mem)
                        if member_index is None:
                            print(f"  Warning: member '{mem}' not found in mapping.")
                            member_index = -1  # fallback or skip if preferred

                        for cube in cubes:
                            coord = iris.coords.DimCoord(member_index,
                                                         standard_name=None,
                                                         long_name='member',
                                                         units='1')
                            cube.add_aux_coord(coord)

                        cubelists_by_stream[stream].append(cubes)
                        print(f"Loaded {len(cubes)} cube(s) from {file_path} with member index {member_index}")
                    except Exception as e:
                        print(f"Failed to load {file_path}: {e}")
                else:
                    print(f"File not found (skipped): {file_path}")

    return cubelists_by_stream
    


def merge_cubelists_by_time(cubelists_by_stream):
    """
    Concatenate or merge cubelists per stream based on 'time' coordinate type.
    Handles scalar 'member' coordinate appropriately.
    Returns a dictionary: {stream: CubeList}
    """
    merged_by_stream = {}

    for stream, cubelists in cubelists_by_stream.items():
        merged_cubelist = iris.cube.CubeList()

        # Combine all cubes into one big CubeList
        for cubelist in cubelists:
            merged_cubelist.extend(cubelist)

        if not merged_cubelist:
            print(f"\nNo cubes found for stream '{stream}'.")
            merged_by_stream[stream] = iris.cube.CubeList()
            continue

        first_cube = merged_cubelist[0]
        time_coord = first_cube.coord('time') if first_cube.coords('time') else None
        is_dim_coord = time_coord and time_coord in first_cube.dim_coords

        try:
            if is_dim_coord:
                # Group cubes by scalar 'member' value
                cubes_by_member = {}
                for cube in merged_cubelist:
                    try:
                        member_val = cube.coord('member').points[0]
                    except:
                        member_val = -1  # Fallback or missing member
                    cubes_by_member.setdefault(member_val, iris.cube.CubeList()).append(cube)

                # Concatenate cubes within each member group
                concatenated_by_member = []
                for member, cube_group in cubes_by_member.items():
                    concatenated = cube_group.concatenate()
                    concatenated_by_member.extend(concatenated)

                # Final merge over scalar coords (if needed)
                final_merged = iris.cube.CubeList(concatenated_by_member).merge()
                merged_by_stream[stream] = final_merged

                print(f"\nConcatenated {len(merged_cubelist)} cubes by member into {len(final_merged)} cube(s) for stream '{stream}' (time is a dimension coordinate)")

            else:
                # Merge if time is scalar or absent
                merged = merged_cubelist.merge()
                merged_by_stream[stream] = merged
                print(f"\nMerged {len(merged_cubelist)} cubes into {len(merged)} cube(s) for stream '{stream}' (time is a scalar coordinate)")

        except Exception as e:
            print(f"\nFailed to merge/concatenate cubelist for stream '{stream}': {e}")
            merged_by_stream[stream] = merged_cubelist  # Return raw list for manual inspection

    # Print the merged result
    for stream, merged_cubelist in merged_by_stream.items():
        print(f"\nMerged cubelist for stream '{stream}' has {len(merged_cubelist)} cube(s):")
        for i, cube in enumerate(merged_cubelist):
            time_coord = cube.coord('time') if cube.coords('time') else None
            time_summary = f", time points: {len(time_coord.points)}" if time_coord else ""
            member_val = cube.coord('member').points[0] if cube.coords('member') else "?"
            print(f"  Cube {i+1}: {cube.name()} [member {member_val}]{time_summary}")

    return merged_by_stream



def standardise_cubelist_times(merged_cubelist_by_stream, sim, filter_start_hour, filter_end_hour):
    """
    Round times to nearest 5 minutes, shift specific STASH codes, and filter cubes
    using an iris Constraint for the desired time window. Modifies in place.
    """

    def round_time_to_nearest_five_minutes(dt):
        discard = timedelta(minutes=dt.minute % 5,
                            seconds=dt.second,
                            microseconds=dt.microsecond)
        dt -= discard
        if discard >= timedelta(minutes=2.5):
            dt += timedelta(minutes=5)
        return dt

    print("\nStandardising times...")

    sim_dt = datetime.strptime(sim, "%Y%m%dT%H%MZ")
    start_time = sim_dt + timedelta(hours=filter_start_hour)
    end_time = sim_dt + timedelta(hours=filter_end_hour)

    for stream, cubelist in merged_cubelist_by_stream.items():
        updated_cubelist = iris.cube.CubeList()

        for cube in cubelist:
            if 'time' not in [coord.name() for coord in cube.coords()]:
                updated_cubelist.append(cube)
                continue

            time_coord = cube.coord('time')
            datetimes = time_coord.units.num2date(time_coord.points)
            new_datetimes = []

            for t in datetimes:
                if cube.attributes.get('STASH') == 'm01s03i463':
                    t = t.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    t = round_time_to_nearest_five_minutes(t)
                new_datetimes.append(t)

            time_coord.points = time_coord.units.date2num(new_datetimes)

            updated_cubelist.append(cube)

        # Apply time filtering using iris.Constraint
        time_constraint = iris.Constraint(
            time=lambda cell: start_time <= cell.point <= end_time
        )

        filtered_cubelist = iris.cube.CubeList()
        for cube in updated_cubelist:
            try:
                constrained = cube.extract(time_constraint)
                if constrained is not None:
                    filtered_cubelist.append(constrained)
                else:
                    print(f"  All time points excluded after constraint for cube '{cube.name()}' in stream '{stream}'")
            except Exception as e:
                print(f"  Error applying constraint to cube '{cube.name()}': {e}")
                filtered_cubelist.append(cube)

        # Replace original cubelist with filtered and standardised version
        merged_cubelist_by_stream[stream] = filtered_cubelist

        print(f"Stream '{stream}' standardised and filtered: {len(filtered_cubelist)} cube(s) retained.")
        
    return merged_cubelist_by_stream



def remove_duplicate_name_cubes_with_cell_methods(cubelist_by_stream):
    """
    For each stream, remove cubes that have cell methods if another cube with the same name exists without cell methods.
    Updates cubelist_by_stream in place.
    """
    for stream, cubelist in cubelist_by_stream.items():
        print(f"\nProcessing stream '{stream}' for duplicate cube name filtering...")
        name_groups = {}

        # Group cubes by name
        for cube in cubelist:
            name_groups.setdefault(cube.name(), []).append(cube)

        # Rebuild CubeList with filtering
        new_cubelist = iris.cube.CubeList()
        for name, cubes in name_groups.items():
            if len(cubes) == 1:
                new_cubelist.append(cubes[0])
                continue

            # Separate cubes with and without cell methods
            cubes_with_cell_methods = [cube for cube in cubes if cube.cell_methods]
            cubes_without_cell_methods = [cube for cube in cubes if not cube.cell_methods]

            if cubes_without_cell_methods:
                print(f"  Keeping {len(cubes_without_cell_methods)} cube(s) without cell methods for '{name}', discarding {len(cubes_with_cell_methods)}.")
                new_cubelist.extend(cubes_without_cell_methods)
            else:
                print(f"  All cubes for '{name}' have cell methods — keeping all {len(cubes)} cube(s).")
                new_cubelist.extend(cubes)

        cubelist_by_stream[stream] = new_cubelist

    return cubelist_by_stream



def orog_ancil_retrieval(orography_ancil_mass_path, output_dir):
    """
    Retrieves the orography ancillary file from MOOSE and loads it as an Iris cube or CubeList.
    Returns the loaded cube(s).
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract the filename from the MOOSE path
    orog_filename = os.path.basename(orography_ancil_mass_path)
    local_orog_path = os.path.join(output_dir, orog_filename)

    if not os.path.exists(local_orog_path):
        # Build and run the command to retrieve the file
        cmd = ['moo', 'get', orography_ancil_mass_path, local_orog_path]
        print(f"\nRetrieving orography ancillary file:\n{' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully retrieved orography file to: {local_orog_path}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to retrieve orography ancillary file: {e}")
            raise

    else:
        print(f"\nOrography file already exists locally at {local_orog_path}")

    # Load the orography file using Iris
    try:
        cube = iris.load_cube(local_orog_path)
        print(f"Loaded orography ancillary cube.")
    except Exception as e:
        print(f"Failed to load orography file: {e}")
        raise

    return cube



def make_target_grid(cube, target_grid_spacing, tol=1e-8):
    """
    Make the target_grid from the high-res orography cube. The 
    target_grid is based on the fixed resolution portion.
    Note: the target grid will remain rotated if the high-res orography 
    grid is. This reduces the amount of the high-res data regridding and 
    means that the high-res data won't have to be interpolated (unless
    it is staggered differently e.g., winds). The low-res data will be 
    regridded (and therefore interpolated) to this high-res target grid.   
    """

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

    return cropped_cube

   
        
### Main MASS retrieval and processing loop ###

if __name__ == "__main__":
    overall_results = {}

    for mod in models:
    
        # Make the the target grid based on the orography ancillary
        orog_cube = orog_ancil_retrieval(orography_ancil_mass_path, temp_out_dir)
        target_grid = make_target_grid(orog_cube, target_grid_spacing, tol=1e-6)        

        for sim in simulations:          
        
            # The retrieval
            if MASS_retrieval:
                retrieve_sim(sim, mod)
            else:
                print("\nSkipped MASS_retrieval.")
            success = validate_and_rerun_failed_jobs_for_sim(sim, [mod])
            overall_results[sim] = 'PASSED' if success else 'FAILED'

            # Process the retrievals
            if success and process_retrieval:
                # Load all the retrieved files for one mod and sim
                cubelists_by_stream = load_cubelists_by_stream(sim, mod)
                        
                # Merge cubelists into one per stream (based on time and ensemble member)
                merged_cubelist_by_stream = merge_cubelists_by_time(cubelists_by_stream)
 
                # Standardise the times
                merged_cubelist_by_stream = standardise_cubelist_times(merged_cubelist_by_stream, sim, filter_start_hour, filter_end_hour) 
                
                # Remove cubes with a cell method (time processing) if there is another cube with the same name
                cubelist_by_stream = remove_duplicate_name_cubes_with_cell_methods(merged_cubelist_by_stream)
                print(cubelist_by_stream)
              
                # Temporary save of data
              
                """
                # Vertical regrid
                cubelist_by_stream, v_regrid_files = vertically_interpolate_to_target_heights(
                    cubelist_by_stream, 
                    target_heights, 
                    temp_out_dir=os.path.join(temp_out_dir, "regrid_temp"),
                    log_dir=os.path.join(temp_out_dir, "regrid_logs")
                )

                # Wait until regrid jobs are done
                wait_for_results(v_regrid_files)

                # Reload regridded cubes into the cubelist_by_stream
                cubelist_by_stream = reload_regrid_cubes(cubelist_by_stream, interpolated_files)
                print(cubelist_by_stream)
                print(cubelist_by_stream['pa'])
                print(cubelist_by_stream['pa'][0].coord('time'))
                print(cubelist_by_stream['pd'])
                print(cubelist_by_stream['pd'][0])
                print(cubelist_by_stream['pd'][0].coord('time'))
                print(cubelist_by_stream['pd'][0].coord('model_level_number'))
                print(cubelist_by_stream['pd'][0].coord('level_height'))
                """

                # Find if they contain the same grids as target_grid
                # If yes, constrain the cube to have the same grid as target_grid
                # If no, horizontal regrid
                
                # Merge streams and save
                
                # Remove files in temp_out_dir
                                    
            else:
                print(f"\nSkipping data processing for sim {sim} due to retrieval failure.")
       
    # Write final MASS_retrieval summary log
    if MASS_retrieval:
        final_summary_path = os.path.join(log_dir, 'final_summary.log')
        with open(final_summary_path, 'w') as f:
            for sim, status in overall_results.items():
                f.write(f"{sim}: {status}\n")

        print(f"\nFinal Summary:\n--------------")
        for sim, status in overall_results.items():
            print(f"{sim}: {status}")
        print(f"\nFinal summary written to: {final_summary_path}")

              
