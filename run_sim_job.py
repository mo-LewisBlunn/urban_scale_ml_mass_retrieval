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
from mass_retrieval import (
    retrieve_sim, 
    validate_and_rerun_failed_jobs_for_sim, 
    process_retrieval, 
    load_processed_cubelists, 
    clean_and_join, 
    get_land_cover
)


        
### Main MASS retrieval and processing loop ###

if __name__ == "__main__":
    sim = sys.argv[1]
    mod = sys.argv[2]

    config = Config()
    print(f"\nRunning job for sim: {sim}, mod: {mod}") 
        
    start_time = time.time()    

    # The retrieval
    if config.MASS_retrieval_switch:
        retrieve_sim(config.log_dir, config.temp_out_dir, config.streams, config.members, sim, mod,
                    config.stash_codes_strs, config.cruns, config.suite, config.land_covers_dt)
    else:
        print("\nSkipped MASS_retrieval.")
    
    # Check the log files to see if each retrieval job completed successfully. If not, try again.
    success = validate_and_rerun_failed_jobs_for_sim(sim, mod, config.temp_out_dir, 
                                    config.log_dir, config.stash_codes_strs, config.cruns, config.streams, 
                                    config.members, config.suite, config.land_covers_dt)

    # Process the retrievals: vertical and horizontal interpolation to target_heights and target_grid
    if success and config.process_retrieval_switch:
        # Load all the retrieved files for one mod and sim, and process
        process_retrieval(config.temp_out_dir, config.grid_dir, config.log_dir, sim, mod, config.streams, config.members, 
        		  config.cruns, config.land_covers_dt, config.target_heights)  
    elif success and config.process_retrieval_switch==False:         
        print(f"\nSkipping data processing for mod: {mod}, sim: {sim} due to process_retrieval_switch=False.")  
    else:
        print(f"\nBreaking at data processing due to retrieval failure.")
        sys.exit(1)

    # Load the processed version back in
    cubelists_by_stream = load_processed_cubelists(config.temp_out_dir, config.log_dir, sim, mod, config.streams, 
                                                   config.members, config.cruns, config.land_covers_dt)
                          
    # Merge cubelists into one per stream (based on time and ensemble member)
    cubelists_by_stream = clean_and_join(cubelists_by_stream, sim, config.filter_start_hour, config.filter_end_hour)
    
    # Put all streams in one cubelist
    sim_cubes = iris.cube.CubeList()
    for stream_cubes in cubelists_by_stream.values():
        sim_cubes.extend(stream_cubes)
        
    # Save the simulation (and create a marker to indicate save has started
    os.makedirs(config.temp_out_dir, exist_ok=True)
    marker_path = os.path.join(config.markers_dir, f"{mod}_{sim}_save_started.txt")
    with open(marker_path, 'w') as f:
        f.write(f"iris.save started at {datetime.now()}\n")
    print(f"\nSaving CubeList for mod: {mod}, sim: {sim} ...")
    iris.save(sim_cubes, config.temp_out_dir+f'{mod}_{sim}.nc', zlib=True)
    print("Saved!")
    
    # Print the saved CubeList
    cubes = iris.load(config.temp_out_dir+f'{mod}_{sim}.nc')
    print("\nCubeList after loading back in:\n",cubes)
    
    # Report timing
    end_time = time.time()
    duration = (end_time - start_time)/60.
    print(f"\nCompleted mod: {mod}, sim: {sim} in {duration:.2f} mins.")

              
