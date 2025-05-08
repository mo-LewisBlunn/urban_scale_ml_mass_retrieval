import iris
import numpy as np
import tempfile
import subprocess
import os
import time
import re
from config import Config
from datetime import datetime, timedelta
from iris.analysis import Linear
from iris.coords import DimCoord, AuxCoord
from iris.cube import CubeList
from iris.util import unify_time_units, equalise_attributes
from mass_retrieval_functions import *

"""
Lewis Blunn
Usage:
- choose settings in the config.yaml
- module load scitools
- python workflow.py
"""

      
### Main MASS retrieval and processing loop ###

if __name__ == "__main__":

    print("\nStarted mass_retrieval.py ...")
    start_time = time.time()

    config = Config()
    print("\nConfig loaded.")

    ### Ancillaries and target grid ###

    # Extract all ancillaries from MASS, looping through suite and res, outputting to ancils_{suite}_{res} directory within temp_out_dir
    print("\nRetrieving ancils ...")
    ancil_retrieval(config.ancil_mass_path, config.ancil_suites, config.ancil_models, config.temp_out_dir)

    # Generate target grid once based on the orography ancillary
    print("\nCreating target grid ...")
    os.makedirs(os.path.dirname(config.grid_dir), exist_ok=True)
    target_grid = make_target_grid(config.temp_out_dir, config.target_grid_spacing, config.grid_dir,\
                                    suite=config.ancil_suites[0], model=config.ancil_models[0], tol=1e-6)
    print("Target grid ready.")

    # Read in all the ancillaries, choose those required, process, and save the ancillaries
    print("\nProcessing ancils ...")
    process_ancils(config.grid_dir, config.temp_out_dir, config.log_dir, config.ancil_suites, config.ancil_models, \
                   config.ancil_stash_codes, config.ancil_lc_names, config.ancil_lai_names)

    ### Forecasts ###

    # Clean old marker files
    print("\nCleaning old marker files...")
    os.makedirs(os.path.dirname(config.markers_dir), exist_ok=True)
    for fname in os.listdir(config.markers_dir):
        if fname.endswith("_save_started.txt"):
            os.remove(os.path.join(config.markers_dir, fname))
    print("Old marker files removed.")

    # Loop over the batch jobs
    for mod in config.models:
        for sim in config.simulations:
            job_name = f"job_{mod}_{sim}"
            print(f"\njob name: {job_name}")
            log_file = os.path.join(config.log_dir, sim, f"{job_name}.log")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Submit the MASS retrieval and processing batch job for one model and simulation
            script_contents = f"""#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output={log_file}
#SBATCH --ntasks=1
#SBATCH --time=300
#SBATCH --mem=8G

module load scitools
python run_sim_job.py {sim} {mod}
"""
            script_path = os.path.join(config.temp_out_dir, f"run_{mod}_{sim}.sh")
            with open(script_path, "w") as f:
                f.write(script_contents)
            os.chmod(script_path, 0o755)
            subprocess.run(["sbatch", script_path])
            os.remove(script_path)

            # Wait for marker file from previous job. 
            # The marker is that {prev_sim}_{prev_mod}_save_started.txt has been produced by run_sim_job.py
            # Won't be entered during the first loop (when submitted_jobs is empty)
            marker_path = os.path.join(config.markers_dir, f"{mod}_{sim}_save_started.txt")
            print(f"Must wait for the marker before moving to the next job. marker_path: {marker_path} ...")
            waiting_start_time = time.time()
            loop_counter = 0
            while not os.path.exists(marker_path):
                loop_counter += 1
                if loop_counter % 5 == 0:
                    current_time = time.time()
                    print(f"Time waiting so far: {(current_time - waiting_start_time) / 60:.2f} minutes.")
                time.sleep(30)
    
    end_time = time.time()
    print(f"\nAll jobs submitted. Total time: {(end_time - start_time) / 60:.2f} minutes.")
              
