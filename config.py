import yaml
from datetime import datetime, timedelta

class Config:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)

        self.MASS_retrieval_switch = raw['settings']['MASS_retrieval_switch']
        self.process_retrieval_switch = raw['settings']['process_retrieval_switch']

        fvars = raw['fixed_variables']
        self.suite = fvars['suite']
        self.temp_out_dir = fvars['temp_out_dir']
        self.grid_dir = fvars['grid_dir']
        self.markers_dir = fvars['markers_dir']
        self.log_dir = fvars['log_dir']
        self.ancil_mass_path = fvars['ancil_mass_path']
        self.filter_start_hour = fvars['filter_start_hour']
        self.filter_end_hour = fvars['filter_end_hour']
        self.target_grid_spacing = fvars['target_grid_spacing']
        self.target_heights = fvars['target_heights']

        lvars = raw['looping_variables']
        self.models = lvars['models']
        def parse_simulations(sim_input):
            if isinstance(sim_input, list):
                return sim_input
            elif isinstance(sim_input, str):
                ranges = sim_input.strip().split(',')
                dates = []
                for r in ranges:
                    parts = r.strip().split()
                    if len(parts) != 2:
                        raise ValueError(f"Each range must have exactly two dates (start and end), but got: '{r}'")
                    start = datetime.strptime(parts[0], "%Y%m%dT%H%MZ")
                    end = datetime.strptime(parts[1], "%Y%m%dT%H%MZ")
                    current = start
                    while current <= end:
                        dates.append(current.strftime("%Y%m%dT%H%MZ"))
                        current += timedelta(days=1)
                return dates
            else:
                raise TypeError(f"Invalid type for simulations: {type(sim_input)}. Expected list or string.")
        self.simulations = parse_simulations(lvars['simulations'])
        self.streams = lvars['streams']
        self.members = lvars['members']
        pd_raw = lvars['cruns']['pd']
        if isinstance(pd_raw, list):
            self.cruns = {
                'pa': lvars['cruns']['pa'],
                'pd': pd_raw
            }
        elif isinstance(pd_raw, str) and "range(" in pd_raw:
            num = int(pd_raw.strip().split("range(")[1].split(")")[0])
            self.cruns = {
	        'pa': lvars['cruns']['pa'],
	        'pd': [f"{i:03}" for i in range(num)]
            }
        else:
            self.cruns = {
                'pa': lvars['cruns']['pa'],
                'pd': lvars['cruns']['pd']  # fallback to whatever's given
            }
        self.stash_codes_strs = lvars['stash_codes_strs']
        self.land_covers = lvars['land_covers']
        self.land_covers_dt = {
            lc: [datetime.strptime(d, "%Y%m%dT%H%MZ") for d in dates]
            for lc, dates in self.land_covers.items() # Convert land cover date ranges to datetime
        }
        self.ancil_suites = lvars['ancil_suites']
        self.ancil_models = lvars['ancil_models']
        self.ancil_stash_codes = lvars['ancil_stash_codes']
        self.ancil_lc_names = lvars['ancil_lc_names']
        self.ancil_lai_names = lvars['ancil_lai_names']
