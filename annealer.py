import math
import random
import logging
import shutil
import os 
import re
import subprocess
import matplotlib.pyplot as plt
import json

logging.basicConfig(level=logging.INFO)

PIPELINE_WIDTH = list(range(1, 11)) # not rhs inclusive, so 1...10
WINDOW_SIZES = {
    'rob': [16, 32, 64, 128, 256, 512, 1024],
    'iq': [11, 21, 43, 85, 171, 341, 682],
    'lsq': [5, 11, 21, 43, 85, 171, 341]
}
BRANCH_PRED_SIZES = {
    'local': [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    'global': [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    'btb': [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
    'ras': [16, 32, 64, 128, 256, 512]
}
L1_DATA_SIZE = [2**i for i in range(2, 10) if 2**i <= 512]
L1_INST_SIZE = [2**i for i in range(2, 10) if 2**i <= 512]
L2_SIZE = [2**i for i in range(2, 6) if 2**i <= 32]

class ProcessorSimulatedAnnealer():
    """
    Performs simulated annealing (a type of optimizer) to find the best combination of parameters to reduce
    energy consumption of a OoO CPU using Gem5. Sure, I could've written this in C++ for coolness and performance gains,
    but the massive bottleneck here is the 30-45 second runtime for each Gem5 test, so the performance of the annealear
    really has no effect on the overall runtime of this optimizer. So, Python is fine!
    """
    def __init__(self):
        self.NUM_SEARCH = 3
        self.MAX_ITER = 250
        self.STOPPING_TEMP = 1e-3
        self.initial_temp = 100
        self.COOLING_RATE = 0.95
        self.DIR_NAME = "annealer_iter"
        self.RESULTS_NAME = "annealer_results.json"
        self.COST_PENALTY = 1000
        self.cost_table = None

    def _run_gem5(self, params: dict):
        """
        Runs Gem5 with the given params
        """
        cmd = f"python /homes/lp721/aca-gem5/simulate.py --window-size {params["rob"]},{params["iq"]},{params["lsq"]} --branch-pred-size {params["local"]},{params["global"]},{params["btb"]},{params["ras"]} --l1-data-size {params["l1_data_size"]} --l1-inst-size {params["l1_inst_size"]} --l2-size {params["l2_size"]} --name {self.DIR_NAME}"

        logging.info(f'Running gem5 with command {cmd}, and params {params}')
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _get_results(self) -> dict:
        """
        Returns a dict of the results, if this fails, then we will throw a RuntimeError and halt
        """
        results_file = os.path.join(self.DIR_NAME, 'results')
        
        if os.path.isfile(results_file):
            with open(results_file, 'r') as file:
                data = file.read()
                
                sim_sec = re.search(r'Simulated seconds\s*=\s*([\d.]+)', data)
                cpi = re.search(r'CPI\s*=\s*([\d.]+)', data)
                area = re.search(r'Area\s*=\s*([\d.]+)\s*mm\^2', data)
                peak_dyn = re.search(r'Peak Dynamic\s*=\s*([\d.]+)\s*W', data)
                power_sub = re.search(r'Subthreshold Leakage\s*=\s*([\d.]+)\s*W', data)
                power_gate = re.search(r'Gate Leakage\s*=\s*([\d.]+)\s*W', data)
                power_run = re.search(r'Runtime Dynamic\s*=\s*([\d.]+)\s*W', data)

                power_sum = float(power_sub.group(1)) + float(power_gate.group(1)) + float(power_run.group(1))
                energy = power_sum * float(sim_sec.group(1))

                result = {
                    'sim_sec': float(sim_sec.group(1)),
                    'cpi': float(cpi.group(1)),
                    'area': float(area.group(1)),
                    'peak_dyn': float(peak_dyn.group(1)),
                    'power_sub': float(power_sub.group(1)),
                    'power_gate': float(power_gate.group(1)),
                    'power_run': float(power_run.group(1)),
                    'energy': energy
                }

        return result

    def _delete_test(self):
        """
        Deletes the test directory
        """
        shutil.rmtree(self.DIR_NAME)
        logging.debug(f'Deleted test directory')

    def _cost_fn(self, params: dict) -> float:
        """
        Returns the cost of the current parameters chosen according to gem5
        """
        self._run_gem5(params)
        results = self._get_results()
        self._delete_test()
        energy = results["energy"]
        logging.info(f'Calculated energy consumption of {energy} with {params}')
        return energy * self.COST_PENALTY

    def _get_neighbour(self, params: dict) -> dict:
        """
        Generates a neighbour by tweaking the current parameters slightly.
        Assumes params is a dictionary of named parameters with specified ranges.
        """
        new_params = params.copy()
        
        logging.debug(f'Get neighbour called')

        def pick_neighbour_value(value, values):
            idx = values.index(value)
            if 0 < idx < len(values) - 1:
                neighbour_idx = idx + random.choice([-1, 1])
            elif idx == 0:
                neighbour_idx = idx + 1
            else:
                neighbour_idx = idx - 1
            
            logging.debug(f'Value change: {value}->{values[neighbour_idx]}')
            return values[neighbour_idx]
        
        if random.random() < 0.4:
            new_params['pipeline_width'] = pick_neighbour_value(
                params['pipeline_width'], PIPELINE_WIDTH
            )
        
        for key in ['rob', 'iq', 'lsq']:
            if random.random() < 0.3:
                new_params[key] = pick_neighbour_value(params[key], WINDOW_SIZES[key])

        for key in ['local', 'global', 'btb', 'ras']:
            if random.random() < 0.3:
                new_params[key] = pick_neighbour_value(params[key], BRANCH_PRED_SIZES[key])

        if random.random() < 0.3:
            new_params['l1_data_size'] = pick_neighbour_value(params['l1_data_size'], L1_DATA_SIZE)

        if random.random() < 0.3:
            new_params['l1_inst_size'] = pick_neighbour_value(params['l1_inst_size'], L1_INST_SIZE)

        if random.random() < 0.3:
            new_params['l2_size'] = pick_neighbour_value(params['l2_size'], L2_SIZE)

        return new_params

    def _acceptance_probability(self, current_cost: float, candidate_cost: float, temperature: float) -> float:
        """
        Determines the acceptance probability of the new solution.
        """
        if candidate_cost < current_cost:
            return 1.0
        else:
            return math.exp((current_cost - candidate_cost) / temperature)

    def _save_results(self, results: tuple[float, dict]):
        """
        Saves the results tuple (best_cost, best_params) to the file at path self.RESUTLS_NAME
        """
        with open(self.RESULTS_NAME, 'w') as file:
            json.dump({"best_cost": results[0], "best_params": results[1]}, file)

    def _search(self, search_params=None) -> tuple[float, dict, float]:
        """
        Performs simulated annealing, returning a tuple of the best result (lowest cost), and the parameters to achieve this.
        """
        # Start with initial parameters randomly chosen from allowed values on first iter
        if search_params is None:
            current_params = {
                'pipeline_width': random.choice(PIPELINE_WIDTH),
                'rob': random.choice(WINDOW_SIZES['rob']),
                'iq': random.choice(WINDOW_SIZES['iq']),
                'lsq': random.choice(WINDOW_SIZES['lsq']),
                'local': random.choice(BRANCH_PRED_SIZES['local']),
                'global': random.choice(BRANCH_PRED_SIZES['global']),
                'btb': random.choice(BRANCH_PRED_SIZES['btb']),
                'ras': random.choice(BRANCH_PRED_SIZES['ras']),
                'l1_data_size': random.choice(L1_DATA_SIZE),
                'l1_inst_size': random.choice(L1_INST_SIZE),
                'l2_size': random.choice(L2_SIZE)
            }
            logging.info(f'STARTING annealing schedule with random params {current_params} and temp {self.initial_temp}')
        else:
            current_params = search_params
            logging.info(f'STARTING annealing schedule with given params {current_params} and temp {self.initial_temp}')
        
        current_cost = self._cost_fn(current_params)
        
        best_params = current_params
        best_cost = current_cost

        temperature = self.initial_temp
        iteration = 1
        
        logging.info('')

        # Annealing loop!!!
        while temperature > self.STOPPING_TEMP and iteration < self.MAX_ITER:
            logging.info(f'----- STARTING Iter {iteration} -----')

            # Generate a new candidate solution!
            candidate_params = self._get_neighbour(current_params)
            candidate_cost = self._cost_fn(candidate_params)
            
            if random.random() < self._acceptance_probability(current_cost, candidate_cost, temperature):
                current_params = candidate_params
                current_cost = candidate_cost

                self.cost_table.append(candidate_cost) # for plotting purposes!

                if candidate_cost < best_cost:
                    best_params = candidate_params
                    best_cost = candidate_cost
                    logging.info(f'ACCEPTED with NEW BEST cost = {candidate_cost} with temp = {temperature}')
                else:
                    logging.info(f'ACCEPTED with cost = {candidate_cost} with temp = {temperature}')
            else:
                    logging.info(f'REJECTED parameters with cost {candidate_cost}, moving to next iter')

            temperature *= self.COOLING_RATE
            logging.debug(f'Updated temp to {temperature}')
            logging.info(f'----- FINISHED Iter {iteration} -----')
            logging.info('')

            iteration += 1

        return best_cost, best_params, iteration

    def run(self) -> tuple[float, dict]:
        """
        Performs multiple simulated annealing iterations, using the best result from each iter to hopefully improve results and get out of local minima (if we are in one)
        """
        best_cost = float('inf')
        best_params = None
        self.cost_table = []
        for i in range(self.NUM_SEARCH):
            self.initial_temp //= (i + 1) # we decrease the temperature each iteration, so that we do more of a "local search" 
            # each time

            (curr_cost, curr_params, iterations) = self._search(best_params) # feedthrough mode
            # (curr_cost, curr_params, iterations) = self._search(None) # no feedthrough
            
            if curr_cost < best_cost or i == 0:
                best_cost = curr_cost
                best_params = curr_params
            logging.info('')
            logging.info(f"COMPLETED Annealing schedule {i} complete after {iterations} iters. Best cost = {best_cost}, best params = {best_params}")
            logging.info('')
            self.plot_cost(f"cost_graph_iter_{i}.svg")
        
        print(f"All Annealing schedules COMPLETED. Best cost = {best_cost}, best params = {best_params}")
        self._save_results((best_cost, best_params))
        return best_cost, best_params

    def plot_cost(self, file_name=None):
        """
        Shows a plot of the cost over each SUCCESSFUL iteration (NOT rejected ones)
        """
        if file_name is None:
            file_name = "cost_graph.svg"
        if self.cost_table is not None:
            logging.info(f'Cost table is {self.cost_table}')
            plt.plot(self.cost_table)
            plt.xlabel("Index")
            plt.ylabel("Cost")
            plt.title("Cost Table Over Time")
            plt.grid(True)
            plt.savefig(file_name, format='svg', dpi=300)
            # plt.show() # cant see it because we are on a server!

# Instantiate and run the annealer
annealer = ProcessorSimulatedAnnealer()
(cost, params) = annealer.run()
