import json

from pig_lite.problem.simple_2d import Simple2DProblem, MazeLevel, TerrainLevel, RoomLevel
from pig_lite.environment.gridworld import Gridworld
from pig_lite.game.tictactoe import TicTacToe
# from . training_set import TrainingSet

# this is the common encoding for different level tiles
encoding = {
    'WALL': 1,
    'SPACE': 0,
    'EXPOSED': -1,
    'UNDETERMINED': -2
}

class ProblemFactory():
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_problem(problem_type, problem_size, rng):
        if problem_type == 'maze':
            level = MazeLevel(rng, size=problem_size)
            return Simple2DProblem(level.get_field(), 
                                   level.get_costs(), 
                                   level.get_start(), 
                                   level.get_end())
        elif problem_type == 'terrain':
            level = TerrainLevel(rng, size=problem_size)
            return Simple2DProblem(level.get_field(), 
                                   level.get_costs(), 
                                   level.get_start(), 
                                   level.get_end())
        elif problem_type == 'rooms':
            level = RoomLevel(rng, size=problem_size)
            return Simple2DProblem(level.get_field(), 
                                   level.get_costs(), 
                                   level.get_start(), 
                                   level.get_end())
        elif problem_type == 'tictactoe':
            return TicTacToe(rng, depth=problem_size)
        elif problem_type == 'gridworld':
            return Gridworld.get_random_instance(rng, size=problem_size)
        elif problem_type =='trainset':
            raise NotImplementedError(f'problem_type {problem_type} is not implemented yet')
        else:
            raise ValueError(f'unknown problem_type {problem_type}')
             

    @staticmethod
    def create_problem_from_json(json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
        problem_type = data['type']

        if problem_type == 'Simple2DProblem':
            problem = Simple2DProblem.from_dict(data)
            return problem
        elif problem_type == 'TicTacToe':
            problem = TicTacToe.from_dict(data)
            return problem
        elif problem_type == 'Gridworld':
             problem = Gridworld.from_dict(data)
             return problem
        # elif problem_type == 'TrainingSet':
        #     problem = TrainingSet()
        #     return problem
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")


    @staticmethod
    def create_problem_from_dict(data, problem_type='Simple2DProblem'):
        import numpy as np
        if problem_type == 'Simple2DProblem':
            if not ('board' in data.keys() and 'costs' in data.keys()
                    and 'start_state' in data.keys() and 'end_state' in data.keys()):
                raise ValueError('data dict must contain: "board", "costs", "start_state" and "end_state"')
            if np.array(data['board']).shape != np.array(data['costs']).shape:
                raise ValueError('data["board"] and data["costs"] must have same shape')
            problem = Simple2DProblem.from_dict(data)
            return problem
        if problem_type == 'TicTacToe':
            if not ('start_state' in data.keys() and 'start_player' in data.keys()):
                raise ValueError('data dict must contain: "start_state", "start_player"')
            problem = TicTacToe.from_dict(data)
            return problem
        if problem_type == 'Gridworld':
            if not ('seed' in data.keys() and 'dones' in data.keys()
                    and 'rewards' in data.keys() and 'starts' in data.keys()):
                raise ValueError('data dict must contain: "seed", "dones", "rewards", "starts"')
            problem = Gridworld.from_dict(data)
            return problem
        else:
            raise NotImplementedError(f'problem_type {problem_type} is not implemented yet')

