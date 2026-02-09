import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context

_unverified_context = ssl._create_unverified_context()
_https_handler = urllib.request.HTTPSHandler(context=_unverified_context)
_opener = urllib.request.build_opener(_https_handler)
urllib.request.install_opener(_opener)

from absl import app, flags
from ml_collections import config_flags

import os
import warnings
import sys
warnings.filterwarnings(action='ignore')
if not sys.warnoptions:
    warnings.simplefilter("ignore", category=DeprecationWarning)
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'


FLAGS = flags.FLAGS

flags.DEFINE_string('env_type', 'None', 'd4rl-mujoco, d4rl-adroit, d4rl-antmaze')

ENV_TYPES = ['d4rl-mujoco', 
             'd4rl-antmaze', 
             'd4rl-adroit',
             'ogbench-antmaze', 
             'ogbench-humanoidmaze', 
             'ogbench-antsoccer', 
             'ogbench-cube', 
             'ogbench-puzzle', 
             'ogbench-scene',
             'visual-ogb']

ENV_NAMES = {'d4rl-mujoco':['halfcheetah-random-v2',
                            'halfcheetah-medium-v2',
                            'halfcheetah-medium-replay-v2', 
                            'halfcheetah-medium-expert-v2', 
                            'hopper-random-v2', 
                            'hopper-medium-v2', 
                            'hopper-medium-replay-v2', 
                            'hopper-medium-expert-v2',
                            'walker2d-random-v2', 
                            'walker2d-medium-v2', 
                            'walker2d-medium-replay-v2', 
                            'walker2d-medium-expert-v2'], 
             'd4rl-antmaze':['antmaze-umaze-v2', 
                             'antmaze-umaze-diverse-v2', 
                             'antmaze-medium-play-v2', 
                             'antmaze-medium-diverse-v2',
                             'antmaze-large-play-v2',  
                             'antmaze-large-diverse-v2'], 
             'd4rl-adroit':['pen-cloned-v1', 
                            'pen-human-v1', 
                            'pen-expert-v1', 
                            'hammer-cloned-v1', 
                            'hammer-human-v1', 
                            'hammer-expert-v1', 
                            'door-cloned-v1', 
                            'door-human-v1', 
                            'door-expert-v1', 
                            'relocate-cloned-v1',
                            'relocate-human-v1', 
                            'relocate-expert-v1'],
             'ogbench-antmaze':['antmaze-medium-navigate-singletask-v0', 
                                'antmaze-large-navigate-singletask-v0', 
                                'antmaze-giant-navigate-singletask-v0'], 
             'ogbench-humanoidmaze':['humanoidmaze-medium-navigate-singletask-v0', 
                                     'humanoidmaze-large-navigate-singletask-v0', 
                                     'humanoidmaze-giant-navigate-singletask-v0'], 
             'ogbench-antsoccer':['antsoccer-arena-navigate-singletask-v0', 
                                  'antsoccer-medium-navigate-singletask-v0'], 
             'ogbench-cube':['cube-single-play-singletask-v0', 
                             'cube-double-play-singletask-v0', 
                             'cube-triple-play-singletask-v0', 
                             'cube-quadruple-play-singletask-v0'], 
             'ogbench-puzzle':['puzzle-3x3-play-singletask-v0', 
                               'puzzle-4x4-play-singletask-v0', 
                               'puzzle-4x5-play-singletask-v0', 
                               'puzzle-4x6-play-singletask-v0'], 
             'ogbench-scene':['scene-play-singletask-v0'],
             'visual-ogb':['visual-cube-single-play-singletask-task1-v0',
                           'visual-cube-double-play-singletask-task1-v0',
                           'visual-scene-play-singletask-task1-v0',
                           'visual-puzzle-3x3-play-singletask-task1-v0',
                           'visual-puzzle-4x4-play-singletask-task1-v0']}


def main(_):
    env_type = FLAGS.env_type

    def analyze_dataset(env_type):
        if 'd4rl' in env_type:
            for env_name in ENV_NAMES[env_type]:
                print(f'Running {env_name}')
                import d4rl

                env = gymnasium.make('GymV21Environment-v0', env_id=env_name)
                dataset = d4rl.qlearning_dataset(env)

        elif 'ogbench' in env_type:
            for env_name in ENV_NAMES[env_type]:
                print(f'Running {env_name}')
                import ogbench

                env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)
        
        elif 'visual-ogb' in env_type:
            for env_name in ENV_NAMES[env_type]:
                print(f'Running {env_name}')
                import ogbench

                env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name)

    if env_type == 'all':
        for et in ENV_TYPES:
            analyze_dataset(et)
    else:
        analyze_dataset(env_type)

if __name__ == '__main__':
    app.run(main)
    