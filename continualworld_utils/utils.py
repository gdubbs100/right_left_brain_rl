import metaworld
from continualworld_utils.constants import MT50
from typing import List

def get_subtasks(name: str, benchmark=MT50, task_set='train') -> List[metaworld.Task]:
    if task_set == 'train':
        return [s for s in benchmark.train_tasks if s.env_name == name]
    elif task_set == 'test':
        return [s for s in benchmark.test_tasks if s.env_name == name]
    else:
        raise ValueError('task_set must be one of test or train')