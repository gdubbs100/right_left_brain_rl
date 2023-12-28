import metaworld
from continualworld_utils.constants import MT50
from typing import List

def get_subtasks(name: str) -> List[metaworld.Task]:
    return [s for s in MT50.train_tasks if s.env_name == name]