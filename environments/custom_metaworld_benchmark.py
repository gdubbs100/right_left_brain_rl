from collections import OrderedDict
from metaworld import Benchmark, _make_tasks, _ML_OVERRIDE, _MT_OVERRIDE
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
from metaworld.envs.mujoco.sawyer_xyz.v2 import (
    SawyerBasketballEnvV2,
    SawyerBinPickingEnvV2,
    SawyerBoxCloseEnvV2,
    SawyerButtonPressEnvV2,
    SawyerButtonPressTopdownEnvV2,
    SawyerButtonPressTopdownWallEnvV2,
    SawyerButtonPressWallEnvV2,
    SawyerCoffeeButtonEnvV2,
    SawyerCoffeePullEnvV2,
    SawyerCoffeePushEnvV2,
    SawyerDialTurnEnvV2,
    SawyerDoorCloseEnvV2,
    SawyerDoorEnvV2,
    SawyerDoorLockEnvV2,
    SawyerDoorUnlockEnvV2,
    SawyerDrawerCloseEnvV2,
    SawyerDrawerOpenEnvV2,
    SawyerFaucetCloseEnvV2,
    SawyerFaucetOpenEnvV2,
    SawyerHammerEnvV2,
    SawyerHandInsertEnvV2,
    SawyerHandlePressEnvV2,
    SawyerHandlePressSideEnvV2,
    SawyerHandlePullEnvV2,
    SawyerHandlePullSideEnvV2,
    SawyerLeverPullEnvV2,
    SawyerNutAssemblyEnvV2,
    SawyerNutDisassembleEnvV2,
    SawyerPegInsertionSideEnvV2,
    SawyerPegUnplugSideEnvV2,
    SawyerPickOutOfHoleEnvV2,
    SawyerPickPlaceEnvV2,
    SawyerPickPlaceWallEnvV2,
    SawyerPlateSlideBackEnvV2,
    SawyerPlateSlideBackSideEnvV2,
    SawyerPlateSlideEnvV2,
    SawyerPlateSlideSideEnvV2,
    SawyerPushBackEnvV2,
    SawyerPushEnvV2,
    SawyerPushWallEnvV2,
    SawyerReachEnvV2,
    SawyerReachWallEnvV2,
    SawyerShelfPlaceEnvV2,
    SawyerSoccerEnvV2,
    SawyerStickPullEnvV2,
    SawyerStickPushEnvV2,
    SawyerSweepEnvV2,
    SawyerSweepIntoGoalEnvV2,
    SawyerWindowCloseEnvV2,
    SawyerWindowOpenEnvV2,
)

### create custom benchmark to remove push-v2 from ML10 train
### replace with 'soccer-v2' for now
### TODO: perhaps make arbitrary custom env selector
CustomML10_V2 = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("reach-v2", SawyerReachEnvV2),
                    # ("push-v2", SawyerPushEnvV2),
                    ("soccer-v2", SawyerSoccerEnvV2),
                    ("pick-place-v2", SawyerPickPlaceEnvV2),
                    ("door-open-v2", SawyerDoorEnvV2),
                    ("drawer-close-v2", SawyerDrawerCloseEnvV2),
                    ("button-press-topdown-v2", SawyerButtonPressTopdownEnvV2),
                    ("peg-insert-side-v2", SawyerPegInsertionSideEnvV2),
                    ("window-open-v2", SawyerWindowOpenEnvV2),
                    ("sweep-v2", SawyerSweepEnvV2),
                    ("basketball-v2", SawyerBasketballEnvV2)
                    )
                ),
            ),
        (
            "test",
            OrderedDict(
                (
                    ("drawer-open-v2", SawyerDrawerOpenEnvV2),
                    ("door-close-v2", SawyerDoorCloseEnvV2),
                    ("shelf-place-v2", SawyerShelfPlaceEnvV2),
                    ("sweep-into-v2", SawyerSweepIntoGoalEnvV2),
                    ("lever-pull-v2",SawyerLeverPullEnvV2)
                    )
                )
        )
    )
)

custom_ml10_train_args_kwargs = {
    key:dict(
        args = [],
        kwargs = {
            'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)
        }
    )
    for key, _ in CustomML10_V2["train"].items()
}

custom_ml10_test_args_kwargs = {
    key:dict(
        args = [],
        kwargs = {
            'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)
        }
    )
    for key, _ in CustomML10_V2["test"].items()
}

CUSTOMML10_ARGS_KWARGS = dict(
    train = custom_ml10_train_args_kwargs,
    test = custom_ml10_test_args_kwargs
)

class CustomML10(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = CustomML10_V2["train"]
        self._test_classes = CustomML10_V2["test"]
        train_kwargs = custom_ml10_train_args_kwargs

        test_kwargs = custom_ml10_test_args_kwargs
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )

        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _ML_OVERRIDE, seed=seed
        )


## ML3 BENCHMARK ##
## for RL2 training - test benchmark uses _MT_OVERRIDE to place goals available at test time for bicameral
ML3_V2 = OrderedDict(
    (
        (
            "train",
            OrderedDict(
                (
                    ("reach-v2", SawyerReachEnvV2),
                    ("push-v2", SawyerPushEnvV2),
                    ("pick-place-v2", SawyerPickPlaceEnvV2),
                    )
                ),
            ),
        (
            "test",
            OrderedDict(
                (
                    ("reach-v2", SawyerReachEnvV2),
                    ("push-v2", SawyerPushEnvV2),
                    ("pick-place-v2", SawyerPickPlaceEnvV2),
                    ('reach-wall-v2', SawyerReachWallEnvV2),
                    ('push-wall-v2', SawyerPushWallEnvV2),
                    ('pick-place-wall-v2', SawyerPickPlaceWallEnvV2),
                    ("door-open-v2", SawyerDoorEnvV2),
                    ("button-press-v2", SawyerButtonPressEnvV2),
                    ("faucet-open-v2", SawyerFaucetOpenEnvV2)
                    )
                )
        )
    )
)

ml3_train_args_kwargs = {
    key:dict(
        args = [],
        kwargs = {
            'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)
        }
    )
    for key, _ in ML3_V2["train"].items()
}

ml3_test_args_kwargs = {
    key:dict(
        args = [],
        kwargs = {
            'task_id': list(ALL_V2_ENVIRONMENTS.keys()).index(key)
        }
    )
    for key, _ in ML3_V2["test"].items()
}

ML3_ARGS_KWARGS = dict(
    train = ml3_train_args_kwargs,
    test = ml3_test_args_kwargs
)

class ML3(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        self._train_classes = ML3_V2["train"]
        self._test_classes = ML3_V2["test"]
        train_kwargs = ml3_train_args_kwargs

        
        self._train_tasks = _make_tasks(
            self._train_classes, train_kwargs, _ML_OVERRIDE, seed=seed
        )

        ## use _MT_OVERRIDE to get test tasks
        test_kwargs = ml3_test_args_kwargs
        self._test_tasks = _make_tasks(
            self._test_classes, test_kwargs, _MT_OVERRIDE, seed=(seed + 1 if seed is not None else seed)
        )