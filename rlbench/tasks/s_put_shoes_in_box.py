from typing import List, Tuple
import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, NothingGrasped


class SPutShoesInBox(Task):

    def init_task(self):
        shoe1, shoe2 = Shape('shoe1'), Shape('shoe2')
        self.register_graspable_objects([shoe1, shoe2])
        box_lid = Shape('box_lid')

        success_sensor = ProximitySensor('success_in_box')
        success_box = ProximitySensor('success_box')
        success_shoe1 = ProximitySensor('success_shoe1')
        success_shoe2 = ProximitySensor('success_shoe2')

        self.register_success_conditions([
            DetectedCondition(box_lid, success_box),
            DetectedCondition(shoe2, success_shoe2, negated=True),
            DetectedCondition(shoe1, success_sensor),
            DetectedCondition(shoe1, success_shoe1, negated=True),
            DetectedCondition(shoe2, success_sensor),
            NothingGrasped(self.robot.gripper)])

        self.register_change_point_conditions([
            DetectedCondition(box_lid, success_box),
            DetectedCondition(shoe2, success_shoe2, negated=True),
            DetectedCondition(shoe1, success_sensor),
            DetectedCondition(shoe1, success_shoe1, negated=True),
            DetectedCondition(shoe2, success_sensor)
        ])

        self.register_instructions([
            [
                'Open the lid of the box',
                'Pick up a shoe',
                'Put the shoe in the box',
                'Pick up the second shoe',
                'Put the second shoe in the box'
            ]
        ])

    def init_episode(self, index: int) -> List[str]:
        return ['put the shoes in the box',
                'open the box and place the shoes inside',
                'open the box lid and put the shoes inside']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 8], [0, 0, np.pi / 8]
