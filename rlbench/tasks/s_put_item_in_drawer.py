from typing import List, Tuple

import numpy as np
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.conditions import DetectedCondition
from rlbench.backend.task import Task
from rlbench.const import colors


class SPutItemInDrawer(Task):

    def init_task(self) -> None:
        self._options = ['bottom', 'middle', 'top']
        self._anchors = [Dummy('waypoint_anchor_%s' % opt)
                         for opt in self._options]
        self._joints = [Joint('drawer_joint_%s' % opt)
                        for opt in self._options]
        self._drawers = [Shape('drawer_%s' % opt)
                        for opt in self._options]
        self.success_drawer = ProximitySensor('success_drawer')
        self.negate = ProximitySensor('negate')

        self._waypoint1 = Dummy('waypoint1')
        self._item = Shape('item')
        self.register_graspable_objects([self._item])

    def init_episode(self, index) -> List[str]:
        option = self._options[index]
        anchor = self._anchors[index]
        drawer = self._drawers[index]
        
        color_choice = np.random.choice(list(range(len(colors))))
        color_name, color_rgb = colors[color_choice]
        self._item.set_color(color_rgb)

        self._waypoint1.set_position(anchor.get_position())
        success_sensor = ProximitySensor('success_' + option)
        
        self.register_success_conditions([
            DetectedCondition(drawer, self.success_drawer),
            DetectedCondition(self._item, self.negate, negated = True),
            DetectedCondition(self._item, success_sensor)])
        
        self.register_change_point_conditions([
            DetectedCondition(drawer, self.success_drawer),
            DetectedCondition(self._item, self.negate, negated = True),
            DetectedCondition(self._item, success_sensor)  
        ])

        self.register_instructions([
            [
                'Pull out the %s drawer' % option,
                'Pick up the %s block from top of the drawer' % color_name,
                'Put the %s block on the %s drawer' % (color_name, option)
            ]
        ])

        return ['put item in %s drawer' % option,
                'put the block away in the %s drawer' % option,
                'open the %s drawer and place the block inside of it' % option,
                'leave the block in the %s drawer' % option]

    def variation_count(self) -> int:
        return 3

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, - np.pi / 8], [0, 0, np.pi / 8]