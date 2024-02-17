from typing import List
import itertools
import math
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.joint import Joint
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import CustomJointCondition, ConditionSet
from rlbench.const import colors


class SPushButtons(Task):

    def init_task(self) -> None:
        self.buttons_pushed = 0
        self.color_variation_index = 0
        self.target_buttons = [Shape('push_buttons_target%d' % i)
                               for i in range(3)]
        self.target_topPlates = [Shape('target_button_topPlate%d' % i)
                                 for i in range(3)]
        self.target_joints = [Joint('target_button_joint%d' % i)
                              for i in range(3)]
        self.target_wraps = [Shape('target_button_wrap%d' % i)
                             for i in range(3)]
        self.boundaries = Shape('push_buttons_boundary')
        # goal_conditions merely state joint conditions for push action for
        # each button regardless of whether the task involves pushing it

        self.register_waypoint_ability_start(0, self._move_above_next_target)
        self.register_waypoints_should_repeat(self._repeat)

    def init_episode(self, index: int) -> List[str]:
        for tp in self.target_topPlates:
            tp.set_color([1.0, 0.0, 0.0])
        for w in self.target_wraps:
            w.set_color([1.0, 0.0, 0.0])
        # For each color permutation, we want to have 1, 2 or 3 buttons pushed
        self.goal_conditions = [CustomJointCondition(self.target_joints[n], 0.003)
                                for n in range(3)]

        color_choice = np.random.choice(list(range(len(colors))),
            size=3, replace=False)
        self.buttons_pushed = 0

        self.buttons_to_push = 3
        color_names = []
        color_rgbs = []
        chosen_colors = []
        i = 0
        for b in self.target_buttons:
            color_name, color_rgb = colors[color_choice[i]]
            color_names.append(color_name)
            color_rgbs.append(color_rgb)
            chosen_colors.append((color_name, color_rgb))
            b.set_color(color_rgb)
            i += 1

        # for task success, all button to push must have green color RGB
        self.success_conditions = []
        for i in range(self.buttons_to_push):
            self.success_conditions.append(self.goal_conditions[i])

        self.register_success_conditions(
            [ConditionSet(self.success_conditions, True, False)])
        self.register_change_point_conditions(self.success_conditions)

        rtn0 = 'push the %s button' % color_names[0]
        rtn1 = 'press the %s button' % color_names[0]
        rtn2 = 'push down the button with the %s base' % color_names[0]
        for i in range(self.buttons_to_push):
            if i == 0:
                continue
            else:
                rtn0 += ', then push the %s button' % color_names[i]
                rtn1 += ', then press the %s button' % color_names[i]
                rtn2 += ', then the %s one' % color_names[i]

        self.register_instructions([
            [
                'Push the %s button' % color_names[0],
                'Push the %s button' % color_names[1],
                'Push the %s button' % color_names[2],
            ],
            [
                'Activate the %s button' % color_names[0],
                'Activate the %s button' % color_names[1],
                'Activate the %s button' % color_names[2],
            ],
            [
                'SKILL_PUSH_%s' % color_names[0],
                'SKILL_PUSH_%s' % color_names[1],
                'SKILL_PUSH_%s' % color_names[2],
            ]
        ])

        b = SpawnBoundary([self.boundaries])
        for button in self.target_buttons:
            b.sample(button, min_distance=0.1)

        num_non_targets = 3 - self.buttons_to_push
        spare_colors = list(set(colors)
                            - set(
            [chosen_colors[i] for i in range(self.buttons_to_push)]))

        spare_color_rgbs = []
        for i in range(len(spare_colors)):
            _, rgb = spare_colors[i]
            spare_color_rgbs.append(rgb)

        color_choice_indexes = np.random.choice(range(len(spare_colors)),
                                                size=num_non_targets,
                                                replace=False)
        non_target_index = 0
        for i, button in enumerate(self.target_buttons):
            if i in range(self.buttons_to_push):
                pass
            else:
                _, rgb = spare_colors[color_choice_indexes[non_target_index]]
                button.set_color(rgb)
                non_target_index += 1

        return [rtn0, rtn1, rtn2]

    def variation_count(self) -> int:
        return 1

    def step(self) -> None:
        for i in range(len(self.target_buttons)):
            if self.goal_conditions[i].condition_met() == (True, True):
                self.target_topPlates[i].set_color([0.0, 1.0, 0.0])
                self.target_wraps[i].set_color([0.0, 1.0, 0.0])

    def cleanup(self) -> None:
        self.buttons_pushed = 0

    def _move_above_next_target(self, waypoint):
        if self.buttons_pushed >= self.buttons_to_push:
            print('buttons_pushed:', self.buttons_pushed, 'buttons_to_push:',
                  self.buttons_to_push)
            raise RuntimeError('Should not be here.')
        w0 = Dummy('waypoint0')
        x, y, z = self.target_buttons[self.buttons_pushed].get_position()
        w0.set_position([x, y, z + 0.083])
        w0.set_orientation([math.pi, 0, math.pi])

    def _repeat(self):
        self.buttons_pushed += 1
        return self.buttons_pushed < self.buttons_to_push
