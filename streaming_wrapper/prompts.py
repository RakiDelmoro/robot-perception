from typing import Dict, List
from dataclasses import dataclass


@dataclass
class PromptContext:
    scene_description: str
    active_objects: List[Dict]
    new_objects: List[int]
    recent_events: List[str]


class PromptBuilder:
    def __init__(self, max_objects: int = 5):
        self.max_objects = max_objects

    def build_scene_prompt(self, context: PromptContext) -> str:
        parts = [self._build_header()]
        if context.active_objects:
            parts.append(self._build_object_context(context.active_objects))
        if context.new_objects:
            parts.append(f"\nNew object(s) appeared: ID={', '.join(map(str, context.new_objects))}")
        if context.recent_events:
            parts.append(f"\nRecent changes: {'; '.join(context.recent_events[-3:])}")
        parts.append(self._build_question())
        return "\n".join(parts)

    def _build_header(self) -> str:
        return """You are analyzing a scene from a robot's perception system.
Describe what happened in this scene:
1. What actions did people/objects COMPLETE (e.g., person sat down, object was picked up)
2. What is the current state of people/objects
3. Any interactions that occurred

Focus on COMPLETED ACTIONS and current states, not ongoing movements.
Keep your response to 1-2 sentences."""

    def _build_object_context(self, objects: List[Dict]) -> str:
        lines = ["Tracked objects in scene:"]
        for obj in objects[:self.max_objects]:
            speed = obj.get('speed_ms', 0)
            speed_str = f"{speed:.1f}m/s" if speed else "?"
            intent = obj.get('intent', '')
            line = f"  ID={obj['id']}: {obj['class']} {obj.get('motion_state', 'unknown')} at {speed_str}"
            if intent:
                line += f", likely {intent}"
            lines.append(line)
        return "\n".join(lines)

    def _build_question(self) -> str:
        return ""


if __name__ == '__main__':
    builder = PromptBuilder(max_objects=5)
    context = PromptContext(
        scene_description="Scene with objects moving",
        active_objects=[
            {'id': 1, 'class': 'person', 'motion_state': 'moving', 'speed_ms': 1.5, 'intent': 'walking'},
            {'id': 2, 'class': 'laptop', 'motion_state': 'stationary', 'speed_ms': 0, 'intent': 'stationary'}
        ],
        new_objects=[1],
        recent_events=['New object: person entered']
    )
    print(builder.build_scene_prompt(context))
