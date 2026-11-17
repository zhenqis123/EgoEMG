"""Identity-based val/test overlay; original train and split IDs stay intact."""

import json
from pathlib import Path

import numpy as np


class EvaluationPartition:
    def __init__(self, path, role):
        if role not in {'val', 'test'}:
            raise ValueError('evaluation_role must be val or test')
        self.policy = json.loads(Path(path).read_text())
        self.code = {'val': 1, 'test': 2}[role]

    def classify(self, subject, split, gesture, objects, session_id=None):
        """Return train=0, val=1, test=2, or excluded=-1 per annotation."""
        split = np.asarray(split)
        result = np.full(split.shape, -1, dtype=np.int8)
        result[split == 0] = 0
        legacy, recent = self.policy['legacy'], self.policy['recent']
        session_grouped = legacy.get('mode') == 'session'
        if session_grouped and session_id is None:
            raise ValueError('Session-grouped evaluation requires a session ID')
        if (session_grouped and np.isin(split, [1, 2, 3]).any()
                and session_id not in legacy['val_sessions'] + legacy['test_sessions']):
            raise ValueError(f'Legacy evaluation session absent from policy: {session_id}')
        for role, code in [('val', 1), ('test', 2)]:
            if session_grouped:
                if session_id in legacy[role + '_sessions']:
                    result[np.isin(split, [1, 2, 3])] = code
                result[(split == 4) & np.isin(objects, recent[role + '_objects'])] = code
                continue
            user = str(subject) in legacy[role + '_subjects']
            action = np.isin(gesture, legacy[role + '_gestures'])
            result[(split == 1) & user] = code
            result[(split == 2) & action] = code
            result[(split == 3) & user & action] = code
            result[(split == 4) & np.isin(objects, recent[role + '_objects'])] = code
        return result


def load_evaluation_partition(path, role, allowed_splits):
    if path is None and role is None:
        return None
    if path is None or role is None:
        raise ValueError('Partition path and evaluation role must be provided together')
    if 'train' in allowed_splits:
        raise ValueError('Evaluation overlays must not change training selection')
    return EvaluationPartition(path, role)
