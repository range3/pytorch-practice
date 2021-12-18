from pathlib import Path
import re
import math


class EpisodePaths:
    def __init__(self, novel_path):
        self._base_dir = Path(novel_path).resolve()

    def __iter__(self):
        file_pattern = re.compile(r'[^-]+-(\d+)')

        def get_key(file):
            match = file_pattern.match(Path(file).name)
            if not match:
                return math.inf
            return int(match.groups()[0])
        return iter(sorted(self._base_dir.glob('*.txt'), key=get_key))


class NovelPaths:
    def __init__(self, base_dir):
        self._base_dir = Path(base_dir).resolve()

    def __iter__(self):
        return self._base_dir.glob('*/N*')

class Episode:
    def __init__(self, episode_path):
        self._episode_path = Path(episode_path)

    def __iter__(self):
        with self._episode_path.open() as f:
            for line in f:
                yield line

class Novel:
    def __init__(self, novel_path):
        self._novel_path = Path(novel_path)
        self._episode_paths = EpisodePaths(novel_path)
    
    def __iter__(self):
        for episode_path in self._episode_paths:
            yield Episode(episode_path)

class DataLoader:
    def __init__(self, base_dir):
        self._novel_paths = NovelPaths(base_dir)

    def __iter__(self):
        for novel_path in self._novel_paths:
            novel = Novel(novel_path)
            for episode in novel:
                for line in episode:
                    yield line
