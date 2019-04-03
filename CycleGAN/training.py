from collections import namedtuple


class Training:
    def __init__(self, n_generations: int, log_period: int=100, stat_entries: tuple=()):
        self.n_generations = n_generations
        self.log_period = log_period
        self.stat_class = namedtuple('Stats', ('x', ) + stat_entries)
        self.gen = 0
        self._stat_list = []
        self.stats = None

    def __iter__(self):
        return self._train()

    def _train(self):
        for i in range(self.n_generations):
            self.gen = i
            yield

        self._compile_stats()

    @property
    def log(self):
        return self.gen % self.log_period == 0

    def append_stats(self, *args):
        self._stat_list.append((self.gen, ) + args)

    def _compile_stats(self):
        self.stats = self.stat_class(*zip(*self._stat_list))
