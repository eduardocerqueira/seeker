#date: 2022-04-25T17:00:35Z
#url: https://api.github.com/gists/438d044b04c2d50b5afcea94307b2f6c
#owner: https://api.github.com/users/qnbhd

import abc
import math
import random
from abc import ABCMeta
from typing import Type

import pandas

def int_float_mapping(n: int, x_min: float, x_max: float, max_value: int) -> float:
    return x_min + n * (x_max - x_min) / max_value


def float_int_mapping(x: float, x_min: float, x_max: float, max_value: int) -> int:
    return round((x - x_min) * max_value / (x_max - x_min))


def int_to_gray_fixed(n: int, n_bit: int) -> str:
    binary_repr = bin(n)[2:]
    delta = n_bit - len(binary_repr)
    return '0' * delta + binary_repr


def gray_to_int_fixed(gray_code: str) -> int:
    if '1' not in gray_code:
        return 0
    first_one = gray_code.index('1')
    return int(gray_code[first_one:], 2)



class Chromosome(metaclass=ABCMeta):

    @abc.abstractmethod
    def resolve(self, space):
        pass

    @classmethod
    @abc.abstractmethod
    def mutate(cls, point) -> 'Chromosome':
        pass

    @classmethod
    @abc.abstractmethod
    def crossover(cls, p1, p2) -> 'Chromosome':
        pass

    @classmethod
    @abc.abstractmethod
    def random(cls) -> 'Chromosome':
        pass

    @property
    @abc.abstractmethod
    def all_bits(self):
        pass


class Chromosome2d20(Chromosome):

    N_BITS = 20
    MAX_ENCODED = 2**N_BITS

    def __init__(self, x_bits, y_bits):
        self.x_bits = x_bits
        self.y_bits = y_bits

    @property
    def all_bits(self):
        return self.x_bits + self.y_bits

    def resolve(self, space):
        assert len(space) == 2

        x_min = space[0][0]
        x_max = space[0][1]
        y_min = space[1][0]
        y_max = space[1][1]

        i_expr_x = gray_to_int_fixed(self.x_bits)
        i_expr_y = gray_to_int_fixed(self.y_bits)

        f_expr_x = int_float_mapping(
            i_expr_x,
            x_min,
            x_max,
            self.MAX_ENCODED
        )

        f_expr_y = int_float_mapping(
            i_expr_y,
            y_min,
            y_max,
            self.MAX_ENCODED
        )

        return f_expr_x, f_expr_y

    @classmethod
    def mutate(cls, point: 'Chromosome2d20') -> 'Chromosome':
        gca_x = list(point.x_bits)
        gca_y = list(point.y_bits)

        mut_bit_x = random.randint(0, len(gca_x) - 1)
        mut_bit_y = random.randint(0, len(gca_y) - 1)

        gca_x[mut_bit_x] = '1' if gca_x[mut_bit_x] == '0' else '0'
        gca_y[mut_bit_y] = '1' if gca_x[mut_bit_y] == '0' else '0'

        joined_gca_x = ''.join(gca_x)
        joined_gca_y = ''.join(gca_y)

        return cls(joined_gca_x, joined_gca_y)

    @classmethod
    def crossover(cls, p1: 'Chromosome2d20', p2: 'Chromosome2d20') -> 'Chromosome':
        threshold_point_x = random.randint(1, cls.N_BITS)
        threshold_point_y = random.randint(1, cls.N_BITS)

        lhs_x = p1.x_bits[:threshold_point_x]
        rhs_x = p2.x_bits[threshold_point_x:]

        lhs_y = p1.y_bits[:threshold_point_y]
        rhs_y = p2.y_bits[threshold_point_y:]

        x_bits = lhs_x + rhs_x
        y_bits = lhs_y + rhs_y

        return cls(x_bits, y_bits)

    @classmethod
    def random(cls) -> 'Chromosome':
        x_n = random.randint(0, cls.MAX_ENCODED)
        y_n = random.randint(0, cls.MAX_ENCODED)

        x_bits = int_to_gray_fixed(x_n, cls.N_BITS)
        y_bits = int_to_gray_fixed(y_n, cls.N_BITS)

        return cls(x_bits, y_bits)

    def __repr__(self):
        return f'Chromosome2d20(x={self.x_bits}, y={self.y_bits})'


class GeneticAlgorithm:

    def __init__(self, space, objective_callee, fitness_callee,
                 population_size, crossover_prob, mutate_prob,
                 chromosome_cls: Type[Chromosome]):
        """

        :param space: [(x1_min, x1_max), (x2_min, x2_max), ...]
        """
        pass

        self.space = space
        self.fitness_callee = fitness_callee
        self.objective_callee = objective_callee
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutate_prob = mutate_prob
        self.chromosome_cls = chromosome_cls

        self.best_result = 1e10
        self.best_member = None
        self.survival_rate = 0.3
        self.parents_count = 2

        self._population = []

    @property
    def population(self):
        if not self._population:
            self._population = [
                self.chromosome_cls.random()
                for _ in range(self.population_size)
            ]
        return self._population

    @population.setter
    def population(self, new_population):
        self._population = new_population

    def search(self, n_trials):

        results_table = {
            'x': [],
            'y': [],
            'chromosome': [],
            'f(x, y)': [],
            'F(x, y)': []
        }

        for i in range(n_trials):
            population = self.population
            print(population)

            results = []
            total = 0

            for p in population:
                pair = p.resolve(self.space)
                fitness_result = self.fitness_callee(*pair)
                objective_result = self.objective_callee(*pair)

                if self.objective_callee(*pair) < self.best_result:
                    self.best_result = self.objective_callee(*pair)
                    self.best_member = p

                total += fitness_result

                results_table['x'].append(pair[0])
                results_table['y'].append(pair[1])
                results_table['chromosome'].append(p.all_bits)
                results_table['f(x, y)'].append(objective_result)
                results_table['F(x, y)'].append(fitness_result)

                results.append(fitness_result)

            mean = total / self.population_size
            print(f'Mean value of fitness function in {i} generation: {mean}')

            for j in range(len(results)):
                results[j] /= total

            new_population = random.choices(self.population, weights=results,
                                            k=int(self.population_size * self.survival_rate))

            while len(new_population) != self.population_size:
                parent1, parent2 = random.choices(self.population, k=2)

                if random.random() < self.crossover_prob:
                    children = self.chromosome_cls.crossover(parent1, parent2)
                    new_population.append(children)

            for j, p in enumerate(new_population):
                if random.random() < self.mutate_prob:
                    new_population[j] = self.chromosome_cls.mutate(p)

            self.population = new_population

        df = pandas.DataFrame.from_dict(results_table)
        print(df)
        print(self.best_result)
        print(self.best_member)
        print(self.best_member.resolve(self.space))


if __name__ == '__main__':

    # def target(x, y):
    #     return 0.4 * math.sin(0.8 * x + 0.9) + 0.7 * math.cos(0.8 * x - 1) * math.sin(7.3 * x) + y

    def objective(x, y):
        return 0.2*x + 0.2*y - 2*math.cos(x) - 2*math.cos(0.9*y) + 4

    def fitness(x, y):
        return -objective(x, y) + 10

    ga = GeneticAlgorithm(
        [(-1.0, 1.0), (-1.0, 1.0)],
        objective,
        fitness,
        3,
        0.8,
        0.2,
        Chromosome2d20
    )

    ga.search(3)