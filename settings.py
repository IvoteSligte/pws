from dataclasses import dataclass
import pygad
import json


# stores settings that can be changed in between training cycles
@dataclass
class Settings:
    num_parents_mating: int = 4
    parent_selection_type: str = "sss"
    crossover_type: str = "scattered"
    mutation_percent_genes: int = 0.1
    random_mutation_min_val: int = -1.0
    random_mutation_max_val: int = 1.0

    def create_ga(self, initial_population, fitness_func, on_start, on_fitness, on_generation, num_generations, keep_parents, keep_elitism) -> pygad.GA:
        return pygad.GA(
            num_generations=num_generations,
            initial_population=initial_population,
            fitness_func=fitness_func,
            on_start=on_start,
            on_fitness=on_fitness,
            on_generation=on_generation,
            num_parents_mating=self.num_parents_mating,
            parent_selection_type=self.parent_selection_type,
            crossover_type=self.crossover_type,
            keep_parents=keep_parents,
            keep_elitism=keep_elitism,
            mutation_percent_genes=self.mutation_percent_genes,
            random_mutation_min_val=self.random_mutation_min_val,
            random_mutation_max_val=self.random_mutation_max_val,
            mutation_type="random",
        )

    def update_ga(self, ga: pygad.GA):
        ga.num_parents_mating = self.num_parents_mating
        ga.parent_selection_type = self.parent_selection_type
        ga.crossover_type = self.crossover_type
        ga.mutation_percent_genes = self.mutation_percent_genes
        ga.random_mutation_min_val = self.random_mutation_min_val
        ga.random_mutation_max_val = self.random_mutation_max_val

    def to_json(self):
        return json.dumps(self.__dict__)

    def from_json(d: dict):
        s = Settings()
        s.num_parents_mating = d["num_parents_mating"]
        s.parent_selection_type = d["parent_selection_type"]
        s.crossover_type = d["crossover_type"]
        s.mutation_percent_genes = d["mutation_percent_genes"]
        s.random_mutation_min_val = d["random_mutation_min_val"]
        s.random_mutation_max_val = d["random_mutation_max_val"]
        return s
