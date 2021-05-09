#!/usr/bin/env python3
#
# Genea: A Genetic Algorithm in Python, for didactic purpose.
# 
# Copyright © 2020-2021 by Domenico De Felice (https://domdefelice.net)
#
# Permission is granted to anyone to use this software, in source or object code form,
# on any computer system, and to modify, compile, decompile, run, and redistribute it
# to anyone else, subject to the following restrictions:
#
#   - The author makes no warranty of any kind, either expressed or implied, about the
#     suitability of this software for any purpose.
#   - The author accepts no liability of any kind for damages or other consequences of
#     the use of this software, even if they arise from defects in the software.
#   - The origin of this software must not be misrepresented, either by explicit claim
#     or by omission.
#
# Altered versions must be plainly marked as such, and must not be misrepresented as
# being the original software. Altered versions may be distributed in packages under
# other licenses (such as the GNU license).
#
# If you find this software useful, it would be nice if you let me (dom@domdefelice.net)
# know about it, and nicer still if you send me modifications that you are willing to
# share. However, you are not required to do so.
#
# License inspired by Peter Norvig's: https://norvig.com/license.html.

import functools
import random
import sys
import time
import typing as t


def make_chromosome_fitness_function(target_chromosome: str) -> t.Callable[[str], float]:
    """
    Returns a fitness function that given a chromosome (a string of genes, from
    a specified genes alphabet) returns a fitness score that is higher the closer
    the chromosome is to the specified target chromosome.
    """
    def fitness_function(chromosome: str) -> float:
        score = len(target_chromosome) - hamming_distance(chromosome, target_chromosome)

        # Let the score grow exponentially, to accelerate
        # the convergence towards better solutions.
        return score ** 10

    return fitness_function


def hamming_distance(a: str, b: str) -> int:
    """
    Returns the hamming distance between strings a and b.
    """
    assert len(a) == len(b), "Strings must have the same length."
    return sum(1 if a[i] != b[i] else 0 for i in range(len(a)))


def override_line(text: str) -> None:
    """
    Helper function to replace the current output line of text.
    """
    # Hackish but works (tested on Linux only). Assumes there are
    # no more than 100 characters written in the current line.
    sys.stdout.write(f"\r{' ' * 100}\r{text}")
    sys.stdout.flush()


class NaturalSelectionExperiment:
    """
    Runs a natural selection simulation, a.k.a. a genetic algorithm.

    Constructor parameters:
      - population_size = number of individuals.

      - genes_alphabet = string of available genes used to compose the chromosome.

      - chromosome_length = length of the chromosome, i.e. number of genes that compose it.

      - fitness_function = function that given a chromosome returns a fitness score.

      - gene_chance_of_mutation = when a new individual is generated, its genes will mutate
                                  randomly with a chance of 1/gene_chance_of_mutation.

      - max_stale_generations = If after this number of generations the fitness of the
                                fittest individual hasn't improved, the algorithm stops
                                assuming it has already found the best possible solution.

      - verbose = If True, debug messages are printed to stdout.

    While genetic algorithms are designed to ideally converge towards a good solution, there
    are cases where a new generation has a general worse fitness score than the previous
    generation. An older generation may have had an individual with a fitness score that hasn't
    been matched by any newer generation. To work around these cases, we keep track of the fittest
    individual that has ever appeared in any generation.
    """

    def __init__(
        self,
        population_size: int,
        genes_alphabet: str,
        chromosome_length: int,
        fitness_function: t.Callable[[str], float],
        gene_chance_of_mutation: int,
        max_stale_generations: int,
        verbose: bool = False
    ) -> None:
        self.population_size = population_size
        self.genes_alphabet = genes_alphabet
        self.chromosome_length = chromosome_length
        self.gene_chance_of_mutation = gene_chance_of_mutation
        self.max_stale_generations = max_stale_generations
        self.verbose = verbose

        # Let's memoize the fitness function for improved performance.
        self.fitness_function: t.Callable[[str], float] = (functools.lru_cache(maxsize=131072))(fitness_function)


    def run(self) -> str:
        """
        Runs the actual simulation.
        """
        start_time = time.time()

        population = self.gen_initial_population()
        generation_number = 1
        if self.verbose and self.population_size <= 50:
            print(f"Initial population: {population}")

        # Best (fittest) individual found so far.
        best_individual = self.get_fittest_individual(population)

        # Score of the fittest individual found so far.
        best_score = self.fitness_function(best_individual)

        # Number of the generation where the best individual appeared.
        best_generation = generation_number

        # How many generations ago have we found the fittest individual?
        # If we generate self.max_stale_generations generations without
        # improvements, we'll assume we have converged to a (hopefully
        # global) maximum.
        generations_since_best = 0
        
        while generations_since_best < self.max_stale_generations:
            population = self.gen_new_generation(population)
            generation_number = generation_number + 1

            generation_fittest = self.get_fittest_individual(population)
            generation_fittest_score = self.fitness_function(generation_fittest)

            if generation_fittest_score > best_score:
                best_individual = generation_fittest
                best_score = generation_fittest_score
                best_generation = generation_number
                generations_since_best = 0
                if self.verbose:
                    override_line(
                        f"[Generation {generation_number:4}] "
                        f"Fittest chromosome: {generation_fittest} "
                        f"(score {generation_fittest_score:10})\n"
                    )
            else:
                generations_since_best = generations_since_best + 1

                if self.verbose:
                    override_line(
                        f"Generation {generation_number}: "
                        f"Fittest: {generation_fittest} "
                        f"(score {generation_fittest_score} "
                        f"elapsed time {(time.time() - start_time):2.2f}s)"
                    )

        if self.verbose:
            total_time = time.time() - start_time
            override_line(
                f"Fittest genome: {best_individual} "
                f"(generation {best_generation}, "
                f"score: {best_score})\n"
            )
            print(f"Generations: {generation_number}")
            print(
                f"Elapsed time: {total_time:.2f}s "
                f"(avg {total_time / generation_number:.2}s)"
            )

        return best_individual


    def gen_initial_population(self) -> t.List[str]:
        """
        Returns a population of self.population_size random individuals.
        """
        return [self.gen_random_chromosome() for _ in range(self.population_size)]


    def gen_random_chromosome(self) -> str:
        """
        Returns a random individual, i.e. a random chromosome composed by
        self.chromosome_length genes.
        """
        return "".join(random.choices(self.genes_alphabet, k=self.chromosome_length))


    def gen_new_generation(self, old_generation: t.List[str]) -> t.List[str]:
        """
        Evolves population old_generation into a new generation.
        """
        population_fitness = self.compute_population_fitness(old_generation)

        # Our sample of individuals that are going to mate. Individuals with higher
        # fitness levels have higher chances of reproducing, i.e. of transmitting their
        # genes into the next generation.
        fit_individuals_iter = iter(
            self.sample_individuals(
                population=old_generation,
                weights=population_fitness,
                # Mating sample size is larger than the population itself: some individuals
                # may mate with multiple partners, including themselves! Asexual reproduction
                # may be helpful in preserving fit chromosomes through multiple generations.
                sample_size=2 * self.population_size,
            )
        )

        # We know our mating sample has an even size (2 * population size), so we can
        # safely advance two items at a time (the parents) and we won't hit an unhandled
        # StopIteration exception.
        new_generation = [self.mate(fit_individual, next(fit_individuals_iter))
                          for fit_individual in fit_individuals_iter]
        
        return new_generation


    def compute_population_fitness(self, population: t.List[str]) -> t.List[float]:
        return [self.fitness_function(individual) for individual in population]


    def get_fittest_individual(self, population: t.List[str]) -> str:
        return max(population, key=self.fitness_function)


    def sample_individuals(
        self, population: t.List[str], weights: t.List[float], sample_size: int
    ) -> t.List[str]:
        """Draws a weighted sample of size sample_size from the population."""
        return random.choices(population, weights, k=sample_size)


    def mate(self, parent_a: str, parent_b: str) -> str:
        """
        Mates parent a and parent b producing a new individual.
        The new individual will have some genes from parent a and
        some genes from parent b (randomly) and each new gene will
        have a small chance to go through a random mutation.
        """
        new_chromosome = self.crossover(parent_a, parent_b)
        return self.mutate(new_chromosome)

    
    def crossover(self, parent_a: str, parent_b: str) -> str:
        """
        Generates a new individual combining genes from parent a
        and parent b randomly.
        """
        crossover_point = self.gen_crossover_point()

        if crossover_point >= self.chromosome_length:
            crossover_point -= self.chromosome_length
            parent_a, parent_b = parent_b, parent_a

        return parent_a[0:crossover_point] + parent_b[crossover_point:]


    def gen_crossover_point(self) -> int:
        """
        When genes of individuals A and B need to be combined, returns a random
        crossover point X:

        X = 0                                                 All genes from B

        X in {1 ... chromosome_length-1}                      The first X genes
                                                              from A, the rest from B

        X = chromosome_length                                 All genes from A

        X in {chromosome_length+1 ... 2*chromosome_length-1}  The first
                                                              X-chromosome_length
                                                              genes from B, the rest
                                                              from A
        """
        return random.randrange(2 * self.chromosome_length)


    def mutate(self, chromosome: str) -> str:
        """
        Each gene has a random mutation with a probability of 1 / self.gene_chance_of_mutation.
        """
        return "".join([self.gen_random_gene()
                        if random.randint(1, self.gene_chance_of_mutation) == 1
                        else gene
                        for gene in chromosome])


    def gen_random_gene(self) -> str:
        return random.choice(self.genes_alphabet)


TARGET_STRING = "Domenico De Felice"

experiment = NaturalSelectionExperiment(
    population_size=250,
    genes_alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz",
    chromosome_length=len(TARGET_STRING),
    fitness_function=make_chromosome_fitness_function(TARGET_STRING),
    gene_chance_of_mutation=5,
    max_stale_generations=1000,
    verbose=True,
)

solution = experiment.run()
print(solution)
