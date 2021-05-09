# Genea: A Genetic Algorithm in Python

[Genea](genea.py) is a [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) written in Python, for didactic purpose.

I started writing it for fun, while learning more about how genetic algorithms work.

Given a list of genes and a fitness function, the algorithm starts from a random population and evolves it, generation after generation, until it has converged to a (hopefully) good solution.

In the example below (default values in the source code), I'm using the English alphabet as list of genes and I'm using a fitness function that is inversely proportional to the [hamming distance](https://en.wikipedia.org/wiki/Hamming_distance) between an individual chromosome (a string of genes) and my name `"Domenico De Felice"`.

In other words, the genetic algorithm starts from a pool of random strings, and make them converge towards my name.

Play with it and have fun.

![Demo](demo.gif?raw=true)

