## Description

This repository holds the code for constructing the env on which our agents; the controlled system robot and the env robot will learn to achieve the given task at hand optimally. This task is hand designed by an humna operator in terms of Reward funnction. The transition system from the game (G_hat) only allows those transition that do not violate the safety guarantees thus ensuring Qualitative behavior. 

We implement reinforcement learning algorithms developed for markov games framework to optimize the quantitative behavior of a robot in presence of an adverserial robot using [minimax_q learning algorithm](https://www2.cs.duke.edu/courses/spring07/cps296.3/littman94markov.pdf).