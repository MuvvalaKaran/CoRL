# Project Description 

This is repo for the deep learning project  : Correct-by-synthesis reinforcement learning with temporal logic constraints.

The main aim of this paper is implementation and evaluating the results for the examples discussed y authors in the paper. 

## Introduction 

Autonomous systems are gradually becoming ubiquitous. Beyond simply
demonstrating these systems, it is becoming increasingly important to pro-
vide guarantees that they behave safely and reliably. We can leverage
methods developed from the Formal methods community to synthesize con-
trollers/strategies to achieve a given task with task critical safety and perfor-
mance guarantees. Traditional open-loop synthesis may work well in static
environments but they may fail to find strategies that guarantee task com-
pletion under uncertain or adversarial behavior of the environment. Reac-
tive synthesis is the field that deals with systems that continuously interact
with the environment they operate in. These interactions have novel con-
straints such as real-time constraints, concurrency, parallelism that make
them difficult to model correctly. We can model the interaction between the
system(robot in our case) and the environment as a two player game and
synthesize a winning strategy that satisfies the given specification formulated
within a fragment of temporal logic framework. Thus we can synthesize con-
trollers that guarantee completion of the task using Formal methods. We
can then employ reinforcement learning techniques to learn to achieve the
given task optimally by learning the underlying unknown reward function.
Thus we establish both correctness (with respect to the temporal logic speci-
fications) and optimality (with respect to the a priori unknown performance
criterion) in regards to the task in a stochastic environment for a fragment
of temporal logic specification. Hence, we can guarantee both qualitative
(encoded as the winning condition) and quantitative (optimal reactive con-
trollers ) performance for a system operating in an unknown environment.

## Proposed approach 

This project can be decoupled into two major sub-problems : 

- Computer a set of admissible (winning) strategies that are realizable for a given gam
- Choose a strategy that maximizes the underlying unknow reward function using maximin-Q learning algorithm.

## Conclusion

This project is a work in progress and any kind of feedback is mostly welcome. Contact me at :karan.muvvala@colorado.edu

