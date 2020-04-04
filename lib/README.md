## Description

This folder is supposed to contin the external libraries. 

### Libraries

1. [Slugs](https://github.com/VerifiableRobotics/slugs) : Tried and tested with v0.9 Tag 

   Slugs is used to compute a (maximally) permissive startegy for the system, which encodes multiple (possible all) ways in which the system can react to the adverserial environment and satisfy the temporal-constraints. 

   Reactive synthesis automates the task of developing correct-by-construction finite state machines : rather than writing an implementation and a specification for verifying the system, the engineer needs only devise the specification and the implementation is computed automatically. 

   Slugs offers a frameowrk for GR(1) synthesis and its modification which has exponential complexity in the number of atomic propositions in the specification to synthesize a strategy.
 
   In our problem we consider to type of specification:

   1. Specification with `only safety` properties : We can compute maximally permissive strategy
   2. Specification with `safety and liveness` properties : We can compute a permissive strategy 