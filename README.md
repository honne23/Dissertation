# DE-NOVO DETERMINATION OF PROTEIN TERTIARY STRUCTURE IN A HHPNX 3-FACE CENTERED CUBIC LATTICE WITH MEAN FIELD MULTI-AGENT REINFORCEMENT LEARNING

***See .pdf for full report.***

# Abstract
> The protein structure prediction (PSP) problem is the search for a function that maps a proteins
primary structure, composed of a string of discrete amino acid residues to their respective native
conformation in 3D space denoted as the protein’s tertiary structure. Recent breakthroughs
in the field have utilize techniques such as multiple sequence alignment coupled with residual
convolutional neural networks to derive candidate posteriors over the distribution of inter-residue
distances from which multiple energetically favourable tertiary structures can be generated; these
results are typically annealed to produce a structure of lowest conformational energy. In this
work I examine PSP within the context of multi-agent agent games by applying newly developed
game theoretic techniques that do not rely on pre-existing datasets. Ultimately, my experiments
proved unsuccesful however my research shows that mean field games have the potential to
succefully model the PSP problem and overcome common difficulties in the field such as the
curse of dimensionality as well as horizontal scalability. The system I propose is one that
effectively incorporates inductive bias into the problem formulation in an effort to provide a
richer training signal to the learning agents. Additional techniques such as reward shaping and
risk-sensitive learning are also applied to reduce the sample complexity of the conformational
search space.

# Implementation
The learning agents have been implemented using the Pytorch machine learning framework whereas the environment was implemented solely using Numpy in order to remain agnostic of the underlying machine learning framework as well as to promote high levels of interoperability with other frameworks and projects as Numpy is a common Python package.

The folder structure of the project is as follows:
- Code/Agent
> This contains various agents used for development and testing including a Rainbow Quantile agent implemented using a regular MLP, another Rainbow Quantile agent used for training on Atari in order to verify the implementation. The final agent is the Reside Agent which has been adapted from the Rainbow Quantile to work specifically with the mean field training regime.
- Code/Environment
> This contains the code for the lattice environment as well as the reward structures
according to the hHPNX model.
- Code/Memory
> This contains the prioritised replay memory module used for all agents.
- Code/MultiAgent
> This directory holds the main classes used for training:
- Code/GlobalBuffer
> This stores the neighbourhood information for all agents and is used to maintain
a cache of previously selected actions and to calculate action distributions for use during training.
- Code/MultiAgentRoutine
> This class contains the core training loop for all agents as well as the initialisation logic. All parameters can be passed in here, key word arguments that are not explicitly named (**kwargs) in the class’ constructor are passed directly into the agents for easy configuration. At initialisation, when the GlobalBuffer is empty, random neighbour action distributions are used just as in the original implementation by (Yang et al. 2018).
- Code/Network
> This directory contains various neural networks that were used to verify my implementation of deep Q learning throughout the project. In particular, the ResidueNetwork was adapted from the the QauntileNetwork to perform additional steps of computation
such as the action distribution embedding and the concatenation of the outputs of various layers.
- Code/Peptides
> This contains the output .json files that can be used to visualise outputs during
training.
- Code/PongVideos
> This directory contains output videos from training the QuantileAgent on the OpenAI Gym Atari suite in order to verify the correctness of the implementation.
- Code/DisplayProtein.ipynb
> This top level file contains the example code to visualise the output peptides.
- Code/run_quantile.py
> This contains the code used to train the QuantilAtariAgent.

# Environment
The environment was successfully implemented as a 3D FCC Bravais lattice. The environment successfully reports each residues immediate neighbours on the lattice while enforcing the integrity of the torsion backbone by calculating the euclidean distance between the current residue and its immediate neighbours. The environment can easily be edited to accomodate different lattice structures by changing the constants within the Bravais matrix. I experimented with various penalty structures in order to promote learning in the environment, none of which were successful. 

I tried multiple reward strategies in order to encourage learning:

(1) Setting a constant reward of -10 for actions which broke the self-avoiding-walk (SAW).
(2) Adding -10 to the total reward at local sites on the lattice.
(3) Rewarding all agents only with their immediate local rewards.
(4) Rewarding all agents with the global reward (sum of all local rewards).
(5) Rewarding all agents with the average of all local rewards.

Strategies 3-5 were tried in combination with strategy 1 or 2, none of which were successful. When none of the reward strategies were shown to work I tried to set different initialisation conditions. At first, all proteins were initialised as a straight line in space but the drawback of this approach was that at the beginning of training, the only neighbours a residue would have access to are it’s immediate neighbours on the torsion backbone. 
 
The shaped reward takes into account the covariance of the coordinates of all residues, when they were initialised as a straightline in space, the covariance calculation would evaluate to 0. In an attempt to remedy this, I instead implemented exploring starts (Sutton 2018). Exploring starts is a method of initialisation that starts the agents in random starting positions in the envrionment rather than the initial fixed position. I constructed the exploring starts as random self avoid walk on the lattice. Each time the environment was reset, the residues were initialised to a new random conformation, this had the added benefit of being able to take advantage of the shaped reward although this did not improve training results.