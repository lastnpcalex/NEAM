# Non-Equilibrium Attention Markets

## Introduction

If you need evidence of the value of Ising model, this is not the post for you &mdash; you're late to the stage and the world has already been swallowed by statistical mechanics &mdash; instead, this post seeks to expand your study from mere statistical mechanics, into the regime of attention transformers. To that end, we have two bridge questions:

  1. What happens if we make the fundamental states of the Ising model continuous?
  2. What happens if these interactions are not symmetric?

We will see that we recover Bal's spin transformers. Which raises a raises a new question: **What are the consequences if we use the tools of econophysics and apply it to Bal's spin transformers?**

Our goal today is to begin answering that question, in some small piece.

While this is not the Ising explainer, which I will never write, here is a sampling of the robust literature on the application of Ising model in, specifically, economics:

  * Bornholdt, S. (2001). "Expectation bubbles in a spin model of markets: Intermittency from frustration across scales." International Journal of Modern Physics C, 12(5), 667-674.  

  * Brock, W. A., & Durlauf, S. N. (2001). "Discrete choice with social interactions." The Review of Economic Studies, 68(2), 235-260.  

  * Cont, R., & Bouchaud, J.-P. (2000). "Herd behavior and aggregate fluctuations in financial markets." Macroeconomic Dynamics, 4(2), 170-196.  

  * Galam, S. (2008). "Sociophysics: A review of Galam models." arXiv preprint, arXiv:0803.1800.  

  * Sornette, D. (2014). "Physics and financial economics (1776-2014): Puzzles, Ising and agent-based models." Reports on Progress in Physics, 77(6), 062001.  

  * Zaklan, G., Westerhoff, F., & Stauffer, D. (2009). "Analysing tax evasion dynamics via the Ising model." Journal of Economic Interaction and Coordination, 4(1), 1-14.

## State Agent Attention 

Let's drop the *cut* as they may say someday, and drop ourselves right to the heart of it: the Ising model is a discrete lattice of particles, they have two states (buy or sell), they only interact with their nearest neighbors, the state space of that system yields surprising results if we have infinitely many points on our lattice, and yields tractable insight to technosocial systems across scales and physical details. 

So, let us instead assume that our trader, our agent, labeled with index $$i$$ has an internal state represented by a vector $$x$$ at time $$t$$, and this vector lives in a $$d$$-dimensional vector space, thus: $$x_i^{(t)} \in \mathbb{R}^d$$.

What is this vector space? It is defined as the embeddings of all states for all agents, such that the matrix $$X \in \mathbb{R}^{N \times d}$$, which is the "hidden state" of all N agents.

**TL;DR** Agents have continuous internal states $$d$$, and if we have $$N$$ agents, the totality of the market is represented by an $$N \times d$$ matrix. 

## Three-Fold Path

In standard spin glass physics &mdash; the Ising model &mdash; each nearest neighbor interacts with a symmetric coupling (think of it as a scalar weight describing how two neighborly agents $$s_i$$ and $$s_j$$ interact, but it doesn't matter the order of the interaction, hence symmetric). The energy of this interaction would be $$-s_i J_{ij} s_j$$. Our agents, our states of spin, are given by $$d$$-dimensional vectors, though the energy is directly analagous: $$-x_i^T J x_j.$$

But we have a problem! Transformers do not let vectors (spins/agents) interact directly. Rather, they project states through the three distinct learned weights: $$W_Q, W_K, W_V \in \mathbb{R}^{d \times d_{attn}}$$. These $$d_{attn} \times d$$ matrices, the Query, Key, and Value, are projections that we will assign to economic terms (for interpretation's sake), alongside their mathematical values:

  *   **The Query ($q_i = W_Q x_i$):** Agent $i$'s active demand for information or signals (their market lens).
  *   **The Key ($k_j = W_K x_j$):** Agent $j$'s observable market signal (public supply/posture).
  *   **The Value ($v_j = W_V x_j$):** Agent $j$'s actual capital intent or underlying action (the payload).

Now, when agent $i$ interacts with agent $j$, we find their alignment via the dot product:

$$q_i \cdot k_j = (W_Q x_i)^T (W_k x_j) = x_i^T (W_Q^T W_K) x_j$$

This matrix multiplication yields our spin coupling matrix, $$J = W_Q^T W_K$$, and we can label our energy as $E_{ij} = -q_i \cdot k_j$.  

Ok so what? We have some slightly more complex agentic model, we have an asymmetric coupling, we have an equation for an "energy." At this point, nothing seems like transformers except via fiat, via assumption. **And this is where Jaynes enters**.

## Enter Jaynes, Suddenly Boltzmann Appears

Let $p_j$ be the probability that Agent $i$ pays attention to Agent $j$. By "paid attention" we mean whatever is exchanged via the Q, K, V interactions; all we really care about is state of agent $i$ changes via interactions with agent $j$, and $j$ likewise changes but perhaps not in equal measure. If we find the optimal attention distribution $P = \{p_1, p_2, \dots, p_N\}$ then we have the probability of how a market change of state occurs.

How do we calculate this? Well, let's just use Jaynes' Principle of Maximum Entropy (MaxEnt). There are many ways to think of MaxEnt. Jaynes might say a bounded agent should assume nothing beyond what is strictly known, and that the agent seeks a distribution that maximizes Shannon Entropy. Another way to think about it, is, I believe more intuitive: diffusion is a tendency of systems, unless otherwise constrained, and it is simply a better prior to assume diffusion under constraint, thus MaxEnt. Either way, we have the well travelled path: **we must maximize Shannon entropy**.

Shannon entropy is given by $$H = - \sum_{i=1}^N p_i \ln p_i $$, with two constraints: 

1. **All probabilities sum to 1:** $\sum_{j=1}^N p_j = 1$
2. **We know the value for the average energy:** $\sum_{j=1}^N p_j E_j = \langle E \rangle$

To maximize Shannon entropy for a probability $p_j$ under the constraints of the system, we simply use Lagrange multipliers:

$$\mathcal{L} = - \sum_{j=1}^N p_j \ln p_j - \alpha \left( \sum_{j=1}^N p_j - 1 \right) - \beta \left( \sum_{j=1}^N p_j E_j - \langle E \rangle \right)$$

We can do the motions here, but the result is the result: 

$$p_{ij} = \frac{\exp(\beta (q_i \cdot k_j))}{\sum_{k=1}^N \exp(\beta (q_i \cdot k_k))}$$


Ok so this is familiar to physicists, some of which might not be in the audience, but this is just softmax!

$$ p_{ij} = Softmax(\beta x_i^T (W_Q^T W_K) x_j) $$

$\beta$ is the Lagrange multiplier representing the agent's computational bound. In statistical mechanics, this is $1/k_B T$, but as the dimension size of the vectors $d_k$ grows, the dot products grow as well, pushing the Softmax function into flat regions with zero gradients. To prevent this, they added a scaling factor. So we identify this constraint with our Lagrange multiplier and identify $\beta=1/\sqrt{d_k}$.

Once Agent $i$ establishes their attention distribution across the market, they update their internal state by taking the expected value of the *actual intent* ($V$) of the agents they are watching:

$$x_i^{(t+1)} = \sum_{j=1}^N p_{ij} v_j = p_i V = Softmax \left( \frac{q_i K^T}{\sqrt{d_k}} \right) V,$$ where $p_i = Softmax(\beta x_i^T(W_Q^T W_K)X^T)$. 

If we update across *all* states (the market) then this is just attention:

$$Attention(Q,K,V) = Softmax \left( \frac{QK^T}{\sqrt{d_k}}\right) V$$

## Bal's Non-Equilbirium Attention Market
In a classical physical spin system (like an Ising model) and in classical Efficient Market Hypothesis (EMH) models, systems are assumed to eventually reach Thermodynamic Equilibrium.

For a system to be in true equilibrium, it must satisfy a strict mathematical condition called Detailed Balance. This means that at the microscopic level, the flow of probability from state $i$ to state $j$ is perfectly balanced by the flow from $j$ back to $i$:

$$ P(i\rightarrow j)P(i) = P(j\rightarrow i)P(j) $$

We can rotate all this nuance down to a pithy statement: an efficient market is a MaxEnt market. A MaxEnt dataset is one in which we can't tell what the next bit is; it is Brownian motion, it is a random coin flip. It is up or donw, buy or sell, randomly, forever. This is precisely why Black-Scholes uses a random walk (Geometric Brownian Motion) to model the price.

But Bal's physical translation of the Spin Transfomer, the one we developed in the prior sections, is explicitly asymmetric. *It cannot satisfy Detailed Balance, and thus cannot reach the standard equilibrium.* 

In non-equlibirum statistical mechanics, when detailed balance is broken, the system experiences a continuous dissipation of heat as they update their states against non-recipricoal gradients of their peers (as an aside, this is why cults formed around Dissipative Adaptation are, at least conceptually, incredibly interesting, though like many in this AI space the intellectual lineages they gestured toward were squandared for grift and graft and clout). 

**As we are not mere alchemists, we need a way to calculate this disspiation.**  Let's start by defining a Markovian random walk on the *attention graph*. Imagine a marginal "packet of influence" traversing the network. If this influence is currently localized at Agent $i$, it transitions to Agent $j$ with probability $A_{ij}$. The agents themselves form the discrete state space of the system.

Because the attention matrix $A$ is row-stochastic ($\sum_j A_{ij} = 1$) but strictly non-symmetric and not doubly stochastic, this Markov chain will converge to a unique stationary distribution $\pi$ over the agents, where: 

$$\pi^T A = \pi^T$$

As an aside, softmax attention with finite $\beta$ gives $A_{ij}>0$ everyhwere, which guarantees ergodicity, and thus secures the unique stationary distribution. The vector $\pi$ represents the steady-state systemic influence of each agent.

To calculate this, we look at the attention weights, $A_{ij}$, that is how Agent $i$ shifts to match Agent $j$ through their interaction. The probability flux from agent $i$ to agent $j$ ($J_{ij} = \pi_i A_{ij}$) is not equal to the reverse flux ($J_{ji} = \pi_j A_{ji}$). 

This persistent imbalance means the market acts as a driven non-equilibrium system. We can quantify the resulting "thermodynamic friction" using the **Schnakenberg formula for entropy production rate ($\Pi$)** in a Markov jump process:


$$\Pi = \frac{1}{2} \sum_{i,j} (J_{ij} - J_{ji}) \ln \frac{J_{ij}}{J_{ji}} = \frac{1}{2} \sum_{i,j} (\pi_i A_{ij} - \pi_j A_{ji}) \ln \frac{\pi_i A_{ij}}{\pi_j A_{ji}}$$

Because $\pi_i A_{ij} \neq \pi_j A_{ji}$, the logarithmic term is non-zero, yielding a strictly positive entropy production rate ($\Pi > 0$). 

Thus, an asymmetric attention market is constantly dissipating thermodynamic heat. It is precisely this persistent, microscopic entropy production—driven by the fact that agents have asymmetric attention prevents the market from ever settling into a thermal equilibrium. The system is continuously forced to search for stability, driving the violent reconfigurations of the attention distribution ($P_t$). Macroscopically, this should be observable as volatility spikes.

## Network Dynamics in your Lightcone

To understand what happens next, we must let the clock run. The market does not just update its internal states ($x_i$); the agents are adaptive. They actively update their Query and Key projections ($W_Q, W_K$) to minimizing their local free energy.

This adaptation creates a dangerous feedback loop fueled by the Entropy Production Rate ($\Pi$). 

### Gradient Descent
Agents adjust their projections based on historical success. If an Agent $P$ loses money to Agent $Q$, Agent $P$ updates their Query vector to pay *closer* attention to Agent $Q$'s Key vector in the next time step. 

Meanwhile, Agent $Q$ updates their Key vector to become more deceptive, or updates their Query vector to exploit other inefficiencies, deliberately ignoring Agent $P$.

Mathematically, the agents are performing stochastic gradient descent on their weight matrices. Because the agents have different objective functions and computational bounds, the asymmetry of the coupling matrix $J = W_Q^T W_K$ **amplifies over time**.

### B. The Accumulation of Thermodynamic Friction
As the non-reciprocity amplifies, the system is driven further and further away from detailed balance. 

Recall the Schnakenberg entropy production rate for the stationary distribution $\pi$:
$$\Pi_t = \frac{1}{2} \sum_{i,j} (\pi_i A_{ij} - \pi_j A_{ji}) \ln \frac{\pi_i A_{ij}}{\pi_j A_{ji}}$$

As the institutional agents (or predatory algorithms) successfully extract capital, they become systemic attractors. The stationary distribution $\pi$ concentrates heavily on a few dominant agents. Simultaneously, the asymmetry between $A_{ij}$ (Retail watching Institutions) and $A_{ji}$ (Institutions ignoring Retail) grows extreme.

Therefore, the Entropy Production Rate $\Pi_t$ climbs steadily. This represents the "thermodynamic friction" of the market: the continuous, violent transfer of wealth and information required to sustain the non-equilibrium steady state.

Claim: *the market cannot sustain infinite entropy production.* As $\Pi_t$ increases, the system becomes structurally fragile:

  1. If you continuously drive a system away from equilibrium by injecting a constant flux or either energy or material, the system eventually reaches a state of Self-Organized Criticality. Imagine you are a sandpile. Dropping a single grain of sand at a time, you are never in equilibrium. The pile gets steeper and steeper. Entropy production is the friction of the sand, stacking against gravity. Eventually, there is a critical slope, and even a tiny perturbation (like the next grain of sand) causes the sand dune to slide.
  2. Suppose an asymmetry between Retail agents represented by Alice, and Institutional agents represented by Bob. Alice is bleeding capital to Bob, but Alice isn't an idiot and won't just bleed out. In the terms of an attention agent: gradient descent trains weight matrices to adjust, to change, until Alice's Query matrices better align to the attention of Bob's with winning Key vectors. But look at softmax again with the physics lens:

$$p_j = \frac{exp(\beta E_{ij}}{\sum_k exp(\beta E_{ik}}$$

An arithematic increase in alignment between Alice and Bob creaates a geometric explosion in probability weight, beacuse of that exponential.

  3. As local field generated by the dominant agents ($q_i \cdot k_j$) grows larger relative to the noise, the exponential function in softmax abruptly dominates the denominator. The diverse, high-entropy attention distribution spontaneously collapses. 

The attention of the entire network "snaps" onto a single signal or a small cluster of agents. In statistical mechanics, this is a **Phase Transition** via spontaneous symmetry breaking. In finance, this is a **Herd**.

  When attention is uniform, agents act as independent random variables, and we can see that market shocks are smoothed out in a usual Brownian motion argument. An efficient market, nearly. But, after softmax phase transitions, agents are perfectly correlated, so exogenous noise every signle agent reacts in the exact same direction simultaneously. 

So, a justified claim!

And so we can argue, the market cannot sustain infinite entropy production. As $\Pi_t$ increases, the system becomes structurally fragile, setting the stage for a mechanical crash.

The moment the attention distribution collapses, the systemic Herfindahl-Hirschman Index (HHI) violently spikes from its diversified baseline ($\approx 1/N$) toward its maximum ($1.0$). 

$$\text{HHI}_t = \sum_{j=1}^N p_j^2 \longrightarrow 1.0$$

The agents are no longer acting as independent random variables; they are acting as a single monolithic block, perfectly correlated by their collapsed attention.

## (Bal?)-Black-Scholes

Ok, but this is supposed to be a prediction about economics right? To quote a friend, "value plus discounted," that is to say: we would like to determine a value, or an asset price. I'm not an economist, however, so what I mean is: there's some variable in the system, some expected value, and it is an observable and we would like to know its value. In a market, like the stock market, I know about two things: what is the price I paid, what is the price I hope to get? That is, I would love to predict the price on the market, and the continuous evolutions of that price.

Call that the macroscopic price $S_t$. In standard quantitative finance, the classical Black-Scholes framework allows us to find this price. It relies on Geometric Brownian Motion (GBM):
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Black-Scholes assumes homoskedasticity, or that volatility ($\sigma$) is an exogenous constant, and the market is a passive vessel reacting to the random macroeconomic effects of Current Thing ($dW_t$). 

The Non-Equilibrium Attention Market (NEAM) completely rewrites the diffusion term. To bridge the gap between our agents' latent vectors and the scalar price $S_t$, we map the aggregate intent of the network to market microstructure. Using the standard Kyle (1985) Market Impact Model, price changes are driven by aggregate order flow crossing a market maker's finite liquidity ($\lambda$). 

Because the aggregate order flow is determined by the covariance of the agents' intents, macroscopic volatility is explicitly scaled by the network's Herfindahl-Hirschman Index. 

### Magnetization and Market Memory (The Drift Term)

Before we write the final equation, we must address the most canonical observable in any spin system: **Magnetization**. 

In a standard Ising model, magnetization is the average of all the discrete spins, representing the net polarity of the material. In our Non-Equilibriam Attention Market framework, the Bal transformers, the Magnetization ($\mathbf{M}_t$) is the mean field of all agent state vectors:
$$\mathbf{M}_t = \frac{1}{N} \sum_{i=1}^N x_i^{(t)}$$

This vector $\mathbf{M}_t \in \mathbb{R}^d$ represents the aggregate momentum and risk-posture of the entire market. To translate this into a scalar drift for our price equation, we apply a fixed pricing projection vector $\mathbf{w}$:
$$\mu_t = \mathbf{w}^T \mathbf{M}_t$$

In classical Black-Scholes, the drift ($\mu$) is a static constant (usually the risk-free rate). The market has no memory; it is a Markov process. But in the NEAM framework, the drift $\mu_t$ is a dynamic, state-dependent variable. Because agents update their internal states ($x_i$) based on historical learning and attention, the Magnetization retains structural memory. If the network aligns into a bullish subspace, the drift organically updates to reflect that momentum.

Therefore, our model extracts the two defining features of a market directly from the microscopic spin states:
1. **The Trend ($\mu_t$):** Derived from the macroscopic *Magnetization* of the state vectors.
2. **The Volatility ($\sigma_t$):** Derived from the *Herfindahl-Hirschman Index* (HHI) of the attention matrix.

### The NEAM Stochastic Differential Equation
With both the drift and diffusion terms micro-founded, we can formulate the complete Stochastic Differential Equation. 

Because the aggregate order flow is determined by the covariance of the agents' intents, macroscopic volatility is explicitly scaled by the network's Herfindahl-Hirschman Index, and the drift is driven by Magnetization.

The NEAM Stochastic Differential Equation becomes:
$$dS_t = \mu(\mathbf{M}_t) S_t dt + \left( \lambda \sigma_v N \sqrt{\text{HHI}_t} \right) S_t dW_t$$

*(Where $\lambda$ is Kyle's inverse liquidity parameter, $\sigma_v$ is baseline idiosyncratic intent variance, and $N$ is the total number of agents).*

This equation is the mathematical payoff of the entire framework:

1. **The High-Entropy Regime:** When attention is diversified and $\text{HHI}_t \approx 1/N$, the diffusion multiplier simplifies. The system acts as a standard random walk, absorbing exogenous macroeconomic shocks ($dW_t$) smoothly. The variance scales as $\mathcal{O}(\sqrt{N})$.

2. **The Critical Phase Transition:** When the thermodynamic friction ($\Pi_t$) forces the Softmax attention to collapse, the HHI violently spikes toward $1.0$. The diffusion multiplier instantly scales up from $\mathcal{O}(\sqrt{N})$ to $\mathcal{O}(N)$. 

At that precise moment, the market is critically susceptible. The next exogenous macroeconomic shock ($dW_t$), no matter how mathematically trivial, is multiplied by a massive, perfectly correlated $\mathcal{O}(N)$ factor. The market maker's liquidity is instantly overwhelmed, and the scalar price $S_t$ violently gaps to clear the one-sided demand.

## Endogenous Fat Tails For Sale, Never Used

In classical finance, a crash is modeled as an extreme, "six-sigma" anomaly drawn from a static normal distribution—a purely exogenous stroke of bad luck. In Bal-Black-Scholes, or NEAM, framework a crash is not a random variable, but a deterministic phase transition caused by the network's bounded agents spotaneously aliging their attion to survive in a non-reciproical learning environment. The extreme leptokurtosis (fat tails) and volatility clustering observed in empirical market data are the direct, measurable artifacts of bounded agents collapsing their attention, thus this proposed SDE appears to formally resolve the malingering "fat tail" problem.

## Conclusion and Summary

**Non-Reciprocity** generates a strictly positive **Entropy Production Rate ($\Pi_t$)**. Agents adapting to this non-equilibrium environment causes the **stationary distribution ($\pi$) to concentrate**. The concentrated signals trigger a **Phase Transition** in the bounded MaxEnt Softmax algorithm, causing **Attention Collapse**. The HHI spike multiplies the covariance of the aggregate order flow, exploding the diffusion term in the **NEAM SDE**, resulting in a **Flash Crash**.

And thus, to answer our questions that started this endeavor:

 1. What happens if we make the fundamental states of the Ising model continuous? 
 
     *With a few assumptions, we are able to derive, from MaxEnt, the attention mechanism.*
 
 2. What happens if these interactions are not symmetric?

    *We get non-equilibrium thermodynamics, and constant dissipation causes volitility, collapsed, phase changes.*
 
 3. What are the consequences if we use the tools of econophysics and apply it to Bal's spin transformers?

     **We can justify, if not derive, modifications to Black-Scholes equation, yielding a testable Non-Equilibrium Attention Market (NEAM).**
