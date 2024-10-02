# General reply

We would like to thank all the reviewers' replies and valuable comments. We believe many of the presentational concerns are easy to correct, and we have already refined them accordingly.

As follows, we reply to all reviews individually and attach supplementary PDF relating to review jiTP.

We are looking forward to the further discussion!

Thank you again for your time and all the feedbacks!

Best regards,\
Authors

# Review BETH

Thank you for your positive feedback. We appreciate your recognition of the technical soundness of our method and the significance of addressing the challenging problem of infinite dimensions and nonlinear bridges.

## Weaknesses

- We appreciate your suggestion on the improvements of the presentation. The general line of reasoning through the equations is as follows:
    - Equation (1) and (2) state the general settings of infinite-dimensional unconditional and conditional SDEs;
    - Equation (3) and (4) state the general form of time-forward/backward SDEs in finite dimensions stated in [1], where Equation (3) is finite-dimensional version of Equation (1), and Equation (4) serve as the finite dimensional case of Equation (7);
    - Equation (5) is the basis expansion of the Equation (1), the expended equations are therefore in finite dimensions;
    - Equation (6) are the results of applying time-reversal theorem in [1] on each equation in Equation (5), which is a general conclusion that is not restricted to the conditioning, and Equation (7) is the conditional special case;
    - Equation (7) is the infinite-dimensional conditional time-reversed SDEs;
    - Equation (8) is the parameterized SDE used for approximating Equation (7);
    - Equation (9) and (10) are the defined tractable loss functions used for the training procedure.
    - Equation (11)-(15) are the numerical implementations of Equation (9)/(10) under finite-dimensional projections.
    
    we will also edit the corresponding contents to clarify this point in the paper.

- The main contributions of our work over [2] besides the application of neural operators are that we derive the time-reversal of infinite-dimensional nonlinear diffusion process, and propose a computable training object, as stated in Theorem 3.1 and 4.1. We followed the idea of [2] to simulate the diffusion by reversing it. However, the discussion of [2] is only limited in finite dimensions, while in our case, we lift the whole framework into infinite dimensions. We not only show the existence and soundness theoretically, but also implement it in practice, which overcomes the limitation of the dimensionality of [2]'s method. We would like to clarify it by changing the first contribution in the paper:

    > Our study introduces...

    into:

    > Our study introduces a novel approach for simulating infinite-dimensional nonlinear diffusion bridges through the score-matching technique. Expanding upon the research of [Heng et al., 2021] that concentrated on simulating bridges in finite dimensions, we further develop this technique to encompass infinite dimensions. By directly approximating the additional drift introduced by the infinite-dimensional Doob’s h-transform, we are able to reverse the diffusion bridge and simulate it backward using the learned drift operator. We not only show the existence of the infinite-dimensional time-reversed diffusion bridge, but also propose a trainable object for optimization. This framework holds the potential for addressing simulating general infinite-dimensional nonlinear diffusion bridges theoretically and practically, especially those that involve continuous data within infinite-dimensional function spaces.

- The challenge of extending finite-dimensional conditional processes into infinite cases mainly lies in three aspects:
    - The theoretical establishment of the conditioning mechanism, i.e., Doob's h-transform, in infinite-dimensional cases, which has already been studied in [3] and is out of our paper's scope;
    - The theoretical soundness of the infinite-dimensional time-reversed nonlinear diffusion bridge, since existing studies mostly focus on the linear case, e.g. [4];
    - An efficient and straight-forward approach to learn the infinite-dimensional bridge, with both the available learning object and the practical computational model.

## Questions
- Such a projection is necessary because an infinite-dimensional object (Equation (9)/(10)) can not be directly computed. Compared with the finite version of Equation (7), the infinite-dimensional perspective is naturally discretization-independent, which is crucial to our application, since in the stochastic morphometry, a shape is treated as a function in infinite dimensions, one should expect the dynamics are independent of discretization. However, in the previous finite framework, it is hard to obtain such a feature since the discretization will inevitably affect the dynamics of the finite-dimensional process. By working directly in infinite-dimensional space and projecting it into finite dimensions, we can model the underlying true process without losing the computational feasibility.

We hope our responses have clarified your concerns. Thank you for your valuable input.

## Reference
[1] Haussmann, Ulrich G., and Etienne Pardoux. "Time reversal of diffusions." _The Annals of Probability_ (1986): 1188-1205.

[2] Heng, Jeremy, et al. "Simulating diffusion bridges with score matching." _arXiv preprint arXiv:2111.07243_ (2021).

[3] Baker, Elizabeth Louise, et al. "Conditioning non-linear and infinite-dimensional diffusion processes." _arXiv preprint arXiv:2402.01434_ (2024).

[4] Pidstrigach, Jakiw, et al. "Infinite-Dimensional Diffusion Models." _arXiv preprint arXiv:2302.10130_ (2023).

# Review jiTP

Thank you for highlighting the novelty of our work on infinite dimensional bridges and recognizing our adaptation of Heng's work. We appreciate your acknowledgment of the tractable bridge matching loss and the provision of our code.

## Weaknesses

- Thank you for pointing out the potential clarity problems in the paper, we would like to clarify them one by one:
    - We understand that the whole learning methodology appears similar to the classical denoising score matching. Even though the DSM has been widely used for the score-based diffusion model, its application in the diffusion bridge simulation is limited. To our knowledge, it is only considered in [1, 2], while [1] only deals with the finite cases, and our work, where we extend it into infinite dimensions, mainly builds on [2]. Simulating infinite-dimensional nonlinear diffusion bridges is crucial in modelling the biological stochastic morphological evolution. In such simulation, the additional drift term in the SDE, i.e., Doob's h function, meets the target of score matching. Therefore, it motivates [1, 2] and us to use this method to approach the unknown conditional SDE by sampling from the unconditional ones.
    - By $\nabla\log\bar{h}(y_{T-t}|y_0)$, we assume you mean $\nabla\log\bar{h}(T-t, y) = \mathbb{P}(Y(T-t)\in dy|Y(0)=y_0)$, which is the law of $Y(T-t)$ given $Y(0)=y_0$. If so, it is fine to say that this is the transition from $X(T)$ to $X(t)$, but since we are looking at the backward process $Y(t)$, it is more natural to treat it as the transition from $Y(0)$ (starting point) to $Y(T-t)$ (intermediate point). In the forward aspect, such transition can be represented using either $Y(t)$ or $X(t)$, so it is correct to say they are equivalent. However, since everything lives in Hilbert space and there is no Lebesgue measure, we use transition operators instead of densities.
    - The superscript $x_0$ denotes the endpoint of $X^c(t)$, which is the starting point of $Y^c(t)$, is a fixed known point $x_0$. Since we are approximating the backward conditional process $Y^c(t)$, such a path measure is only used to identify the original forward process, but the time reversal should always starts from $Y^c(0)$.
    - Thank you for finding this typo, we intended to make the conditioning between different groups of components, for example, instead of using $p_{t|0}(Y^{c,i}(t)|\hat{Y}^{c,i}(t), Y^{c,i}(0)|\hat{Y}^{c,i}(0))$, we should write $p_{t|0}(Y^{c,i}(t)|\hat{Y}^{c,i}(t), Y^{c,i}(0),\hat{Y}^{c,i}(0))$, which is given $Y^{c,i}(0),\hat{Y}^{c,i}(0)$ at time $0$ and $\hat{Y}^{c,i}(t)$ at time $t$, the probability of $Y^{c,i}(t)$, and the same applies to the rest. We agree that such double "$|$" expression is confusing and wrong, and we have corrected all the occurrences in the text.
    - Theorem 3.1 uses the backwards time index, and the process $Y^c(t)$ runs backwards. Since both $f$ and $a$ are defined in the forward process $X(t)$, we use the $T-t$ index to indicate the reversed time direction. We also find that there is another typo in Line 168, where the last term in the definition of $\bar{f}(t, x)$ should be $\nabla\cdot a(T-t, x)$, i.e., the reversed time, instead of $\nabla\cdot a(t, x)$. For Theorem 4.1, the time index is forward, as we are sampling from the unconditional forward process $X(t)$. 
    - $\nabla\log\mathbb{P}^{x_0}(X(t)\in dx | X(t_{i-1})=x_{t_{i-1}})$ is indeed the infinite-dimensional transition operator in the forward path. As far as we know, the existing infinite-dimensional diffusion models focus on the linear cases (see [2, 3, 4, 5]), while the nonlinearity of the process is one of our interests (please refer to our response to Review k9bH, the second paragraph in Weaknesses section, for the consideration of choosing nonlinear processes) . We generalize the linear method to nonlinear cases, and propose a suitable network structure to learn such an infinite-dimensional object, which reaches the performance to our satisfaction.

- We also appreciate your concerns on the experiments, and we would like to follow up some comments individually:
    - No, our method can scale up to rather high dimensions in practice. We tested it up to 2048 landmarks and it still works well, the reason why we didn't show is because of the lack of clearness in visibility of trajectories in the main paper, but we attach results in the supplementary PDF for demonstration.
    - To our best knowledge, except [6], there is no other infinite-dimensional nonlinear diffusion bridge simulation baseline that we can compare with, our proposed method builds on [6] with improvements on the neural network structure. A qualitative comparison with [6] shall be included in the revision of the paper. For the quantitative evaluations, we couldn't find a reliable and scalable criterion exclusively available to our purpose. A future work could be to develope such baselines for more comprehensive evaluations.
    - As we understand [1]'s work, to simulate the conditional forward process $X^c(t)$, one needs to sample from the learned backward conditional process $Y^c(t)$, and the amount of sampling has to be sufficiently large to reach a reasonable performance. However, since the simulation of $Y^c(t)$ involves the computation of the Jacobian of $a(t, x)$, which is quite expensive when dealing with high dimensions since $a(t, x)$ is a $N\times N$ matrix where $N$ is the dimension, besides, using the learned score for simulating $Y^c(t)$ will inevitably introduce error of estimation, which will be magnified if we use the estimated $Y^c(t)$ for training $X^c(t)$. Even though in the low dimensions that [1] is studying, such error is unobvious, while in our experiment, we observed that there is a significant deviation in high dimensions. Therefore, we look forward to an improvement that only use approximation once, and we take it as one of our future goals.

## Questions
- Yes. In fact, all the training has been done in finite dimensions after discretization. If we fix a finite number of dimensions, our method will reduce to [1]'s method and, therefore, be applicable to finite nonlinear cases.
- Yes. Other potential applications, for example, can be to model variations of shape in human organs in medical imaging [7, 8]. The SDEs we choose for the shape evolution can also be used to model the stochastic fluid dynamics [9], where the bridge simulation can serve as a mesh-free interpolation between the observations.

We hope our responses have addressed your concerns and appreciate your constructive feedback.

## Reference
[1] Heng, Jeremy, et al. "Simulating diffusion bridges with score matching." _arXiv preprint arXiv:2111.07243_ (2021).

[2] Pidstrigach, Jakiw, et al. "Infinite-Dimensional Diffusion Models." _arXiv preprint arXiv:2302.10130_ (2023).

[3] Hagemann, Paul, et al. "Multilevel diffusion: Infinite dimensional score-based diffusion models for image generation." _arXiv preprint arXiv:2303.04772_ (2023).

[4] Zhuang, Peiye, et al. "Diffusion probabilistic fields." _The Eleventh International Conference on Learning Representations_ (2023).

[5] Lim, Jae Hyun, et al. "Score-based diffusion models in function space." _arXiv preprint arXiv:2302.07400_ (2023).

[7] Arnaudon, Alexis, et al. "A stochastic large deformation model for computational anatomy." _Information Processing in Medical Imaging: 25th International Conference, IPMI 2017, Boone, NC, USA, June 25-30, 2017, Proceedings 25. Springer International Publishing_ (2017).

[6] Baker, Elizabeth Louise, et al. "Conditioning non-linear and infinite-dimensional diffusion processes." _arXiv preprint arXiv:2402.01434_ (2024).

[8] Arnaudon, Alexis.,et al. "Stochastic Shape Analysis." In: Chen, K., Schönlieb, CB., Tai, XC., Younes, L. (eds) _Handbook of Mathematical Models and Algorithms in Computer Vision and Imaging. Springer, Cham._ (2023)

[9] Holm, Darryl D. "Variational principles for stochastic fluid dynamics." _Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 471.2176 (2015): 20140963_.

# Review k9bH

Thank you for your positive remarks on the clarity and correctness of our paper. We are glad that you found the experiment results compelling and appreciate your acknowledgment of our approach to discretization invariance.

## Weaknesses

- Your suggestion on the presentation is appreciated. Please refer to our third response to the weaknesses section for Review BETH.

- Thank you for your nice words on the experimental demonstration. The reason why we choose to model the shape dynamics as a nonlinear process is because the linear model will cause unreasonable behaviours of landmarks, including but not limited to the collapsing and overlapping of the landmarks, which reflects on the change of topology of the shapes. We want the change of shape to be diffeomorphic [1], where such mentioned behaviours should be prohibited. An intuitive aspect of our used nonlinear process is that the Gaussian kernel enforce the landmarks to move in sync as they are closer, so that the collapsing and overlapping will never happen theoretically. In theory, such a nonlinear SDE ensures a flow of diffeomorphism [2], while simple linear processes don't.

- Please see the Questions section below.

## Questions
- Thank you for pointing it out. It is indeed "variance exploding SDE"; We agree that the statement is imprecise, and it should be strict to "entropy-regularized optimal transport problem". Thank you for pointing it out.
- We agree with the point and would like to correct the statement as: "Let $x_0$ be a point in $H$ and $\Gamma$ be a set with non-zero measure that contains $x_T$", therefore, the bridge problem we study becomes starting a fixed individual point $x_0$ and conditioning on hitting a non-zero-measure set, in this case, the $h$ function is defined as $h(t, x):=\mathbb{P}(X(T)\in\Gamma\mid X(t)=x)$. Please refer to [3] for more details.
- Yes. A linear process will always have a tractable transition, while a nonlinear process don't usually have a closed form solution, and the "$g$-independent" statement is not completely true, we would like to correct it. Thank you!
- We use $i$ as the basis index of $H$ where $X(t)$ is defined, and $j$ as the basis index of $U$ where $W(t)$ is defined, and $i, j\in\mathbb{Z}$. For each unique component of $X(t)$ along the basis vector $e_i$, we denote $I(i)$ as a finite set of indices $j$ identified by $i$, such set $I(i)$ may contain one or more indices $j$ s, but it should be finite to make $X^i(t) := (X_j(t), j\in I(i)) \in \mathbb{R}^{\#I(i)}$ to be a finite vector, where $\#I(i)$ is the number of elements in $I(i)$. In this case, we split the infinite-dimensional $X(t)$ into infinitely many finite vectors $X^i(t)$ s, and we assume as in [4] that the Lebesgue measure exists in each of the subspace $\mathbb{R}^{\#I(i)}$, therefore, we can define the probability density within these subspaces and sum them up later. A simple example of such setting is that if we choose $H$ and $U$ to be equal with the same dimensionality, then $I(i)=j, X^i(t) = X_i(t) = X_j(t), i=1,2,\dots, j=1,2,\dots$, and the process of each component is a scalar-valued SDE. For Line 476, yes, as we mentioned, the backward process has been split into infinitely many components indexed by $i$ in the same way as we did for $X(t)$, the sum takes over all these components and recovers the infinite-dimensional $Y^c(t)$.
- We have observed the typos and corrected them, thank you!

We appreciate your helpful feedback and hope our responses have addressed your concerns.

## Reference
[1] Younes, Laurent. _Shapes and diffeomorphisms_. Vol. 171. Berlin: Springer (2010).

[2] Kunita, Hiroshi. _Stochastic flows and stochastic differential equations_, volume 24. Cambridge university press (1990).

[3] Baker, Elizabeth Louise, et al. "Conditioning non-linear and infinite-dimensional diffusion processes." _arXiv preprint arXiv:2402.01434_ (2024).

[4] Millet, Annie, David Nualart, and Marta Sanz. "Time reversal for infinite-dimensional diffusions." _Probability theory and related fields_ 82.3 (1989): 315-347.

## Further questions
- If we mark 2D images or 3D scans with particles along their outlines or surfaces, then the image registration problem can be treated as moving those particles into the correct positions. For both deterministic and stochastic registration, if we use a linear process to model the displacements of particle, that is, particles travelling with constant velocities, can collide in finite time (in an extreme scenario, imagine two particles travelling straight towards each other at constant speed, eventually they will collide). This scenario corresponds to the interactions and overlapping between the different parts of the outlines/surfaces, and it is the reason for not using a single displacement-field in image registration when working with large deformations. An intuitive solution to this is to use a nonlinear model that depends on the current state, so that the particles are forced to leave when they are close, and the magnitude should depend on their relative distance. The nonlinear models of deformations are constructed essentially as a sequence of compositions of diffeomorphisms, thus staying diffeomorphic, keeping the one-to-one property, thus particles cannot collide and the warp cannot fold. The non-linear Kunita SDEs in the paper fall into this category. The state-dependent vector field models infinitesimal elements of the sequence of compositions. In the two-particle example above, the particles will here slow down in a way so that they don’t hit each other in finite time.
- Thanks for pointing it out. No, the infinite-dimensional time reversal is unique, but the way of discretizing it into a collection of subprocesses (in either finite or infinite dimensions) is non-unique. The uniqueness of time reversal comes from Equation (3.1) and (3.3) in [4], where the time reversal is treated as the solution of a martingale problem, and the integrability of the coefficient in the time reversal has been verified in the Remark after Theorem 4.3. Thus, the martingale problem is well-defined and equips with a unique solution. In addition, after a choice of discretization which turns the SDEs into finite with an existed density (please see the next response for the condition of the existence of the density), the finite-dimensional time reversal is also unique given a finite forward SDE (see [5]). In fact, we first ensure the uniqueness of the infinite-dimensional time reversal, then discretize it in a way that the density can be defined, and finally gain the time reversal of the finite-dimensional sub-equations (6). We agree that the statement
> "This has a family of time-reversed processes: ..."

is ambiguous, we clarify it by correcting it into:

> "The finite-dimensional processes $\{X_i(t)\}_{i\in\mathbb{Z}^+}$ have unique finite-dimensional time reversals $\{Y_i(t)\}_{i\in\mathbb{Z}^+}$ that satisfy: ..."
- The conditions for ensuring the existence of transition density are stated in [4], Theorem 5.3, we show them briefly as follows and check them against our settings:
    - The diffusion term satisfies assumption (H1), is jointly continuous, and is a $L^2$-function of $x$ with bounded derivatives; The continuity and the integrability can be achieved if we choose smooth kernel $k$ in Equation (34) in our paper, and the assumption (H1) is guaranteed by the per-conditions of infinite-dimensional Doob's $h$-transform.
    - The components of diffusion term must all be positive; can be true if we choose the eigenbasis of the diffusion operator, since the operator is chosen as a trace-class one.
    - The drift term satisfies assumption (H1) and (H2'); can also be true if we choose drift as bounded functions.

    In summary, all the settings required for the existence of the transition density can be satisfied under our choice of nonlinear SDE, and all the linear SDEs commonly used in diffusion models apply to these settings as well.

[5] Anderson, Brian DO. "Reverse-time diffusion equation models." _Stochastic Processes and their Applications_ 12.3 (1982): 313-326.

# Review Eq1m

Thank you for your comprehensive review and kind words. We are pleased that you found the subject matter interesting and relevant, and that the manuscript was clear and informative. We appreciate your recognition of the soundness of our approach and the clarity of our experiment results.

## Weaknesses

- We agree that [1]'s work is fundamental and outstanding, and it indeed motivates our paper. We believe our contributions lie in not only designing a suitable neural operator structure, but more importantly, proving the feasibility of extending the score-matching-based nonlinear diffusion bridge simulation methodology into infinite dimensions, theoretically and practically. In theory, we show the existence of the time-reversed diffusion bridge and construct a computable loss function for optimization; in practice, we demonstrate our method in the challenging biological shape evolutionary tasks. Because of the capacity of directly handling the infinite dimensions, we are no longer restricted to limited landmarks, which is a common aspect in this field [2, 3], and therefore greatly improve the computational efficiency. We also believe our method has more potentials in the fields other than stochastic shape evolution, for example, in the medical image analysis and stochastic fluid dynamics, where the dynamics can be modelled functionally.

- In [4], the theory of infinite-dimensional Doob's h-transform has been established, which is the first step of extending to infinite dimensions (please see our response to Review BETH, where we state the main challenges of our studied task in the Weaknesses section). We follow the work and make the existence of the infinite-dimensional reversed bridge rigorous. Also, [4] uses finite representation of the truncated Fourier basis, and train a MLP to learn the coefficients as finite vectors, while in our work, we use FNO to extract features in Fourier domain at different levels. Empirically, we find such a parameterized neural operator is more natural to use compared with the basis coefficients.

- Thank you for your advice, and we agree. We believe incorporating the starting point $x_0$ is trivial and it will greatly improve the efficiency, we would like to ascribe it into our future goals.

- Yes, and it is even worse when training the forward bridge with the backward bridge, as we discussed in the limitation and the response to Review jiTP in the Weaknesses section. Fortunately, the great feature of neural operator enable us to train on low dimensions and generalize to high dimensions, which shall reduce the computational cost more or less. But still, it requires a rather sufficient data points as we found. Developing a more efficient framework is also our future target, and we are glad to accept your concern.

## Questions

- Yes, more rigorously, the UNO is discretization equivariant (thank you for your suggestion on changing "invariant" to "equivariant", the latter is more precise). The original FNO has been proved equivariant [5], the "U"-shaped structure is designed to reduce the size of the network, compared to the original FNO structure and extract the features in multi-level spatial domains (see also [6]).

- Yes, we study the impact of the step size, but not in a very systemic manner. The choice of 0.01 is because we always normalize our time to be 1, in this case, 0.01 means 100 steps, which is the cheapest amount without losing too much precision. We expect a smaller time step will improve the precision, which has been demonstrated in 200-step, 500-step experiments individually, but visually, the difference is ignorable. Since we don't have reliable quantitative measure, we decide to not show the influence of the step size. 

- We appreciate your careful comments on the texts, and accept the grammatical suggestions. We also follow up comments on some of the questions:
    - Yes, it is the coefficient under $e_i$;
    - Please refer to our response to Review k9bH, the fourth reply in the Questions section;
    - No, we denote $\mathrm{d}x$ as an arbitrary small set with non-zero measure that contains $x$;
    - The "information of the process" mainly refer to the transition (densities if there exists available measure, otherwise operators) of the process in our context, since for a known unconditional SDE, we know everything about its corresponding conditional SDE except the transition density/operator, normally such transition density/operator crosses large steps, e.g., from $0$ to $t$, however, due to the Markovian property, within this large step, the only thing matters is the small step before the current point, while the rest is independent. So they say the "information (transition density/operator) is compressed into this small step".
    - Yes, we forget to clarify it, thanks! :)
    - Yes, the expectation should take over $X(t)$, thank you again!

We would like to thank you for your detailed comments and affirmative feedback, and hope our response address your concerns.

## Reference

[1] Heng, Jeremy, et al. "Simulating diffusion bridges with score matching." _arXiv preprint arXiv:2111.07243_ (2021).

[2] Arnaudon, Alexis, Darryl D. Holm, and Stefan Sommer. "A geometric framework for stochastic shape analysis." _Foundations of Computational Mathematics_ 19 (2019): 653-701.

[3] Arnaudon, Alexis, et al. "A stochastic large deformation model for computational anatomy." _Information Processing in Medical Imaging: 25th International Conference, IPMI 2017, Boone, NC, USA, June 25-30, 2017, Proceedings 25. Springer International Publishing_ (2017).

[4] Baker, Elizabeth Louise, et al. "Conditioning non-linear and infinite-dimensional diffusion processes." _arXiv preprint arXiv:2402.01434_ (2024).

[5] Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." _arXiv preprint arXiv:2010.08895_ (2020).

[6] Rahman, Md Ashiqur, Zachary E. Ross, and Kamyar Azizzadenesheli. "U-no: U-shaped neural operators." _arXiv preprint arXiv:2204.11127_ (2022).