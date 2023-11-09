# %% [markdown]
# <a href="https://colab.research.google.com/github/JasonGross/neural-net-coq-interp/blob/main/writeups/Formalizing_Transformers_For_Mech_Interp_Folks_Max_of_small_n_Jason_Gross%2C_Thomas_Kwa%2C_Rajashree_Agrwal.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
## Formalizing Transformers, For Mech Interp Folks
#

# %% [markdown]
#In [*Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*](https://transformer-circuits.pub/2023/monosemantic-features), Bricken et al. say:
#> #### How can we tell if the autoencoder is working?
#>
#> Usually in machine learning we can quite easily tell if a method is working by looking at an easily-measured quantity like the test loss. We spent quite some time searching for an equivalent metric to guide our efforts here, and unfortunately have yet to find anything satisfactory.
#>
#> We began by looking for an information-based metric, so that we could say in some sense that the best factorization is the one that minimizes the total information of the autoencoder and the data. Unfortunately, this total information did not generally correlate with subjective feature interpretability or activation sparsity. (Runs whose feature activations had an average L0 norm in the hundreds but low reconstruction error could have lower total information than those with smaller average L0 norm and higher reconstruction error.)
#>
#> Thus we ended up using a combination of several additional metrics to guide our investigations:
#>
#> 1. **Manual inspection:** Do the features seem interpretable?
#> 2. **Feature density:** we found that the number of “live” features and the percentage of tokens on which they fire to be an extremely useful guide. (See appendix for details.)
#> 3. **Reconstruction loss:** How well does the autoencoder reconstruct the MLP activations? Our goal is ultimately to explain the function of the MLP layer, so the MSE loss should be low.
#> 4. **Toy models:** Having toy models where we know the ground truth and so can cleanly evaluate the autoencoder’s performance was crucial to our early progress.
#>
#>Interpreting or measuring some of these signals can be difficult, though. For instance, at various points we thought we saw features which at first didn’t make any sense, but with deeper inspection we could understand. Likewise, while we have identified some desiderata for the distribution of feature densities, there is much that we still do not understand and which prevents this from providing a clear signal of progress.
#>
#>We think it would be very helpful if we could identify better metrics for dictionary learning solutions from sparse autoencoders trained on transformers.

# %% [markdown]
#This write-up is, in some sense, a response to the above quote.
#
#**Vision:** We want to be able to automate the discovery of mechanistic interpretation of neural networks.
#
#There are two big questions:
#1. How do we discover the interpretation? (Answer: Engineering!)
#2. How do we ensure that the interpretation is good (correct, accurate, human-understandable, etc.)?
#
#We provide a case-study-based plausible theoretical grounding for answering the second question.
#
#Claim: Minimizing information *does* lead to human interpretability, *if* we are clear on what we are proposing to explain.
#
#There are three problems with [the information-based metric](https://transformer-circuits.pub/2023/may-update/index.html#simple-factorization) proposed by the authors of *Towards Monosemanticity*, but none of them are problems with the idea of minimizing information.
#
#*Towards Monosemanticity*, as we understand it, first specifies a problem to be solved and a language of solutions ("reconstruct the matrix of activations $A \approx SD$ where $S$ is sparse and $D$ is a dictionary of features"); then specifies tunable hyperparameters (the coefficient of the sparsity penalty and the size of the dictionary); then finds for each hyperparameter setting the solution that minimizes loss (as measured by combining the reconstruction error and the sparsity penalty); and finally proposes a metric (total information) for comparing the solutions for different hyperparameter settings.
#
#1. Claim: We actually want a tunable knob for picking out interpretations on the pareto-frontier of complicatedness of interpretation (total information) vs. accuracy of interpretation (reconstruction error).  We don't expect there to be a single "best" interpretation in general, but instead a spectrum of interpretations that trade off accuracy for simplicity.  It might make sense to analyze the curve here, perhaps looking for inflection points (intuitively, places where a little more complexity buys a lot more accuracy, or conversely a small relaxation in accuracy saves a lot of complexity), but we shouldn't expect information-based metrics to pick out a single "best interpretation" for us.
#2. The correct trade-off to be looking at is either "for a given bound on information, what's the explanation that provides the most accuracy" or "for a given bound on reconstruction loss, what's the most compact explanation".  The procedure described above does not do this.  Both hyperparameters already control trade-offs between reconstruction loss vs two different proxies for compactness / information / interpretability, and then the resulting optimal solutions are compared using a third proxy.  We have no reason to expect sensible behavior from this procedure.
#3. We can try to measure the understandability / compactness of four different things:
#   1. Local behavioral description: The description of *what* the MLP is doing
#   2. Global behavioral description: The description of *what* the entire neural net is doing
#   3. Global behavioral expectation: The description of *how* it is that *what* the neural net is doing results in our desired behavior (e.g., low loss on the training distribution)
#   4. Global behavioral guarantee: The description of *how* it is that, when accounting for *everything* the neural net is doing, the neural net achieves our desired behavior.  This is (3), but also accounting for all of the "boring" things, e.g., this includes explanations of how it comes to be that the "random noise" from "unimportant heads" is small enough that it doesn't destroy the "important signal".
#
#Total information of the sparse autoencoder decomposition is a measurement of (1).
#There's no reason to expect that the directions in activation space should have a particularly compact representation (and most "human interpretations" just say "there is a direction" for a feature, not what the direction is), which, we believe, is why sparsity seems a better proxy for human interpretability in sparse autoencoders than total information.
#
#This is a problem when trying to minimize information metrics of (1), but *not* when correctly minimizing information metrics for (3) or (4) (and possibly not even for (2)).
#
#There are two things (TODO: replace "things") that ameliorate this issue:
#1. When minimizing information for (3) and (4), we expect the largest gains (complexity reductions) to come not from compact descriptions of the input distribution factoring, but in the ways a given factoring allows a computational complexity reduction.  The *reason* a feature is useful is that you can compute a desired property more simply using that feature than without it.  The benefit of a "DNA feature" is not just in having a compact description of when something is DNA and what should be predicted when something is DNA, but in that other computations can be more simply described when conditioned on DNA.  Or to take a simpler example, the benefit of having a "size" direction that is the principle component of the QK circuit in max-of-n is that it allows us to reason about the OV behavior *as a function of how much attention is paid to the largest element* ($\mathcal{O}(\text{d\_vocab})$ possibilities) rather than as a function of all the sequences of attention weights ($\mathcal{O}(\text{d\_vocab}^{\text{n\_ctx}})$ possibilities).  Next to this asymptotic reduction, the compactness of the description of the particular size direction is peanuts.
#2. If we pick a measure of information [that does not peanalize us for arbitrary choices](https://www.lesswrong.com/posts/KcvJXhKqx4itFNWty/k-complexity-is-silly-use-cross-entropy-instead), such as cross entropy, then we can can do even better!  If a choice is arbitrary (such as the image of the size direction under embedding followed by the query matrix), we won't get docked for the size of that description.  If a choice is not arbitrary, then there's something interesting going on, and we shouldn't be excluding it from our interpretation.
#
# %% [markdown]
###Proofs and Heursitic Arguments from the Mech Interp Lens
#Proofs give a guarantee not just of the biggest thing that happens, but also how it comes to be that nothing else of interest is happening.  Heuristic arguments promise to solve the problem of how to (rigorously) separate out the “nothing interesting is going on” “default assumption” so we can measure its complexity separately, and find actually compact arguments of just the “interesting” “interpretable” stuff.
# %% [markdown]
#We would love to be able to claim that compactness of {proof, heuristic arg} is a good evaluation metric for human interpretability.  We don’t have nearly enough evidence for this, alas (future work!), so instead we aim to present a case-study as evidence that compactness for human interp has *firm formal grounding*.
# %% [markdown]
### A Claim That Needs A Better Section Heading
#We probably ultimately want a measurement of (2) "the thing the neural net is doing", as compactly as possible.
#But if you look closely, we don't have a direct way to measure this.  Instead we're saying "what's a compact way of computing the same thing the neural net computes".  But "computing the thing the neural net computes" *is not the same* as "the computation the neural net is running"!
#
#As far as we can tell, the current approaches try to proxy the gap with reconstruction loss on the quirks and errors made by the model.
#There's some value to this ("there's many ways for relationships to fail, but the only way for them to succeed is respect"), but it clearly isn't adequate for perfect models implementing distinct algorithms.
#And it feels bad to me, saying that we're relying on networks being quirky and error-prone to get decently accuracte explanation evaluations.
#
#We want to claim that a better proxy for the gap between "computing the same thing" and "the computation that's being run" is a *guarantee* (either a proof or a heuristic argument) that the particular computation being run computes the desired result.
#We'll pay some cost by using this as our proxy (we'll have no choice but to include the complete description of the computation being run, though hopefully we can avoid being bitten by this by using cross-entropy), but we believe this proxy gives much firmer grounding for evaluating mech interp.
# %% [markdown]
#In this document, we'll walk through a small case-study or two, applying this frame of proofs and guarantees for mech interp.
# %% [markdown]
### Model Setup: Max of 2
# %% [markdown]
#We'll be looking at the problem of computing the max of two numbers.  We use a 1L attention-only transformer with vocab size 64, model size 32, no layer norm, no biases.  The input is a sequence of two (or later $n$) numbers, and we train on the cross-entropy loss of the prediction in the final sequence position and the correct maximum.  The model has been adversarially overtrained to the point where the accuracy is 100% and the loss is dominated by 32-bit floating point error in the final log-softmax.
# %% [markdown]
### Interpretation
#High-level: The model pays more attention to larger numbers, and copies whatever it's paying attention to.
#
#More detail:
#- There is a "size direction" and a "query direction".  Tokens are embededded with more-or-less uniform overlap with the query direction and more-or-less monotonically-increasing overlap with the size direction (the curve seems to be cubic or quintic, for unclear reasons).
#- The QK circuit has extremely high overlap between the query-direction on the query side and the size-direction on the key side, so that the pre-softmax attention is essentially a scalar multiple of the overlap between the one-hot token vector and the size direction.  Everything else the QK circuit does is essentially negligable.
#- The OV circuit is a low-rank representation of a matrix with high (and more-or-less uniform) diagonal entries and low off-diagonal entries.  We have no explanation for how this comes to be the case.
#- (TODO: check this) There's some additional structure in the noise: query tokens with less overlap with the query direction have (a) less skip-connection noise, (b) larger gaps between the diagonal and off-diagonal entries in the OV circuit, and (c) smaller errors in size-direction overlap.  That is, the errors conspire in our benefit: query tokens that are worse for paying attention to larger tokens have correspondingly larger gaps between them and adjacent tokens in the size-direction, so that we still succeed in paying more attention to tokens larger than the query and less attention to tokens smaller than the query, and the copying behavior on small-gap sequences lines up for reasons we have not yet understood (merely verified).
# %%
#@title setup
# Is this the development version?
DEV = True #@param {type:"boolean"}

try:
    import google.colab # type: ignore
    IN_COLAB = True
except:
    IN_COLAB = False

import os
import sys

if IN_COLAB:
    # Install packages
    %pip install einops
    %pip install jaxtyping
    %pip install transformer_lens

    # Code to download the necessary files (e.g. solutions, test funcs)
    import os, sys
    if not os.path.exists("utils"):
        !curl -o /content/main.zip https://codeload.github.com/JasonGross/neural-net-coq-interp/zip/refs/heads/main
        !unzip /content/main.zip 'neural-net-coq-interp/training/*'
        sys.path.append("/content/utils")
        os.remove("/content/main.zip")
        os.rename("neural-net-coq-interp/training", "utils")
        os.rmdir("neural-net-coq-interp")
else:
    from IPython import get_ipython
    ipython = get_ipython()
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
    import os, sys
    if DEV:
        sys.path.append(f"{os.getcwd()}/../training")
    elif not os.path.exists("utils"):
        !curl -o /content/main.zip https://codeload.github.com/JasonGross/neural-net-coq-interp/zip/refs/heads/main
        !unzip /content/main.zip 'neural-net-coq-interp/training/*'
        sys.path.append(f"{os.getcwd()}/utils")
        os.remove("/content/main.zip")
        os.rename("neural-net-coq-interp/training", "utils")
        os.rmdir("neural-net-coq-interp")

# %%
#@title imports
import train_max_of_2 as max_of_2
from tqdm.auto import tqdm
from analysis_utils import find_size_and_query_direction, imshow, analyze_svd, line
import torch

# %%
model = max_of_2.get_model(train_if_necessary=False)
# %% [markdown]
### Finding the Size Direction
#
# %%
size_direction, query_direction = find_size_and_query_direction(model, plot_heatmaps=True, colorscale='Picnic_r')

# %%