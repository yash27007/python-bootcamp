# 01 – GANs (Generative Adversarial Networks)

## Problem

Every model built through `09-pytorch` and `06-deep-learning` was trained to answer a question about
existing data: is this email spam, what digit is this, what's the next word. A different, harder
problem shows up the moment the goal becomes "produce a *new* example that looks like it belongs to
this dataset" — a new face, a new sentence, a new handwritten digit that was never in the training
set. You are not given the data distribution's equation; you are given a finite set of samples
drawn from it, and asked to build something that can draw *more* samples from (approximately) that
same distribution. There is no label to compute a loss against, because there is no "correct answer"
for what a new sample should be — any sufficiently realistic point counts.

## Intuition

Suppose the real data is "points scattered around 8 locations arranged in a circle" (a picture is
worth a thousand words here — this is the exact toy dataset `01-gans.ipynb` uses). You don't get to
see that description; you only see 500 dots on a page. How would you build a machine that can
produce more dots that belong on that same page?

One instinct: train a network to compress each real dot down to a small code and back out again
(an autoencoder), minimizing the pixel/coordinate-wise reconstruction error. The next section
explains concretely why this produces the *wrong kind* of output. The adversarial idea instead sets
up a contest between two networks: a **forger** (the generator, $G$) that tries to produce
convincing fake dots from random noise, and a **detective** (the discriminator, $D$) that tries to
tell real dots from $G$'s fakes. Neither network is ever told "the right answer" in the direct sense
a classifier is — $D$'s only training signal is "did I correctly call this dot real or fake," and
$G$'s only training signal is "did I fool $D$." Train both simultaneously, and — if it works — $G$
is forced to get better at producing convincing dots precisely because $D$ keeps getting better at
catching the unconvincing ones. Neither network alone knows what "the real distribution" looks like;
the *competition* is what forces $G$ toward it.

## Why simpler approaches fail

The natural simpler alternative is an **autoencoder trained on reconstruction error**: encode each
real point $x$ to a latent code $z = \text{Enc}(x)$, decode back to $\hat x = \text{Dec}(z)$, and
minimize $\lVert x - \hat x \rVert^2$ (mean squared error). Once trained, sample random $z$ and feed
it through $\text{Dec}$ to generate new points.

This fails at the *generation* step for a structural reason, not a tuning reason. Squared error is
minimized, for any fixed input, by predicting the **conditional mean** of plausible targets — this
is the same reason a regression model asked to predict "the next frame of video" under MSE
famously outputs a blurry average of possible frames rather than committing to one sharp one. Applied
to this topic's toy ring: if a decoded latent code $z$ sits ambiguously between two of the 8 modes,
minimizing expected squared error pulls $\text{Dec}(z)$ toward the **midpoint** between those two
modes — a point that is realistic under neither real cluster — rather than toward either sharp mode.
Reconstruction-error training has no mechanism to penalize "producing a blurry average that is
technically low-error but structurally unlike any real sample"; it only measures distance to *known*
training points, never diversity or sharpness of *generated* ones. A GAN's discriminator does have
that mechanism: $D$ can specifically learn to reject blurry, off-mode points, because at training
time it sees real dots that are never in the gaps — the adversarial loss directly penalizes
unrealistic generations, not just distance from one particular reconstruction target.

## Mathematical foundation

**The players.** $G$ maps noise $z \sim p_z$ (a simple, fixed distribution, e.g. standard Gaussian in
$\mathbb{R}^2$) to the data space: $G(z) \sim p_g$, where $p_g$ is whatever distribution $G$'s current
weights induce. $D$ maps a point $x$ in the data space to a scalar in $(0,1)$: its estimated
probability that $x$ came from the real data distribution $p_{\text{data}}$ rather than from $p_g$.

**The objective.** Goodfellow et al.'s original min-max game:

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Read the two halves separately:

- $D$ is trained to **maximize** $V$: push $D(x) \to 1$ for real $x$ (making $\log D(x) \to 0$, its
  best-case value) and push $D(G(z)) \to 0$ for fakes (making $\log(1-D(G(z))) \to 0$ too). $D$'s
  training signal is exactly binary classification — real-vs-fake — which is why the from-scratch
  implementation below trains it with ordinary `BCELoss`.
- $G$ is trained to **minimize** $V$, but $G$ only affects the second term (it has no influence over
  $\mathbb{E}[\log D(x)]$, which doesn't depend on $G$). Minimizing $\mathbb{E}[\log(1-D(G(z)))]$
  means pushing $D(G(z)) \to 1$ — making $D$ call $G$'s fakes real.

**Nash-equilibrium framing.** This is not a single loss surface being descended by one set of
parameters (as in every prior topic) — it is a two-player zero-sum game, where $D$'s move (its
current weights) changes what $G$ is optimizing against, and vice versa. The target is not a
minimum, but a **Nash equilibrium**: a point where neither player can improve their own objective by
moving unilaterally, given the other player's current strategy. Goodfellow et al. show that at the
theoretical optimum (assuming both $G$ and $D$ have unlimited capacity and are optimized exactly),
the optimal discriminator is $D^*(x) = \dfrac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$, and
substituting this back into $V$ reduces the game to minimizing the Jensen-Shannon divergence between
$p_g$ and $p_{\text{data}}$ — which is zero exactly when $p_g = p_{\text{data}}$. That is the sense in
which "$G$ wins" means "$G$'s output distribution matches the real one": not because $G$ was ever
shown the real distribution directly, but because at equilibrium $D$ can do no better than randomly
guess ($D^*(x) = 1/2$ everywhere once $p_g = p_{\text{data}}$), which is only possible once $G$'s
samples are statistically indistinguishable from real ones.

**Mode collapse, mathematically.** The min-max game does not by construction force $G$ to cover
*all* of $p_{\text{data}}$'s support — it only forces $G(z)$ to fool the *current* $D$. Consider a $G$
that has collapsed to producing (almost) one fixed point $x^*$ regardless of $z$: if $x^*$ currently
achieves $D(x^*) \approx 1$ (fools $D$), then $\mathbb{E}_z[\log(1-D(G(z)))] \approx \log(1 - 1) =
-\infty$-ish (a very low, i.e. very good, loss for $G$) with **zero gradient pressure to diversify**
— $\nabla_G \mathbb{E}_z[\log(1-D(G(z)))]$ is small everywhere near this point once $D(x^*)$ is close
to 1, because the loss is already near its minimum for every $z$. $G$'s objective is an *expectation
over $z$*, not a per-mode coverage constraint; a $G$ that ignores $z$ entirely and always emits
$x^*$ can still achieve a low expected loss as long as $D$ hasn't (yet) learned to reject $x^*$. This
is the mathematical root of mode collapse: the objective rewards "fool $D$ on average," which a
single well-chosen point can satisfy just as well as full-distribution coverage would, and gradient
descent has no built-in preference for the latter. Section 6 of the notebook demonstrates this
concretely by deliberately unbalancing training so $G$ can lock onto one point before $D$ catches up.

## Algorithm

Per iteration (this is the standard alternating-update scheme; the notebook uses exactly this):

1. Sample a minibatch of real data $x \sim p_{\text{data}}$ and a minibatch of noise $z \sim p_z$.
2. Compute $D$'s loss on real points ($D(x)$ should be $\to 1$) and on fake points $G(z)$ ($D(G(z))$
   should be $\to 0$) — update **only $D$'s** parameters by descending this loss (`fake_x.detach()`
   ensures no gradient flows into $G$ during this step).
3. Sample a fresh minibatch of noise $z' \sim p_z$, compute $D(G(z'))$, and update **only $G$'s**
   parameters to push $D(G(z')) \to 1$ (fool $D$).
4. Repeat.

In practice, step 3 uses the **non-saturating** loss $-\log D(G(z))$ (minimized) rather than the
literal $\log(1-D(G(z)))$ (minimized) from the original min-max formulation — both push $G$ in the
same direction (fool $D$), but $-\log D(G(z))$ has a much larger gradient early in training, when
$D(G(z))$ is close to 0 and $\log(1-D(G(z)))$'s gradient is nearly flat (see Failure modes,
vanishing gradients). The notebook implements this by training $G$ with `BCELoss(D(G(z)), 1)`
(target label 1 = "real"), which is exactly $-\log D(G(z))$.

## From-scratch implementation

The training-loop *mechanics* here are not new — they are `09-pytorch/02-nn-module-and-training-loop`'s
5-step pattern (forward → loss → `.backward()` → `optimizer.step()` → `optimizer.zero_grad()`),
applied twice per iteration with two separate `nn.Module`s and two separate optimizers instead of
one. `01-gans.ipynb` implements this directly:

```python
class Generator(nn.Module):
    def __init__(self, z_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)
```

— structurally identical in spirit to `09-pytorch`'s `MLP`: sub-modules as attributes, `forward()`
takes only the input, `.parameters()` discovers everything automatically. The only new element is
that `opt_G = torch.optim.Adam(G.parameters(), ...)` and `opt_D = torch.optim.Adam(D.parameters(),
...)` are told to manage *disjoint* parameter sets, so each `.step()` call only ever updates its own
model — this is what makes "update only $D$" / "update only $G$" (Algorithm, steps 2–3) work without
any manual parameter bookkeeping.

Trained on the toy 8-mode ring for 2000 full training iterations (batch size 128, Adam, `lr=1e-3`
for both networks — actually executed, not simulated):

```
epoch     0  loss_D=1.3977  loss_G=0.7466
epoch   400  loss_D=0.7621  loss_G=1.6840
epoch   800  loss_D=0.6876  loss_G=2.2385
epoch  1200  loss_D=1.1411  loss_G=0.8300
epoch  1600  loss_D=1.1470  loss_G=0.8273
epoch  1999  loss_D=1.1151  loss_G=0.7722
training time: 3.5s for 2000 epochs
```

Notice the losses do **not** monotonically decrease the way a supervised loss does — see Failure
modes below for why that's expected, not a bug.

## Practical implementation

At toy 2D scale, "from-scratch" and "practical" are the same code — there is no separate
production-library step the way `sklearn`/`torch.optim` abstracted plain gradient descent in earlier
topics, because a GAN's core mechanism (two `nn.Module`s, two optimizers, alternating updates) *is*
the practical implementation; PyTorch already provides the `nn.Module`/`autograd`/`optim` machinery
this needs. What changes at production scale (image GANs like DCGAN, StyleGAN) is the network
architecture (convolutional generators/discriminators instead of MLPs, progressive growing,
normalization tricks) and training stabilization techniques (spectral normalization, gradient
penalties, Wasserstein loss) layered on top of exactly this same alternating min-max loop — the
loop itself does not change.

## Experiment

**Hypothesis (stated before running):** after training, `G`'s output distribution (sampled via
`G(z)` for fresh random `z`) should visibly spread across the real ring's structure — a
substantially larger, more structured spread than the untrained generator's tight, arbitrary blob —
rather than staying concentrated near its random initialization.

**Setup:** `Generator`/`Discriminator` as defined above, Adam (`lr=1e-3` both), batch size 128, 2000
full iterations, on the 8-mode Gaussian ring (`sample_real`, 500-point real reference sample, fixed
seed 0). Mean and standard deviation of 500 fresh generated points were recorded before and after
training and compared against the real sample's mean/std.

**Actual result (from the executed notebook):**

```
real sample:               mean=[-0.023,  0.062]  std=[1.436, 1.392]
fake, before training:     mean=[ 0.042,  0.117]  std=[0.072, 0.027]
fake, after training:      mean=[ 0.455,  0.519]  std=[1.703, 0.893]
```

**Interpretation:** the untrained generator's output std (`[0.07, 0.03]`) reflects nothing but its
random initialization near the origin — a tight, arbitrary blob unrelated to the real data's shape.
After training, the generated std (`[1.70, 0.89]`) is the same *order of magnitude* as the real
data's (`[1.44, 1.39]`) — a roughly 24x and 33x increase respectively from the untrained baseline —
confirming the hypothesis directionally: training pushed $G$ from "ignores the target distribution's
scale entirely" to "produces a spread comparable to the real distribution's," visible directly in
the notebook's before/after scatter plots (real vs. generated side by side).

**Limitations:** 2000 iterations of a 2-layer, 32-hidden-unit MLP on a 2D toy problem is enough to
show the *mechanism* working, not enough to guarantee full, even coverage of all 8 modes — the
after-training std is not identical to the real std (asymmetric: closer on one axis than the other),
consistent with partial-but-incomplete mode coverage rather than either total collapse or perfect
convergence. A production GAN run for far longer, with a larger network and stabilization tricks,
would be expected to match the target distribution's shape far more precisely; this experiment's
job was only to demonstrate that the described mechanism measurably moves $G$ toward the target
distribution, which it does.

## Failure modes

- **Mode collapse**, demonstrated concretely in `01-gans.ipynb` section 6: deliberately unbalancing
  training (generator learning rate `5e-3` vs. discriminator learning rate `1e-4`, 5 generator
  updates per discriminator update) lets $G$ lock onto a single point before $D$ can learn to reject
  it. Actual measured result: `collapsed generator std: [0.0038, 0.0014]` against a real std of
  `[1.436, 1.392]` — roughly **1000x less spread** than real data and than the properly-balanced run
  above, the concrete signature of mode collapse (all `z` inputs mapping to nearly the same output
  point). Mathematically this is exactly the "Mathematical foundation" section's collapse
  explanation: once one point reliably fools the (undertrained) $D$, $G$'s expected loss is already
  near its minimum and there is no gradient pressure left to diversify.
- **Training instability / oscillating loss:** visible directly in this topic's main training run —
  `loss_D` and `loss_G` rise and fall (`loss_G` goes from `0.75` → `1.68` → `2.24` → `0.83` across
  the run) rather than monotonically decreasing the way a supervised classifier's loss does. This is
  expected behavior for a two-player game (per the Nash-equilibrium framing above): every time $D$
  improves, $G$'s loss temporarily rises until $G$ adapts, and vice versa. Because there is no single
  scalar that certifies "training succeeded," judging a GAN run requires looking at generated
  samples directly (as this notebook does), not just watching a loss curve trend toward zero — a
  GAN loss curve trending toward zero for $G$ specifically can even indicate $D$ has stopped learning
  anything useful, not that $G$ has converged.
- **Vanishing gradients when $D$ becomes too confident too early:** if $D$ learns to perfectly
  separate real from fake before $G$ has learned anything useful, $D(G(z)) \to 0$ for essentially
  every $z$, and the *original* min-max loss $\log(1-D(G(z)))$ has a gradient that flattens out near
  $D(G(z))=0$ (its derivative $\frac{-1}{1-D(G(z))} \cdot D'(G(z))$ shrinks as $D(G(z))\to 0$ relative
  to the alternative) — $G$ receives almost no learning signal despite being far from the target.
  This is exactly why the non-saturating loss $-\log D(G(z))$ (used throughout this notebook, per
  Algorithm above) is the practical default: its gradient stays large precisely when $D(G(z))$ is
  small, i.e. when $G$ most needs to learn.

## Real-world usage

The adversarial min-max game and alternating-optimizer training loop in this topic are the exact
mechanism behind DCGAN (convolutional GANs for images), StyleGAN (the architecture behind most
photorealistic AI-generated faces before diffusion models became dominant), CycleGAN (unpaired
image-to-image translation, e.g. horse-to-zebra), and GAN-based data augmentation for domains with
scarce labeled data (e.g. synthetic medical imaging). Every one of these differs from this notebook
only in network architecture (convolutional generators/discriminators, much larger, many more
training steps) and stabilization tricks layered on top of the same two-network,
alternating-`optimizer.step()` core built here.

## Mental model

A GAN is not "one model learning a distribution" — it is two `nn.Module`s each running the exact
same `09-pytorch` 5-step training loop against each other's current weights instead of against fixed
labels: the discriminator is an ordinary binary classifier whose "labels" happen to include the
generator's live output, and the generator's "loss" is just "how badly did I fool that classifier" —
whichever network is currently weaker sets the training signal for the other, which is precisely
why the loss curves oscillate instead of converging like a normal training run.

## Questions to think about

1. `fake_x.detach()` appears in the discriminator's training step but not in the generator's. Walk
   through exactly what would go wrong (in terms of `.grad` accumulation and which parameters get
   updated) if that `.detach()` were removed.
2. The Mathematical foundation section shows that at the *theoretical* optimum, $D^*(x) =
   \frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$, which equals $1/2$ everywhere once $p_g =
   p_{\text{data}}$. If, partway through a real training run, you observed $D$'s accuracy on
   held-out real-vs-fake points sitting near 50% (random guessing), is that alone good news
   (convergence) or could it also be consistent with a failure mode from this topic? What additional
   signal (this notebook uses one) would you check to tell the difference?
3. The mode-collapse demo used a 5:1 generator:discriminator update ratio with mismatched learning
   rates. Predict, before running it, what would happen if you *reversed* the imbalance (5:1
   discriminator:generator updates, with $D$'s learning rate much higher than $G$'s). Does the
   Mathematical foundation section's collapse argument still predict collapse, a different failure
   mode, or roughly stable training? Why?
4. This topic's non-saturating generator loss ($-\log D(G(z))$) was motivated entirely by the
   vanishing-gradient failure mode. Sketch (no need to fully derive) why $-\log D(G(z))$ and
   $\log(1-D(G(z)))$ push $G$'s parameters in the *same direction* on average, despite having
   different gradient magnitudes — i.e. why swapping one for the other doesn't change what $G$ is
   ultimately trying to do, only how fast/reliably it learns to do it.
5. `notes.md`'s Why-simpler-fails section argues autoencoder reconstruction error produces blurry
   averages because MSE is minimized by the conditional mean. The discriminator in a GAN is itself
   trained with a form of cross-entropy loss, not MSE — why doesn't $D$'s loss suffer from the same
   "averaging" failure when $D$ has to score an ambiguous point that's between two real clusters?
   (Hint: think about what $D$'s loss is a function of — a probability of a binary label — versus
   what the autoencoder's loss is a function of — a predicted point in the same space as the data.)
