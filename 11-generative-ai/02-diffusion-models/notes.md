# 02 – Diffusion Models

## Problem

`01-gans/notes.md` established a real, working solution to "generate new samples from a distribution
you only have examples of": train two networks against each other in a min-max game. It also
documented, with actually-measured numbers, exactly how unstable that solution is in practice.
Concretely, from `01-gans/notes.md`'s Failure modes section: the loss curves of a *successfully*
trained GAN **do not monotonically decrease** the way an ordinary supervised loss does — `loss_G`
rose and fell across training (`0.75` → `1.68` → `2.24` → `0.83`) even in the well-behaved run — and
a deliberately unbalanced training schedule (5:1 generator:discriminator updates, mismatched learning
rates) reliably produced **mode collapse**: a measured generated-sample std of `[0.0038, 0.0014]`
against a real std of `[1.436, 1.392]`, roughly 1000x too tight, because the min-max objective
rewards "fool the discriminator on average" and a single well-chosen point can satisfy that just as
well as covering the full distribution. This raises an obvious question: is adversarial competition
between two networks actually *necessary* to solve the generation problem, or is it possible to get
comparable results from a single network trained with an ordinary, single-loss regression objective —
one where "loss going down" reliably means "training is working," the same guarantee every earlier
topic in this course had?

## Intuition

Instead of a forger-vs-detective contest, imagine a much more mundane task: take a real data point,
**deliberately corrupt it with a small amount of Gaussian noise**, and train a network to predict
*exactly the noise that was added* so it can be subtracted back off. This is ordinary supervised
regression — the network is given a clear, correct target ("here is the noise vector you should
predict") every single step, unlike a GAN's discriminator whose target quality is itself a moving,
simultaneously-training model. Do this corruption gradually and repeatedly, in many small steps,
until the data has been diffused all the way into pure noise indistinguishable from
$\mathcal N(0, I)$. Now run the trained noise-predictor in reverse: start from literal random noise,
and repeatedly ask "what noise was just added to get here, and what does the world look like with
that noise subtracted off?" Each reverse step nudges the sample a little closer to something the
data distribution would actually produce. Chain enough of these small, individually-easy denoising
steps together, and pure noise gradually resolves into a sample from (approximately) the real
distribution — with no adversary, no second network, and no min-max game anywhere in the loop.

## Why simpler approaches fail

The "simpler approach" that generation history has already tried and that `01-gans/notes.md`
documents failing is the GAN's own predecessor comparison target: a single-network reconstruction
autoencoder, which produces blurry, off-mode averages because MSE-on-raw-data-space is minimized by
the conditional mean of plausible outputs (`01-gans/notes.md`'s Why-simpler-fails section, in full).
Diffusion's key move is not "use MSE instead of adversarial loss" in the naive autoencoder sense —
it is **use MSE to predict a specific, well-defined per-step *noise vector*, not to reconstruct the
data point itself**. Predicting "how much Gaussian noise was added to get from $x_{t-1}$ to $x_t$"
has a unique, unambiguous correct answer for every individual noisy sample $x_t$, because $\epsilon$
was actually sampled and is actually known during training (see Mathematical foundation) — there is
no "multiple valid targets get averaged into a blurry compromise" failure mode, because the target
for each training example is one specific noise vector, not an ambiguous, multi-modal data point.
This is the structural reason diffusion sidesteps *both* documented failure modes at once: it avoids
GAN instability by using one ordinary regression loss instead of a min-max game (Problem section),
and it avoids the autoencoder's blurry-average failure by predicting well-defined per-step noise
instead of ambiguous full data points.

## Mathematical foundation

**Forward process (fixed, not learned).** Define a noise schedule $\beta_1, \dots, \beta_T \in (0,1)$
(this notebook uses a linear schedule) and the Markov chain

$$q(x_t \mid x_{t-1}) = \mathcal N\!\left(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\right)$$

i.e. each step scales the previous sample slightly toward zero and adds a small amount of Gaussian
noise. Let $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$.

**Closed form via the reparameterization trick.** Naively, sampling $x_t$ requires simulating all $t$
intermediate steps. But because each step is Gaussian, the *composition* of $t$ Gaussian steps is
itself Gaussian, and — using the reparameterization trick ($x_t = \sqrt{1-\beta_t}\,x_{t-1} +
\sqrt{\beta_t}\,\epsilon_t$, expand recursively, and use the fact that a sum of independent Gaussians
is Gaussian with variance equal to the sum of variances) — the whole chain collapses to a single
closed-form expression:

$$q(x_t \mid x_0) = \mathcal N\!\left(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I\right)
\quad\Longleftrightarrow\quad
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\, \epsilon,\ \ \epsilon\sim\mathcal N(0,I)$$

This is exactly what `q_sample()` in the notebook implements — a noisy version of $x_0$ at *any*
timestep $t$ can be produced in one line, with **no loop over intermediate steps**, because the
reparameterization trick made the noise term's mean and variance analytically tractable in closed
form. As $t \to T$, $\bar\alpha_t \to 0$ (verified numerically in the notebook: `beta_end=0.2` over
`T=40` steps drives $\bar\alpha_{40}$ down to `0.0135`), so $x_T \approx \epsilon$ — pure noise, with
essentially no trace of $x_0$ left. This is the necessary condition for the reverse process (which
only ever gets to start from actual Gaussian noise at sampling time) to have any chance of working.

**Reverse process (learned).** The reverse conditional $q(x_{t-1}\mid x_t)$ is intractable in general
(it technically depends on the entire data distribution), but Sohl-Dickstein et al. and Ho et al.
(DDPM) show it can be well-approximated by a Gaussian when $\beta_t$ is small, parameterized as

$$p_\theta(x_{t-1} \mid x_t) = \mathcal N\!\left(x_{t-1};\ \mu_\theta(x_t, t),\ \sigma_t^2 I\right)$$

A neural network is trained to predict this Gaussian's parameters. Ho et al.'s key simplification:
rather than have the network predict $\mu_\theta$ directly, reparameterize it in terms of a
**noise-prediction network** $\epsilon_\theta(x_t, t)$ (exactly the notebook's `DenoiseNet`):

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)$$

which gives the reverse sampling step implemented in the notebook's `reverse_sample()`:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z,\qquad z\sim\mathcal N(0,I)\text{ if }t>1\text{ else }0,\quad \sigma_t=\sqrt{\beta_t}$$

**The simplified training objective.** Ho et al. further show that the variational lower bound used
to train $p_\theta$ reduces (after dropping a $t$-dependent weighting term empirically found not to
hurt, and often help) to a strikingly simple form:

$$L_{\text{simple}}(\theta) = \mathbb{E}_{t,\, x_0,\, \epsilon}\left[\ \lVert \epsilon - \epsilon_\theta(\underbrace{\sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon}_{x_t},\ t) \rVert^2\ \right]$$

— sample a real $x_0$, a random timestep $t$, a random noise vector $\epsilon$; construct $x_t$ using
the one-shot closed form above; and train $\epsilon_\theta$ with **ordinary mean-squared error**
against the *known* $\epsilon$ that was actually used to build $x_t$. This is exactly the notebook's
training loop: no adversary, no discriminator, no second loss to balance — one network, one MSE loss,
one target that is always well-defined and always known during training.

## Algorithm

**Training** (per step, exactly what `train_diffusion()` runs):

1. Sample a batch of real $x_0 \sim p_{\text{data}}$.
2. Sample a random timestep $t \sim \text{Uniform}\{1,\dots,T\}$ per example, and noise
   $\epsilon \sim \mathcal N(0,I)$.
3. Build $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$ (closed form, one shot).
4. Predict $\hat\epsilon = \epsilon_\theta(x_t, t)$, compute $\lVert\hat\epsilon - \epsilon\rVert^2$,
   and take one ordinary gradient step (`.backward()` → `optimizer.step()` → `zero_grad()`).

**Sampling** (exactly what `reverse_sample()` runs): start from $x_T \sim \mathcal N(0, I)$; for
$t = T, \dots, 1$, compute $\hat\epsilon_\theta(x_t, t)$, use it to compute $\mu_\theta(x_t,t)$, and
set $x_{t-1} = \mu_\theta(x_t,t) + \sigma_t z$ (with $z=0$ at the final step $t=1$). After $T$
sequential steps, $x_0$ is the generated sample.

## From-scratch implementation

`02-diffusion-models.ipynb` reuses the exact same `sample_real()` toy 8-mode Gaussian ring dataset as
`01-gans/01-gans.ipynb` (same seed-0 usage pattern), for a direct, apples-to-apples comparison. The
denoising network:

```python
class DenoiseNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x, t_frac):
        return self.net(torch.cat([x, t_frac], dim=1))
```

— structurally identical to `09-pytorch`'s `MLP` and `01-gans`'s `Generator`/`Discriminator`:
sub-modules as attributes, `forward()` computing one thing, `.parameters()` discovering everything
automatically. Its only new element versus prior topics' networks is a third scalar input feature
($t/T \in [0,1]$, concatenated to the 2D point) telling the network *which* noise level it's being
asked to denoise — no sinusoidal positional embedding was needed at this toy 2D scale, a plain
scalar fraction sufficed.

Trained on the toy ring for 3000 steps (batch size 128, Adam `lr=1e-3`, `T=40`, `beta_end=0.2`,
`torch.set_num_threads(1)` — actually executed, not simulated):

```
step     0  noise-prediction MSE loss = 1.0135
step   600  noise-prediction MSE loss = 0.4261
step  1200  noise-prediction MSE loss = 0.3753
step  1800  noise-prediction MSE loss = 0.4480
step  2400  noise-prediction MSE loss = 0.3614
step  2999  noise-prediction MSE loss = 0.3497
training time: 1.43s for 3000 steps, T=40
```

Loss decreases roughly monotonically to a plateau (some noise near the end is expected — MSE against
a *randomly re-sampled* $\epsilon$ every step has irreducible variance even at convergence) — in sharp
contrast to `01-gans/notes.md`'s documented `loss_G` oscillation (`0.75 → 1.68 → 2.24 → 0.83`) for
what was still a *successful* GAN run. This is the stability difference predicted in the Problem
section, now actually measured on the same dataset.

The reverse process was then run from pure Gaussian noise, using the trained network:

```
real sample:               mean=[-0.0228,  0.0620]  std=[1.4356, 1.3915]
generated, untrained model: mean=[-3.9013,  7.4302]  std=[11.6368, 16.0552]
generated, trained model:   mean=[-0.0180,  0.0126]  std=[1.4131, 1.4059]
sampling time (T=40 reverse steps, 500 points): 0.0037s
```

The untrained model's reverse process (an untrained noise predictor run for 40 reverse steps)
diverges to a huge, arbitrary scale — passing garbage predictions through the reverse update
compounds over 40 steps rather than converging. The trained model's reverse process, starting from
the *same kind* of pure noise, lands within a few percent of the real distribution's mean and std.

## Practical implementation

As with `01-gans`'s toy scale, "from-scratch" and "practical" collapse into the same code here — the
core mechanism (closed-form forward diffusion, an `nn.Module` noise predictor, one MSE loss, an
ancestral reverse-sampling loop) *is* what production diffusion models (DDPM on images, Stable
Diffusion, Imagen) run; PyTorch's `nn.Module`/`autograd`/`optim` are unchanged. What scales up in
production: the network architecture (U-Nets with attention instead of a 3-layer MLP), $T$ (typically
hundreds to 1000+ steps for the original DDPM formulation), the noise-embedding scheme (sinusoidal
timestep embeddings instead of a raw scalar fraction), and — critically for making sampling fast
enough to be usable — the sampling algorithm itself (see Failure modes: DDIM, distillation). The
alternating-network complexity that made `01-gans` require careful learning-rate balancing has no
analogue here: there is exactly one network and one optimizer.

## Experiment

**Hypothesis (stated before running):** generated-sample statistics — mean, std, and (more sensitive
to per-mode sharpness than aggregate std alone) distance to the nearest of the 8 true ring centers —
should move substantially closer to the real sample's statistics after training than before.

**Setup:** `DenoiseNet` as defined above, `T=40`, `beta_end=0.2`, 3000 training steps, Adam
`lr=1e-3`, batch 128. 500 fresh points generated by running the full reverse process from
$x_T\sim\mathcal N(0,I)$, both with an untrained network (random init) and with the trained network,
compared against the 500-point real reference sample (fixed seed 0, same as `01-gans`).

**Actual result (from the executed notebook):**

```
real                         mean nearest-mode dist=0.0605  frac(dist>0.3)=0.000
generated, untrained         mean nearest-mode dist=16.3610  frac(dist>0.3)=0.996
generated, trained (T=40)    mean nearest-mode dist=0.3095  frac(dist>0.3)=0.450
```

**Interpretation:** mean nearest-mode distance drops by roughly **two orders of magnitude** (16.36 →
0.31) after training — a far larger, more discriminating signal than the mean/std comparison alone,
because it directly measures whether generated points land *on* one of the 8 real clusters rather
than merely having a similar aggregate spread. The trained model's `frac(dist>0.3)=0.45` (still
below the real data's `0.000`) shows the fit is real but imperfect at this toy scale (3000 steps, a
64-hidden-unit MLP) — consistent with, not contradicting, the hypothesis. Directly comparable to
`01-gans/notes.md`'s Experiment section on the identical dataset: this run's generated std
(`[1.413, 1.406]`) lands closer to, and more symmetrically around, the real std (`[1.436, 1.392]`)
than the GAN's (`[1.703, 0.893]`, asymmetric) — one toy-scale data point consistent with (not proof
of) the stability argument from the Problem section, without needing any adversarial balancing act.

**Limitations:** same caveat as `01-gans/notes.md` — a 3-layer, 64-hidden-unit MLP trained for 3000
steps on 2D data demonstrates the *mechanism* working, not a production-grade generative model; a
real diffusion model uses far more steps, a much larger network, and orders of magnitude more
training.

## Failure modes

- **Too few diffusion steps $T$ → harder per-step denoising, worse samples.** Trained a second model
  with `T=5`, `beta_end=0.9` — chosen so $\bar\alpha_5\approx 0.0138$ matches the main run's
  $\bar\alpha_{40}\approx 0.0135$ (same total noise destroyed by the end of the forward process, a
  fair comparison isolating step-count as the only variable), 1500 training steps. Actual measured
  result:

  ```
  model                  sample time (500 pts)    mean nearest-mode dist   frac(dist>0.3)
  T=5 (too few)                          0.0010s                 0.6529   0.860
  T=40 (main)                            0.0037s                 0.3095   0.450
  ```

  `T=5`'s nearest-mode distance (`0.65`) is roughly **2x worse** than `T=40`'s (`0.31`), and its
  `frac(dist>0.3)` (`0.86` vs `0.45`) shows far more generated points landing off any real mode.
  With only 5 steps to reach the same final noise level, each individual $\beta_t$ must be much
  larger, so each reverse step must remove a much bigger, harder-to-predict chunk of noise —
  regression error compounds faster per step when each step's job is coarser.

- **Too many diffusion steps $T$ → slower sampling, a real inference-time cost.** Trained a third
  model with `T=150`, `beta_end=0.055` (again tuned so $\bar\alpha_{150}\approx 0.0148$ matches the
  main run's endpoint), same 1500 training steps. Actual measured result:

  ```
  model                  sample time (500 pts)    mean nearest-mode dist   frac(dist>0.3)
  T=40 (main)                            0.0037s                 0.3095   0.450
  T=150 (too many)                       0.0141s                 0.3596   0.524
  ```

  Sample *quality* is roughly comparable to (marginally worse than, likely noise/undertraining at a
  fixed 1500-step training budget spread thinner across more timesteps) the `T=40` run, but
  **sampling time is ~3.8x larger** — the reverse process is inherently sequential (each step's input
  is the previous step's output, so the $T$ reverse passes cannot be parallelized across time),
  meaning wall-clock sampling cost scales roughly linearly with $T$. At toy 2D scale this difference
  is milliseconds; at production image scale (a full U-Net forward pass per step, hundreds to
  thousands of steps), naive ancestral sampling can take tens of seconds to minutes per image — this
  exact tradeoff is why later work introduced faster samplers: **DDIM** (Denoising Diffusion Implicit
  Models — a non-Markovian reverse process that reaches similar quality in far fewer steps by
  skipping ahead) and **distillation** techniques (training a student model to match a many-step
  teacher's output in one or a few steps). Neither is implemented here — this notebook only
  demonstrates *why* they were needed, by measuring the actual too-few/too-many tradeoff they
  address.

## Real-world usage

The forward-diffusion-then-learn-to-reverse-it mechanism built here — scaled up with U-Net
architectures (convolutional, with attention layers), text conditioning (cross-attention to a text
encoder's embeddings), and orders of magnitude more training — is exactly what powers DDPM (the
original image diffusion formulation this notebook implements at toy scale), Stable Diffusion,
Imagen, and DALL-E 2/3's diffusion-based image decoders, as well as diffusion-based audio (e.g.
WaveGrad) and video generation. The quality/speed sampling tradeoff measured in Failure modes is the
direct motivation for DDIM and distillation-based fast samplers (e.g. progressive distillation,
consistency models) that make these production systems usable at interactive latency instead of
requiring hundreds of sequential network evaluations per output.

## Mental model

A diffusion model is not "one network that learns to generate" the way a plain autoencoder decoder
is — it is one network that learns to solve a much narrower, well-posed problem ("given a noisy
point and how noisy it is, what noise was added") *many times over*, at every noise level from
nearly-clean to pure noise, and generation emerges from chaining many small, individually-easy
denoising steps rather than from any single forward pass. Where `01-gans`'s mental model was "two
networks each running the ordinary `09-pytorch` training loop against each other's live output,"
this topic's is "one network running the ordinary `09-pytorch` training loop against a
*mathematically constructed*, always-known target (the actual noise sampled at that step)" — the
same 5-step loop, but with a training signal that never depends on another simultaneously-changing
model, which is exactly why its loss curve behaves like every earlier topic's and unlike `01-gans`'s.

## Questions to think about

1. The closed-form $q(x_t\mid x_0) = \mathcal N(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I)$ lets
   `q_sample()` jump straight to any timestep $t$ without simulating intermediate steps. Walk through
   why this specifically depends on Gaussian noise (rather than some other noise distribution) at
   every step of the forward chain — what property of Gaussians makes the reparameterization trick's
   "sum of Gaussians is Gaussian" step valid, and what would break if $q(x_t\mid x_{t-1})$ used, say,
   uniform noise instead?
2. This topic's failure-mode experiment tuned `beta_end` differently for `T=5`, `T=40`, and `T=150`
   so that $\bar\alpha_T$ landed at approximately the same value (~0.013-0.015) in all three cases.
   Why was that necessary for the comparison to isolate "step count" as the variable — what would
   have gone wrong (and which of the two measured effects, quality or speed, would have become
   uninterpretable) if all three had instead used the *same* `beta_end`?
3. `01-gans/notes.md`'s Mathematical foundation shows GAN training targets a Nash equilibrium, not a
   minimum of a single loss — there is no scalar that certifies convergence, so judging success
   requires looking at generated samples directly. Does the diffusion training loss
   ($\mathbb E[\lVert\epsilon-\epsilon_\theta\rVert^2]$) have an analogous problem, or does "loss
   trending toward its plateau" reliably indicate the reverse process will produce good samples?
   (Consider: what does the *irreducible* noise-fitting variance in this notebook's loss curve
   plateau — never reaching exactly 0 even at convergence — actually represent?)
4. The reverse sampling loop in `reverse_sample()` is inherently sequential — step $t-1$'s input is
   step $t$'s output. DDIM (mentioned but not implemented in Failure modes) achieves faster sampling
   partly by making the reverse process *non-Markovian* and able to skip steps. Sketch what would
   need to change in `reverse_sample()`'s loop structure (not the full derivation — just the
   mechanical change) to sample using only every other trained timestep (e.g. $t=40,38,36,\dots$)
   instead of every one, and what you'd predict happens to sample quality if you tried this naively
   with the `T=40` model already trained in this notebook (which was never trained to denoise across
   a skipped 2-step gap).
5. This topic's Why-simpler-fails section argues the noise-prediction target is "unambiguous" because
   $\epsilon$ is actually known during training, unlike an autoencoder's ambiguous multi-modal
   reconstruction target. But at a *specific* $x_t$ that could plausibly have come from diffusing
   several different real $x_0$'s (e.g. a heavily noised point near the ring's center, roughly
   equidistant from several of the 8 modes), is the "correct" $\epsilon_\theta(x_t,t)$ prediction
   still unambiguous, or does some of the autoencoder's averaging problem creep back in at high
   noise levels? What does this suggest about why sample quality (Failure modes) degrades more from
   *too few* steps than from *too many*?
