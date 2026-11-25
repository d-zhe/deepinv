r"""
Experimental DDNM soft updates for noisy observations
==========================================================

This script investigates a noise-aware extension of DDNM within
the library's SDE-based posterior sampling framework.

Following the soft-update formulation discussed in
*A Survey on Diffusion Models for Inverse Problems*
(Section 3.1.9), we consider the approximation

 D_t^y \appro D_t + lambda_t A^dagger (y - A D_t),

where D_t is the unconditional denoised estimate (by applying the denoiser) and D_t^y
approximates E[x_0 | x_t, y].

We choose

    lambda_t = sigma_t^2 / (sigma_t^2 + sigma_y^2),

which can be motivated by Gaussian conditioning for binary-mask
inpainting, assuming

    p(x_0 | x_t) \appro N(D_t, sigma_t^2 I).

Note that this choice presents a similar update to the score-ALD.
"""

# %%
# Posterior sampling and the intractable likelihood
# -------------------------------------------------
#
# We use a Variance-Exploding (VE) diffusion, whose scaling is
# :math:`s(t)=1`:
#
# .. math::
#
#     x_t = x_0 + \sigma_t\omega,
#     \qquad \omega\sim\mathcal N(0,\mathrm I).
#
# By Bayes' rule, the conditional score is
#
# .. math::
#
#     \nabla_{x_t}\log p_t(x_t\mid y)
#     = \nabla_{x_t}\log p_t(x_t)
#     + \nabla_{x_t}\log p_t(y\mid x_t).
#
# A diffusion denoiser :math:`D_\sigma` estimates the unconditional part using
# Tweedie's formula. The second term is harder because
#
# .. math::
#
#     p_t(y\mid x_t)
#     = \int p(y\mid x_0)p(x_0\mid x_t)\,\mathrm d x_0
#
# is generally intractable. The goal of the methods below is to replace
# :math:`p(x_0\mid x_t)` by a Gaussian approximation
#
# .. math::
#
#     p(x_0\mid x_t)
#     \approx \mathcal N\!\left(
#         x_0;D_{\sigma_t}(x_t),\Sigma_t(x_t)
#     \right).
#
# For a linear forward model and Gaussian measurement noise, the
# integral of the two Gaussian densities is available in closed form:
#
# .. math::
#
#     p_t(y\mid x_t)
#     \approx \mathcal N\!\left(
#         y;A D_{\sigma_t}(x_t),
#         A\Sigma_t(x_t) A^\top + \mathrm I
#     \right).
#
# The approximations differ primarily in their choice of
# :math:`\Sigma_t(x_t)`.
# DPS is the degenerate, zero-covariance case; PiGDM uses an isotropic
# covariance; and Moment Matching estimates a structured covariance with the
# second-order Tweedie formula. A
# :class:`deepinv.sampling.NoisyDataFidelity` approximates the gradient of the
# resulting negative log-likelihood. Consequently,
# :class:`deepinv.sampling.PosteriorDiffusion` subtracts the gradient returned
# by the data-fidelity object from the unconditional score.
#
# The word *noisy* here refers to the diffusion-corrupted variable
# :math:`x_t`; the measurements themselves do not have to be noisy.

# %%
# Create one inverse problem and one noisy diffusion state
# --------------------------------------------------------
#
# We use linear inpainting here. DPS also supports differentiable nonlinear
# forward operators, whereas the PiGDM and Moment Matching implementations
# below require a linear operator.

import matplotlib.pyplot as plt
import torch

import deepinv as dinv

device = dinv.utils.get_device()
dtype = torch.float32 if "mps" in str(device) else torch.float64

x_true = dinv.utils.load_example(
    "FFHQ_example.png", img_size=64, resize_mode="resize", device=device
)
mask = torch.ones_like(x_true)
mask[..., 24:40, 24:40] = 0.0
measurement_noise = 0.05
physics = dinv.physics.Inpainting(
    img_size=x_true.shape[1:],
    mask=mask,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=measurement_noise),
)
y = physics(x_true)

sigma_t = 0.15
rng = torch.Generator(device=device).manual_seed(0)
x_t = x_true + sigma_t * torch.randn(
    x_true.shape, generator=rng, device=device, dtype=x_true.dtype
)

# We use the same FFHQ NCSNpp denoiser for every approximation.
denoiser = dinv.models.NCSNpp(pretrained="download").to(device)
with torch.no_grad():
    x_0_denoised = denoiser(x_t, sigma_t)

dinv.utils.plot(
    {
        "Ground truth": x_true,
        "Measurement": y,
        r"Noisy $x_t$": x_t,
        r"Denoised $D_{\sigma_t}(x_t)$": x_0_denoised,
    },
    figsize=(12, 3),
)

# %%
# Score-ALD: guide with the noisy iterate itself
# ----------------------------------------------
ald = dinv.sampling.ALDDataFidelity(weight=1.0)
# ILVR: noise the measurements instead
# --------------------------------------------------
ilvr = dinv.sampling.ILVRDataFidelity(weight=1.0)

# %%
# DDNM: from hard projection to soft updates
# ------------------------------
# For noiseless measurements, DDNM uses the hard projection
#
#     D_t^y ≈ D_t + A^dagger (y - A D_t).
#
# When y contains measurement noise, enforcing A D_t^y = y
# can retain the noise in the reconstructed image.
#
# The survey suggests replacing the hard projection with
# a soft correction:
#
#     D_t^y ≈ D_t + Sigma_t A^dagger (y - A D_t).
#
# Here we choose \Sigma_t = \lambda_t I, with
#
#     \lambda_t = \sigma_t^2 / (\sigma_t^2 + \sigma_y^2).
#
# For binary-mask inpainting, this choice follows from
# Gaussian conditioning under the approximation
#
#     p(x_0 | x_t) ≈ N(D_t, sigma_t^2 I).
#
# Using the conditional Tweedie identity:
#
#     score_post ≈ (D_t^y - x_t) / sigma_t^2,
#
# or, equivalently, the negative measurement score is
#
#     g_DDNM = lambda_t A^dagger (A D_t - y) / sigma_t^2.
#
# DeepInverse implements the latter and combines it with
# the unconditional score in PosteriorDiffusion.
#
# ------------------------------
# This approximation modifies the conditional-mean estimate
# but does not reproduce the noise correction of the original
# discrete DDNM+ transitions.


ddnm = dinv.sampling.DDNMDataFidelity(
    denoiser=denoiser,
    weight=1.0,
)
# %%
# Compare the guidance terms
# --------------------------
#
# Calling ``grad`` is enough to compare the six approximations at the same
# :math:`x_t`. Their scales are method-dependent, so ``weight`` should be
# tuned separately in a reconstruction. The plot independently normalizes the
# magnitude of each gradient to emphasize its spatial structure.
#
# The two families look different:
#
# Score-ALD, Score-SDE and ILVR compare the *noisy* iterate to the measurements.
# Here :math:`x_t = x_0 + \sigma_t\omega` is built from the ground truth, so
#
# .. math::
#
#     A x_t - y = A(x_0 + \sigma_t\omega) - (A x_0 + \sigma_y\eta)
#               = \sigma_t A\omega - \sigma_y\eta :
#
# the signal cancels exactly, and the residual is *pure noise*. Their guidance
# maps therefore look like noise, and they vanish on the masked pixels, where
# :math:`A^\top` and :math:`A^\dagger` are zero.
#
# DPS, PiGDM and Moment Matching instead compare a *denoised* estimate to the
# measurements, so their residual reflects genuine reconstruction error rather
# than the noise in :math:`x_t`. Backpropagating it through the denoiser spreads
# the guidance over the whole image, including inside the mask.

data_fidelities = {
    "Score-ALD": ald,
    "ILVR": ilvr,
    "noise-aware DDNM": ddnm,
}

gradients = {}
for name, data_fidelity in data_fidelities.items():
    gradient = data_fidelity.grad(x_t.clone(), y=y, physics=physics, sigma=sigma_t)
    gradients[name] = gradient
    norm = torch.linalg.vector_norm(gradient).item()
    print(f"{name:>15s} gradient norm: {norm:.3e}")

dinv.utils.plot(
    {f"{name}": gradient.abs() for name, gradient in gradients.items()},
    figsize=(15, 3),
    suptitle="Gradient magnitude",
)

# %%
# Posterior sampling experiment
# -----------------------------
#
# The noisy data-fidelity object is one interchangeable component of
# :class:`deepinv.sampling.PosteriorDiffusion`. We now hold the denoiser, VE
# diffusion, Euler solver, measurements, and random seed fixed, and change only
# the noisy data-fidelity approximation. Using the same seed gives every method
# the same initial noise and Brownian increments.
#
# All six use ``weight=1``: every term is normalized by a guidance strength of
# order :math:`\sigma_y^2+\sigma_t^2`, so a single scale works for all of them.
#
# Moment Matching is considerably slower on CPU because every diffusion step
# contains an inner conjugate-gradient solve. On CPU, we therefore display a
# precomputed reconstruction; on GPU and MPS devices, we compute it normally.


num_steps = 100
timesteps = torch.linspace(
    1.0,
    0.001,
    num_steps,
    device=device,
    dtype=dtype,
)

sde = dinv.sampling.VarianceExplodingDiffusion(
    alpha=0.25,
    device=device,
    dtype=dtype,
)

clean_physics = dinv.physics.Inpainting(
    img_size=x_true.shape[1:],
    mask=mask,
    device=device,
    noise_model=dinv.physics.GaussianNoise(sigma=0.0),
)

# current physics has sigma_y = measurement_noise
# and is used by ALD / ILVR / noise-aware DDNM
methods = {
    "Score-ALD": {
        "data_fidelity": ald,
        "y": y,
        "physics": physics,
    },
    "ILVR": {
        "data_fidelity": ilvr,
        "y": y,
        "physics": physics,
    },
    "noise-aware DDNM": {
        "data_fidelity": ddnm,
        "y": y,
        "physics": physics,
    },
    "original DDNM": {
        "data_fidelity": ddnm,
        "y": y,                 # keep the same noisy observation
        "physics": clean_physics,  # but ignore sigma_y in the guidance
    },
}

posterior_samples = {}
# for name, data_fidelity in data_fidelities.items():
#     if name == "Moment Matching" and device.type == "cpu":
#         precomputed_sample = dinv.utils.load_url_image(
#             "https://huggingface.co/deepinv/demo/resolve/main/" "moment_matching.png",
#             device=device,
#             dtype=x_true.dtype,
#         )
#         posterior_samples[name] = precomputed_sample[:, : x_true.shape[1]]
#         continue
for name, cfg in methods.items():

    solver = dinv.sampling.EulerSolver(
        timesteps=timesteps,
        rng=torch.Generator(device=device),
    )
    posterior_sampler = dinv.sampling.PosteriorDiffusion(
        data_fidelity=cfg["data_fidelity"],
        denoiser=denoiser,
        sde=sde,
        solver=solver,
        device=device,
        dtype=dtype,
        verbose=False,
    )

    with torch.no_grad():
        posterior_samples[name] = posterior_sampler(
            y=cfg["y"],
            physics=cfg["physics"],
            seed=1,
            denoise_output=True,
        ).clip(0.0, 1.0)

# One row per family: the denoiser-free terms on top, the denoiser-based ones
# below, each preceded by its reference image.
rows = [
    {
        "Ground truth": x_true,
        "Measurement": y,
        
    },
    {
        **{
            name: posterior_samples[name] for name in ("Score-ALD", "ILVR",)
        },
    },
    {
        **{
            name: posterior_samples[name] for name in ("original DDNM",)
        },
        **{
            name: posterior_samples[name] for name in ( "noise-aware DDNM",)
        },
    }
]

# fig, axs = plt.subplots(
#     len(rows), 4, figsize=(12, 6.5), squeeze=False, layout="compressed"
# )
fig, axs = plt.subplots(
    len(rows), 2, figsize=(5, 8), squeeze=False, layout="compressed"
)
for row, images in zip(axs, rows, strict=True):
    n = len(images)
    dinv.utils.plot(images, fig=fig, axs=row[None, :], show=False)
    for ax in row[n:]: ax.set_visible(False)
plt.show()


# %%
# Check the cancellation of prior and measurement scores
# The posterior score can be directly computed from the conditional Tweedie equation.
# However, the library decomposes posterior score into prior score and measurement score, and 
# within DDNM, the computation of measurement score makes use of the prior score.
# The two ways should be equivalent, and we test it here in this cell.
#
#
# IMPORTANT:
# This verifies the score construction and implementation,
# not whether D_t^y accurately approximates the true
# conditional expectation E[x_0 | x_t, y].
probe = dinv.sampling.PosteriorDiffusion(
    data_fidelity=ddnm,
    denoiser=denoiser,
    sde=sde,
    solver=dinv.sampling.EulerSolver(
        timesteps=timesteps,
        rng=torch.Generator(device=device),
    ),
    device=device,
    dtype=dtype,
    verbose=False,
)

denoiser.eval()

t = torch.tensor(0.5, device=device, dtype=dtype)
sigma = sde.sigma_t(t)

rng = torch.Generator(device=device).manual_seed(123)
eps = torch.randn(
    x_true.shape,
    generator=rng,
    device=device,
    dtype=dtype,
)

# State in [-1, 1] coordinates
x = 2 * x_true.to(dtype) - 1 + sigma * eps

# State and noise level in [0, 1] coordinates
z = (x + 1) / 2
tau = sigma / 2

with torch.no_grad():
    # Denoiser output used by the DDNM data fidelity
    grad_z, D_data = ddnm.grad(
        z,
        y,
        physics=physics,
        sigma=tau,
        get_model_outputs=True,
    )

    D_data = D_data.to(dtype)

    # Denoiser output used by the unconditional SDE score
    D_prior = (
        probe.sde.denoiser(
            x.float(),
            sigma.float(),
        ).to(dtype) + 1
    ) / 2

    # Prior score from the library
    s_prior = probe.sde.score(x, t)

    # Prior score reconstructed from the DDNM denoiser
    s_prior_expected = (2 * D_data - 1 - x) / sigma**2

    # Posterior score from the full DeepInverse call chain
    s_post = probe.score(y, physics, x, t)

    # Direct soft conditional-mean approximation
    sigma_y = physics.noise_model.sigma
    lambda_t = tau**2 / (tau**2 + sigma_y**2)

    D_cond = (
        D_data
        + lambda_t
        * physics.A_dagger(y - physics.A(D_data))
    )

    # Conditional Tweedie, without Bayes decomposition
    s_post_direct = (2 * D_cond - 1 - x) / sigma**2


def compare(name, a, b):
    diff = (a - b).norm()
    relative = diff / (b.norm() + 1e-12)
    print(f"{name:25s} relative error = {relative.item():.3e}")


print(
    "Same underlying denoiser:",
    getattr(probe.sde.denoiser, "model", None) is ddnm.denoiser,
)

compare("Denoiser outputs", D_prior, D_data)
compare("Prior score", s_prior, s_prior_expected)
compare("Posterior score", s_post, s_post_direct)

# %%
print("Noisy physics sigma:", physics.noise_model.sigma)
print("Clean physics sigma:", clean_physics.noise_model.sigma)

for name, cfg in methods.items():
    print(
        name,
        "sigma_y =", cfg["physics"].noise_model.sigma,
        "same physics =", cfg["physics"] is physics,
        "same y =", cfg["y"] is y,
    )
# %%
# comment:
# - ALD and ILVR produce visibly cleaner reconstructions.
# - Hard DDNM-style guidance retains substantial
#   measurement noise.
# - The current soft-update DDNM guidance also retains
#   noticeable measurement noise.
# - The numerical posterior-score consistency check
#   confirms the expected Tweedie/Bayes decomposition.
#
# Two hypotheses for investigating the failure:
# 1. the conditional mean approximation D_t^y is not sufficiently accurate.
# 2. Continuous SDE sampling with the approximate posterior score may be different
# than the DDNM+ in the original paper which employs discrete reverse transitions and adjust transition noise.
# 
# This experiment should not be interpreted as DDNM+ fails to be fit in the SDE sampling framework.
# %%
# :References:
#
# .. footbibliography::
