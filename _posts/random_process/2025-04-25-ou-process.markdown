---
layout: post
title:  "Ornstein-Uhlenbeck (OU) Process"
date:   2025-04-19 10:57:10 +0800
categories: random-process
---

Let $X_t$ be a random variable at the timestamp $t$,
$W_t$ be Brownian motion, and $\theta>0$ be convergence speed.

Standard Ornstein-Uhlenbeck (OU) process in differential form is

$$
dX_t=\theta(\mu-X_t)dt+\sigma dW_t
$$

where $\mu$ is the long term mean and $\sigma$ is the fluctuation rate (NOT the same as $X_t$'s variation).

## Prerequisite: Ito's Lemma （伊藤引理） and Process

Ito process formulates stochastic process as

* an overall drift/trend described by $\mu$
* variation described by $\sigma$ as Wiener process progresses

Let $X_t$ be a random variable that updates per Ito process, the differential expression is

$$
dX_t=\mu(X_t, t)dt+\sigma(X_t, t)dW_t
$$

where

* $W_t$ is Brownian motion
* $\sigma(X_t, t)$ is a diffusion term
* $\mu(X_t, t)$ is a drift term

Let $f(X_t, t)$ is (at least) a second-order differentiable $f\in\mathbb{C}^2$, then

1. Start with a Taylor expansion

$$
\begin{align*}
    df=\frac{\partial f}{\partial t}dt+\frac{\partial f}{\partial X}dX_t +
    \frac{1}{2}\frac{\partial^2 f}{\partial X^2}(dX_t)^2+...
\end{align*}
$$

2. Replace $dX_t$ with its Ito process definition

$$
\begin{align*}
dX_t&=\mu dt + \sigma dW_t \\
(dX_t)^2&=\mu^2 dt^2 + \sigma^2 (dW_t)^2 + 2\mu\sigma dt dW_t
\end{align*}
$$

3. Approximation as $dt\rightarrow 0$

$dt^2$ and $dt dW_t$ vanishes as $dt\rightarrow 0$.
By quadratic variation of Brownian motion, there is $(dW_t)^2\approx \sigma^2 dt$.

Substitute back into the Taylor expansion, Ito process can be expressed as

$$
df=\Big(\frac{\partial f}{\partial t}+\mu\frac{\partial f}{\partial X}+\frac{1}{2}\frac{\partial^2 f}{\partial X^2}\sigma^2\Big)dt+
\sigma \frac{\partial f}{\partial X} dW_t
$$

### Ito's Isometry

Ito's Isometry gives the expectation of random differentials over a time range $[0,T]$ if the random process is a Brownian motion $W_t$.

If $E\left[\int^T_{0} g(t)^2 dt \right]<\infty$, let $W_t$ be Brownian motion, Ito's Isometry is

$$
E\left[\Big(\int^T_{0} g(t)dW_t\Big)^2 \right]=
E\left[\int^T_{0} g(t)^2 dt \right]
$$

Assume $g(t)$ be a step function such that $g(t)=\sum^n_{i=1} g_i \cdot I_{(t_{i-1},t_i]}(t)$, where $I$ is a identity matrix.

$$
\int^T_0 g(t) dW_t = \sum^n_{i=1} g_i (W_{t_i}-W_{t_{i-1}})
$$

The expectation is

$$
E\left[\Big(\sum^n_{i=1} g_i \cdot \Delta W_i \Big)^2\right]=
\sum^n_{i=1} g_i^2 \cdot E\left[\Big(\Delta W_i \Big)^2\right]=
\sum^n_{i=1} g_i^2 \cdot \Delta t_i
$$

## Ornstein-Uhlenbeck Process Derivation

To find the analytic solution, define $f(X_t, t)=X_t e^{\theta t}$ at the initial state/moment $t$, to find how OU progresses as $t\rightarrow t+\tau$, according to Ito's Lemma, here formulates the OU process

$$
\begin{align*}
df&=\Big(\frac{\partial f}{\partial t}+\mu\frac{\partial f}{\partial X}+\frac{1}{2}\frac{\partial^2 f}{\partial X^2}\sigma^2\Big)dt+
\sigma \frac{\partial f}{\partial t} dW_t \\
&=\Big(\frac{\partial f}{\partial t}+\theta(\mu-X_t)\frac{\partial f}{\partial X}+\frac{1}{2}\frac{\partial^2 f}{\partial X^2}\sigma^2\Big)dt+
\sigma \frac{\partial f}{\partial t} dW_t \\
&=\Big(\theta X_t e^{\theta t}+\theta(\mu-X_t)e^{\theta t}+\frac{1}{2}\cdot 0 \cdot\sigma^2\Big)dt+
\sigma e^{\theta t} dW_t \\
&=\theta\mu e^{\theta t} + \sigma e^{\theta t} dW_t
\end{align*}
$$

Integrate $df$ over the time range $[t, t+\tau]$, there is

$$
\begin{align*}
&& X_{t+\tau}e^{\theta (t+\tau)}-X_{t}e^{\theta t}&=
\int^{t+\tau}_{t}\theta\mu e^{\theta s} ds+\int^{t+\tau}_{t} \sigma e^{\theta s} dW_s \\
&& &= \mu e^{\theta s}\Big|^{s=t+\tau}_{s=t}+\sigma\int^{t+\tau}_{t}e^{\theta s} dW_s \\
\Rightarrow && X_{t+\tau}e^{\theta (t+\tau)} &= \mu e^{\theta s}\Big|^{s=t+\tau}_{s=t}+\sigma\int^{t+\tau}_{t}e^{\theta s} dW_s + X_{t}e^{\theta t} \\
&& &= \mu e^{\theta t} \big(e^{\theta \tau}-1\big) + X_{t}e^{\theta t}+\sigma\int^{t+\tau}_{t}e^{\theta s}dW_s \\
\Rightarrow && X_{t+\tau} &= \mu \big(1-e^{-\theta \tau}\big) + X_{t}e^{-\theta t}+\sigma\int^{t+\tau}_{t}\big(e^{\theta s - \theta (t+\tau)}\big)dW_s \\
&& &=\mu \big(1-e^{-\theta \tau}\big) + X_{t}e^{-\theta t}+
\underbrace{\sigma\int^{t+\tau}_{t}\big(e^{-\theta (t+\tau-s)}\big)dW_s}_{\sim N(0, \sigma^2\int^{t+\tau}_{t}e^{-2\theta (t+\tau)-s}ds)}
\end{align*}
$$

By Ito's isometry,

$$
E\left[\Big(\sigma\int^{t+\tau}_{t}\big(e^{-\theta (t+\tau-s)}\big)dW_s\Big)^2\right]=
\sigma^2\int^{t+\tau}_{t}e^{-2\theta (t+\tau)-s}ds=
\frac{\sigma^2}{2\theta}\big(1-e^{-2\theta\tau}\big)
$$

Rewrite the formula that integrates over $[0,t]$, and let $X_0$ be the initial state, then

$$
X_t=\mu \big(1-e^{-\theta t}\big)+X_{0}e^{-\theta t}+
\sigma\int^{t}_{0}\big(e^{-\theta (t-s)}\big)dW_s
$$
