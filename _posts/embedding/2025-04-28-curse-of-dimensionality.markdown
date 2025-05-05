---
layout: post
title:  "Curse of Dimensionality"
date:   2025-02-08 00:57:10 +0800
categories: embedding
---

When num of dimensions goes large, it is generally hard to train and vector representation is sparse.

## Random Unit Vector Inner Product Orthogonality

Let $\bold{x}, \bold{y}\in\mathbb{R}^n$ constrained on $\|\|\bold{x}\|\|=\|\|\bold{y}\|\|=1$ be random unit vectors, here to show

$$
\bold{x}\cdot\bold{y} \rightarrow 0, \quad\text{as } n\rightarrow\infty
$$

Assume each dimension of $\bold{x}, \bold{y}$ follows Gaussian distribution

$$
\begin{align*}
\bold{x}&=\frac{\bold{g}}{||\bold{g}||}, \text{ where } \bold{g}=[g_1, g_2, ..., g_n] \text{ with } g_i\sim\mathcal{N}(0,1) \\
\bold{y}&=\frac{\bold{h}}{||\bold{h}||}, \text{ where } \bold{h}=[h_1, h_2, ..., h_n] \text{ with } h_i\sim\mathcal{N}(0,1)
\end{align*}
$$

So that, the inner product can be written as

$$
\bold{x}\cdot\bold{y}=
\frac{\bold{g}\cdot\bold{h}}{||\bold{g}||\space||\bold{h}||}
$$

where the numerator and denominator follow

* $\bold{g}\cdot\bold{h}=\sum^n_{i=1}g_i h_i$. Since $g_i, h_i\sim\mathcal{N}(0,1)$, and the product of two standard normal distribution random variables also follows a standard normal distribution, i.e., $g_i h_i\sim\mathcal{N}(0,1)$, by the Central Limit Theorem (CLT) (Let $X_i$ be a random variable; as sample size $n$ gets larger, there is ${\sqrt{n}}({\overline{X}}_{n}-\mu) \rightarrow \mathcal{N}(0, \sigma^2)$), there is $\bold{g}\cdot\bold{h}\sim\mathcal{N}(0,n)$
* The Law of Large Numbers states that $\|\|\bold{g}\|\|$ and $\|\|\bold{h}\|\|$ approach their truth means as $n\rightarrow\infty$; the truth means are $\|\|\bold{g}\|\|\approx \sqrt{n}$ and $\|\|\bold{h}\|\|\approx \sqrt{n}$

As a result for a very large $n\rightarrow\infty$,
the inner product goes to $\rightarrow \mathcal{N}(0,0)$, therefore totally orthogonal.

$$
\bold{x}\cdot\bold{y}\approx\frac{\mathcal{N}(0,n)}{n}=
\mathcal{N}(0,\frac{1}{n})
$$

## High Dimensionality Geometric Explanation by HyperSphere

A hypersphere in $n$-dimensional space is defined by all points at a fixed radius $R$ from the origin $\bold{0}$.
Denote its volume as $V_n(R)$ and surface area as $S_n(R)$:

$$
\begin{align*}
    V_n(R) &=\frac{\pi^{n/2}}{\Gamma(\frac{n}{2}+1)}R^n \\
    S_n(R) &=\frac{d}{dR}V_n(R)=\frac{2\pi^{n/2}}{\Gamma(\frac{n}{2})}R^{n-1}
\end{align*}
$$

where $\Gamma(z)$ is gamma function such that

* $\Gamma(k)=(k-1)!$ for $k\in\mathbb{Z}^+$
* $\Gamma(1/2)=\sqrt{\pi}$

$\bold{x}, \bold{y}$ can be said be drawn from the surface of a unit hypersphere $S_{n}(1)$.
For $\bold{x}\cdot\bold{y}\sim\mathcal{N}(0,\frac{1}{n})$, as $n\rightarrow\infty$, the two vectors $\bold{x}, \bold{y}$ are likely irrelevant/orthogonal.

The surface area of a hypersphere scales as $S_{n}(1)\propto n^{n/2}$.
Consequently, the number of points needed to "cover" the surface grows exponentially.

## Example of HyperSphere Surface Point Sampling and Density/Sparsity Intuition

Define two vector $\bold{x}, \bold{y}$ that are dense/relevant/close to each other if they happen to fall in the same $\pi/2$ segment of a hypersphere (for $\text{cos}(\bold{x}, \bold{y})\in[0, 1]$ it can be said that the two vectors are positively related).

* For $n=2$ (a circle), there are $4$ segments
* For $n=3$ (a sphere), there are $8$ segments
* For $n=4$ (a hyper-sphere), there are $16$ segments

For $n$ is very large such as $n=10000$, there are $2^{10000}$ vectors from which on average only one vector is considered close to a vector existed in an arbitrary $\pi/2$ hypersphere segment.
It is impractical to collect such a large size sample hence the sample feature space is sparse.