# Repurchase Agreement (REPO) Risks

Broadly speaking, there are two types of risk

* Counterparty / Credit Risk
  * Counterparty risk is risk of default due to financial difficulty or withdrawal from business
    Banks internally rate all counterparties and assign exposure limits to each, by firm and sector
* Market Risk
  * Collateral risk / Issuer risk
    Risk exposure from changes in market levels, interest rates, asset values, etc.
    One reason for continuing popularity in stock lending
  * FX risk
    Cross-currency repo, or a stock loan collateralized with assets denominated in a different currency

## Trading and Hedging Strategies

* Yield Curve Arbitrage

Anticipated that some securities' price might go up; some might go down.
Buy both so that future yield would be neutral to avoid drastic rises or falls in values.

* Credit Intermediation

Gov bonds usually have high credibility.
They can be used to hedge risks.

Typically, use SOFR (for USD treasury bond) or LIBOR (for British treasury bond) as the benchmark to measure the risk of a bond.
A haircut/spread/added floating rate can be applied on top of SOFR or LIBOR as the risk hedging quantified strategy.

* Matched Book Trading

Make sure underlying securities have its value as stated by daily monitoring.

Counterparties are monitored as well.

Consider *Prime*: prime bonds/trades/clients are private agreements that establish good trust-worthy relationships with the counterparties.
The prime agreements offer good repo rate to counterparties.

* Derivative Market Anticipation

Demand for short-term cash is often correlated to the derivative market.
High volatility market means that people are rush to raise funds to short/long derivatives.

## Market Risk: Discount Security Risk Analysis by NPV

*Present value* (PV) is defined as

$$
\begin{align*}
    & \text{exponential form }
    && PV = \frac{C_n}{(1+r)^n} \\\\
    & \text{linear form }
    && PV = \frac{C_n}{1+rn}
\end{align*}
$$

where

* $C_n$ cash flow at the $t$ time
* $r$ discount rate/IRR (Internal Risk of Return)
* $n$ the number of time periods

*Net Present Value* (NPV) is computed by measuring the difference between present cash (PC) flow and cost of cash (CC) over $N$ periods.

$$
\begin{align*}
    & \text{exponential form }
    && \text{NPV} \sum_{n=1}^{N} \frac{PC_n}{(1+r)^n} - \sum_{n=1}^{N} \frac{CC_n}{(1+r)^n} \\\\
    & \text{linear form }
    && \text{NPV} \sum_{n=1}^{N} \frac{PC_n}{1+rn} - \sum_{n=1}^{N} \frac{CC_n}{1+rn} \\\\
\end{align*}
$$

The NPV says if $NPV>0$, there is surplus in cash flow (profit); if $NPV<0$, there is loss in cash flow.

NPV can be used as a risk indicator before a security (typically bonds) mature.

In practice, rather than using one NPV targeting one particular date, risks are mapped to the maturity date's nearest tenor.
Only close-to-yield/mature securities should receive high attention; for securities' yield/maturity dates far in the future, the attention granularity level is month/year, not days.

| Tenor$n$                        | Rate          |
| --------------------------------- | ------------- |
| $\text{o}/\text{n}$ (overnight) | $r_0$       |
| $\text{t}/\text{n}$ (tomorrow)  | $r_1$       |
| 2 days                            | $r_2$       |
| 5 days                            | $r_5$       |
| 1 week                            | $r_7$       |
| 2 weeks                           | $r_{14}$    |
| ...                               | ...           |
| 30 years                          | $r_{10950}$ |

where

* $n$: grid point of period with $n \in \{ 0, 1, 2, 5, 7, ..., 10950 \}$
* $r_n$ interest rate of instrument for period $n$

Set $t$ as the remaining days to maturity, $t_{\text{lower}}$ and $t_{\text{upper}}$ as the lower nearest tenor and upper nearest tenor, e.g., $t=10$ sees $t_{\text{lower}}=n_{\text{1week}}$ and $t_{\text{upper}}=n_{\text{2week}}$.

Define a ratio $\gamma_{\text{lower}}$ and $\gamma_{\text{upper}}=1-\gamma_{\text{lower}}$ that splits NPV as risk to

$$
\gamma_{\text{lower}} =
\frac{t_{\text{lower}}(t_{\text{upper}}-t)}{t(t_{\text{upper}}-t_{\text{lower}})}
$$

So that, the final tenor-mapped NPVs are
$\text{NPV}_{\text{lower}}=\gamma_{\text{lower}} \text{NPV}$ and $\text{NPV}_{\text{upper}}=\gamma_{\text{upper}} \text{NPV}$.

### Example

For example, a one-year bond's spot price is \$102.28, and this bond has 10 days to mature paying \$100 + \$2.3 (assumed annual payment).

The present value of the bond is \$ $102.24 = 100 + 2.3  \times \frac{355}{365}$.
The NPV is \$ $-0.04 = 102.24 - 102.28$ that indicates possible loss of \$0.04 per \$100 when this bond matures.
$\text{NPV} < 0$ indicates this bond is likely over-priced. A further explanation can be that market is in turmoil and money seeks refuge by investing in good quality bonds.

To map the NPV risk to its nearest tenors, there is

$$
\begin{align*}
\gamma_{\text{lower}} &=
\frac{t_{\text{lower}}(t_{\text{upper}}-t)}{t(t_{\text{upper}}-t_{\text{lower}})} =
\frac{7 \times (14 - 10)}{10 \times (14 - 7)} = 0.4 \\\\
\gamma_{\text{upper}} &= 1 - \gamma_{\text{lower}} = 0.6
\end{align*}
$$

## Bond-Based NPV and Risks

In bonds, $\text{SecurityPrice}$ is used as the spot price of the bond (present value).

$$
\text{SecurityPrice} =
\text{SecurityMidPrice} \times \text{CurrencyExchange} \times \text{IssueFactor}
$$

where $\text{SecurityMidPrice} = \frac{1}{2}(\text{SecurityAskPrice} + \text{SecurityBidPrice})$.
The $\text{IssueFactor}$ refers to partial purchase of the total amount of the issued bonds.
$\text{CurrencyExchange}$ is often used to convert bond value to US dollars.

$\text{StartCash}$ represents cost of cash.

$$
\text{StartCash} =
(\text{SettlementPrice} / 100) \times \text{Quantity} \times \text{HaircutRatio}
$$

where $\text{SettlementPrice}$ is a bond face value (typical \$100 per coupon).

The NPV as risk is $\frac{1}{(1+r)^n}\text{SecurityPrice}-\text{StartCash}$, where $r$ is coupon rate and $n$ is the total number of coupon yields.

### REPO Tenor vs Bond Maturity Risk Matrix

Generally speaking, REPO risk increases as

* a bond borrowing and lending (REPO) period is longer
* a bond has long maturity time
* a bond type, e.g., government, corporate, part of collateral in third party custody, etc.
* Currency represented country is economy-healthy
* reputed institution ratings, e.g., S&P

For example, a group of financial experts come up with this determination of risk weights.
They might review it quarterly to update the risk weights with examined market conditions, e.g., following up central bank policies.

In the example below, US Treasure bond REPO by USD purchase has the lowest risk, while lower-quality bond REPO trading in non-USD currency has much higher risks.

| Currency | Bond Type | Bond Ratings, e.g., S&P | Days to REPO End Date | Days to Bond Maturity | Risk |
| -------- | --------- | ----------------------- | --------------------- | --------------------- | ---- |
| USD      | GOV       | AAA                     | Tomorrow              | Tomorrow              | 0.01 |
| USD      | CORP      | AAA                     | Tomorrow              | Tomorrow              | 0.02 |
| USD      | CORP      | AA+                     | Tomorrow              | Tomorrow              | 0.05 |
| ...      | ...       | ...                     | ...                   | ...                   | ...  |
| USD      | CORP      | AA-                     | 1 month               | 1 year                | 17   |
| USD      | CORP      | A+                      | 1 month               | 1 year                | 23   |
| ...      | ...       | ...                     | ...                   | ...                   | ...  |
| USD      | CORP      | BBB-                    | 1 month               | 1 year                | 52   |
| USD      | CORP      | BBB-                    | 2 months              | 1 year                | 63   |
| ...      | ...       | ...                     | ...                   | ...                   | ...  |
| USD      | CORP      | BBB-                    | 1 month               | 3 years               | 55   |
| USD      | CORP      | BBB-                    | 1 month               | 7 years               | 59   |
| ...      | ...       | ...                     | ...                   | ...                   | ...  |
| CNY      | GOV       | AAA                     | Tomorrow              | Tomorrow              | 0.02 |
| CNY      | CORP      | AAA                     | Tomorrow              | Tomorrow              | 0.03 |
| ...      | ...       | ...                     | ...                   | ...                   | ...  |
| CNY      | CORP      | BBB-                    | 1 month               | 3 years               | 57   |
| CNY      | CORP      | BBB-                    | 1 month               | 7 years               | 61   |
| ...      | ...       | ...                     | ...                   | ...                   | ...  |
| VND      | CORP      | BBB-                    | 1 month               | 3 years               | 138  |
| VND      | CORP      | BBB-                    | 1 month               | 7 years               | 145  |
| VND      | CORP      | BBB-                    | 2 month               | 3 years               | 149  |

where

|              | S&P  | Moody's |
| ------------ | ---- | ------- |
| Top Quality  | AAA  | Aaa     |
| High Quality | AA+  | Aa1     |
|              | AA   | Aa2     |
|              | AA-  | Aa1     |
| Upper Medium | A+   | A1      |
|              | A    | A2      |
|              | A-   | A1      |
| Medium       | BBB+ | Baa1    |
|              | BBB  | Baa2    |
|              | BBB- | Baa3    |

## Counterparty Risk: Counterparty Exposure

If a counterparty sees default on one security (e.g., bond), this counterparty might be insolvent at this moment, and all trades with this counterparty are at risk.

The total risk exposure to a counterparty can be computed by simply summing up all PVs of all assets traded with this counterparty.

### Repo Trade Exposure

$$
\begin{align*}
\text{Exposure}&=\underbrace{\text{BondMarketValue}-\text{CollateralAmount}-\text{REPOAccruedInterest}+\text{BondAccruedInterest}}_{MTM} \\
&+\text{BondMarketValue}\times\text{FluctuationWeight}
\end{align*}
$$

### Weighted PVs as Counterparty Risks

NPVs of different trades can be weighted then summed.
Weighted NPVs consider the below factors.

* Exist An Agreement with the Counterparty

An agreement lists what action to take to hedge default risks (e.g., provide proof of possession of immovable valuables such as real estate).
An agreement states that what collateral this trade's underlying security is based on, or purely by credit.

One manually set up weights per trade per different agreements.
