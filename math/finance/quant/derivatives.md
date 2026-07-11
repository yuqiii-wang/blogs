# Derivative

A type of financial contract whose value is dependent on an underlying asset, group of assets, or benchmark.

The most common underlying assets for derivatives are stocks, bonds, commodities, currencies, interest rates, and market indexes. Contract values depend on changes in the prices of the underlying asset.

Some common financial contracts are futures (an underlying asset traded on a specific date and contracts itself can be traded), options (right to buy/sell an underlying asset)

## Swap

A swap is a derivative contract through which two parties exchange the cash flows or liabilities from two different financial instruments. The two product to be swapped are not necessary the same underlying asset type.

For example, Party A regards the price of one security *SB* rising and currently holds another security *SA* expecting going down; Party B thinks of the opposite and currently holds *SB*. They can sign a swap contract for speculation/risk hedging.

### CDS (Credit Default Swap)

A credit default swap (CDS) is a financial swap agreement that the seller of the CDS will compensate the buyer in the event of a debt default (by the debtor) or other credit event.

In other words, it is an insurance to an event. In 2008 financial crisis, people buy CDS from banks to insure the event that some real estate companies would not bankrupt. People paid premiums to banks, and banks repaid a huge sum of money as compensation to people when those real estate companies saw credit defaults.

## Futures

Derivative financial contracts that obligate the parties to transact an asset at a predetermined future date and price.

* Parties can `long`/`short` derivatives, a typical work flow such as

1) Given a spot price of $1,000 for one unit of a derivative contract at a time

2) Parties expecting a higher close price can buy (bought the derivative, spent \$1,000, paid to the selling party, who received \$1,000 per unit) at the spot price (called `long`), while the opposite expecting lower can sell (called `short`). Here assumes both buy/sell one unit of the derivative.

3) There is no actual purchase action on the commodity stipulated on the derivative contract, just two opposite parties make order to each other. The selling party does not need to possess (not already purchased) any unit of the derivative-defined commodity. The selling party is just labelled `-1` to its account book, accordingly, the buy side's account book receives `+1`.

4) As time goes by, the spot price of the derivative goes up and down. Parties can trade before the clearing date, or not to.

5) On the date of clearing, the spot goods (the derivative contract defined commodity) must be purchased/sold by the parties.

6) If the spot price of the clearing date is lower than \$1,000, for example, \$300. The selling party must prepare one unit of the actual derivative-defined commodity by spending \$300 to purchase one, and the buying party must pay $300 to the selling party and receives the commodity.

7) The selling party buys the actual physical commodity from a third party (usually through an exchange broker), since the selling party does not reserve any physical commodity, just wants to speculate. The buying party once receives the physical commodity, can immediately sell it to another third party at the same spot price of $300, since it has no interest in preserving the commodity, just wants to speculate as well. The exchange of commodity does not even happen sometimes, just two parties speculating against each other.

8) Hence, the selling party (the short) earned \$700 ($1,000 - $300) as profit, while the buying party (the long) lost \$700 accordingly. The logic holds truth if the clearing date price went up, that indicates loss for the short and win for the long.

## Options (期权)

An option starts a contract that allows the holder the right to buy or sell an underlying asset or financial instrument at a specified strike price on or before a specified date.

* *Call options* allow the holder to **buy** the asset at a stated price within a specific timeframe.
* *Put options*, on the other hand, allow the holder to **sell** the asset at a stated price within a specific timeframe.

Example: Daily Oil investment, where people pays a premium to obtain a right to buy/sell oil at a stated price within a day.

### Vanilla Option Trading

* Buy Call (买涨期权) and Sell Call（卖涨期权）

For example, a Call option buyer pays premium \$0.3 per share to a Call option seller that, on a specified date, say in 2 weeks, that the buyer can demand buying 100 shares from the seller with a share price anchored on today.

If the share price goes up in 2 weeks, say will have risen by \$3.4 per share, the buyer can profit from the price growth \$340 with only having paid \$33 to the option seller.
The option seller, despite having earned a premium \$33, loses \$307 as the seller needs to buy 100 shares with the 2-week-future price.

However, if the share price goes down, the buyer does not need to exercise the privilege to demand the option seller to give him the 100 shares.

* Buy Put（买跌期权）and Sell Put（卖跌期权）

For example, a Put option buyer pays premium \$0.3 per share to a Put option seller to obtain the right that he can force selling 100 shares in 2 weeks to the option seller with a share price anchored on today.

If the share price goes down, the option buyer can exercise his right to force selling the 100 shares to the option seller with the price at the 2 weeks ago's.
If the price goes up, the option buyer does not need to take action.

### Quantitative Analysis

Black Sholes Formula

### Snowball Structure Income Certificates/Barrier Option (雪球结构收益凭证/障碍期权)

A barrier option starts an option whose payoff is conditional upon the underlying asset's price breaching a barrier level during the option's lifetime.

* *Knock-Out* (KO，敲出) options are options that expire worthless if a specified price level in the underlying asset is reached.

* *Knock-In* (KI，敲入) options are options that only come into existence if the pre-specified barrier level is crossed by the underlying asset's price.

*Snowball Structure Income Certificate* is a marketing term that describes profit growth comparable to a snowball getting bigger as it rolls.

#### Profit and Loss

Essentially, a Snowball Structure Income Certificate is a Put option + insurance.
A client provides loan to an investment bank earning interest/premium so that the investment bank can invest in some securities, and the client sells insurance to the investment bank to help the investment bank hedge risks of the anchored underlying securities' prices dropping too much, as the client burdens the loss.

* Bull Market: both clients and investment banks are winners

Clients earn interest/premium; investment banks with clients money earn from investing in the bull markets

* Flat: clients are winners, investment banks are losers with trivial losses of interest/premium

Clients earn interest/premium nevertheless; investment banks pay interest/premium to clients, but earn little from a market with little volatility.

* Bear Market: Clients are losers, investment banks are winners

Clients can only claim back a portion of its principle; investment banks do not need to pay interest/premium to clients, but have exploited market opportunities during the valid option period.

#### Example:

|Items|Specifications|
|-|-|
|Anchor/Underlying|CSI500 (中证500)|
|Option Valid Period|Max 24 months, or early termination when knock-out/knock-in event happens|
|Knock-in/Know-out Event Observation Frequency|Everyday|
|knock-out Interest/Premium|15% annualized|
|Knock-out Price|The underlying spot price when option starts valid $\times 1.1$|
|Knock-in Price|The underlying spot price when option starts valid $\times 0.8$|
|Knock-out Event|The underlying spot price $\ge$ Knock-out price|
|Knock-in Event|The underlying spot price $\le$ Knock-in price|
|Implications After Knock-out Event|Option invalidated; clients claim back the principle (same as the underlying spot price when option starts valid) + interest/premium|
|Implications After Knock-in Event|Option invalidated; clients claim back the principle worth the underlying spot price on the option termination date|
|Example: Client Interest Given A Triggered Knock-out Event|Given underlying start price $\times 1.1$ as the knock-out price, when CSI500 rises by $11\%$ in the eighth month, knock-out event gets triggered, and the client can claim back all of the principle + the interest/premium worth of principle $\times \frac{8}{12} \times 0.18$.|
|Example: Client Interest Given A Triggered Knock-in Event|Given underlying start price $\times 0.8$, when CSI500 drops by $21\%$, knock-in event gets triggered, and the client can only claim back $79\%$ of the principle.|


https://www.hanspub.org/journal/PaperInformation?paperID=55528

## Forward

A forward contract or simply a forward is a non-standardized contract between two parties to buy or sell an asset at a specified future time at a price agreed on at the time of conclusion of the contract.

### Forward price

Forward price is the price at which a seller delivers an underlying asset, financial derivative, or currency to the buyer of a forward contract at a predetermined date.

It is roughly equal to the spot price plus associated carrying costs such as storage costs, interest rates, etc.

### Forward vs Futures

Forward contracts are not exchange-traded, or defined on standardized assets.

Forwards typically have no interim partial settlements or "true-ups" in margin requirements like futures, that is the parties do not exchange additional property securing the party at gain and the entire unrealized gain or loss builds up while the contract is open.

Hence, forward is not open to retail investors. 
