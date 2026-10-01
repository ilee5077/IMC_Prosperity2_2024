# IMC Prosperity 2
in April 2024, I participated in IMC trading's Prosperity 2, a global algorithmic and manual trading competition for the first time.

Prosperity 2 is a worldwide trading competition that runs for 15 days, with 5 rounds of 3 days. Each round, new products are introduced and participants develop and refine their trading algorithms while also solving a separate manual trading challenge.

I participated as a solo team, and finished 199th out of approximately 10,000 teams. (team name: Koreant)

## Round results

<table>
    <thead>
        <tr>
            <th colspan="4" style="text-align: center">Profit / loss</th>
            <th colspan="2" style="text-align: center">Leaderboard position</th>
            <th colspan="2" style="text-align: center">Visualizer links</th>
        </tr>
        <tr>
            <th>Overall</th>
            <th>Manual</th>
            <th>Algo</th>
            <th>Round</th>
            <th>Overall</th>
            <th>OverallΔ</th>
            <th>Submission</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>122,043</td>
            <td>92,367</td>
            <td>29,676</td>
            <td>122,043</td>
            <td>615</td>
            <td></td>
            <td></td>
        </tr>
        <tr>
            <td>283,709</td>
            <td>113,938</td>
            <td>47,727</td>
            <td>161,665</td>
            <td>370</td>
            <td>+245</td>
            <td></td>
        </tr>
        <tr>
            <td>453,483</td>
            <td>61,314</td>
            <td>108,460</td>
            <td>169,774</td>
            <td>314</td>
            <td>+56</td>
            <td><a href="https://jmerle.github.io/imc-prosperity-2-visualizer/?open=https://raw.githubusercontent.com/ilee5077/IMC_Prosperity2_2024/main/logs/round3_result.log">link</a></td>
        </tr>
        <tr>
            <td>816,823</td>
            <td>99,152</td>
            <td>264,188</td>
            <td>363,340</td>
            <td>227</td>
            <td>+87</td>
            <td><a href="https://jmerle.github.io/imc-prosperity-2-visualizer/?open=https://raw.githubusercontent.com/ilee5077/IMC_Prosperity2_2024/main/logs/round4_result.log">link</a></td>
        </tr>
        <tr>
            <td>1,042,480</td>
            <td>58,210</td>
            <td>167,446</td>
            <td>225,656</td>
            <td>199</td>
            <td>+28</td>
            <td><a href="https://jmerle.github.io/imc-prosperity-2-visualizer/?open=https://raw.githubusercontent.com/ilee5077/IMC_Prosperity2_2024/main/logs/round5_result.log">link</a></td>
        </tr>
    </tbody>
</table>

## Round summaries
### Round 1
Two tradable products, STARFRUIT and AMETHYSTS, were introduced.

For AMETHYSTS, the price fluctuated within ±3 of a mean price of 10,000. I used a combination of market taking and market making. For market taking, I traded more aggressively as the market price deviated further from the 10,000 mean, taking advantage of the expected reversion towards fair value. For market making, I continuously placed orders one tick better than the best bid or ask, while avoiding trades that crossed the 10,000 fair-value level.

For STARFRUIT, I found that a five-timestamp moving average provided a useful short-term estimate of the next price. I used this forecast as an estimate of fair value and traded when the market price appeared under- or over-valued relative to the forecast.

For the manual trading challenge, I was given two opportunities to bid for SCUBA_GEAR from a group of goldfish, each with an individual reserve price between 900 and 1,000 SeaShells. The reserve-price distribution increased linearly towards 1,000, and each goldfish would accept the bid if it met or exceeded its reserve price, and any SCUBA_GEAR acquired could subsequently be resold for 1,000 SeaShells.

I used simulation to evaluate the two bids, estimating the expected number of units acquired and the resulting profit for each combination. I then selected the pair of bids that maximised expected profit. ([round1.ipynb](./manual/round1.ipynb))

### Round 2
Round2 introduced ORCHIDs, an asset of which it's production rate was influenced by environmental factors (sunlight and humidity) and tradeable against a foreign exchange on the South Island subject to transport, import, and export tariffs.

A major technical nuance arose in the PNL accounting for ORCHIDs. Shorting large volumes of ORCHIDs generated massive upward PNL spikes during the round, only for those profits to abruptly collapse on the final timestamp when open positions were force-closed.

Skimming the Discord discussions revealed that certain participants maintained smooth, steady PNL curves without these visual artifacts. I suspected this was tied to the foreign conversion mechanism—settling local positions directly with the South Archipelago in SeaShells—but balancing a full-time job as a solo competitor delayed my deeper investigation.

I initially assumed the engine executed in an Conversion → Order Execution → PNL Recording → Conversion loop. In reality, the simulator's true sequence was Conversion → PNL Recording → Order Execution → Conversion. Because PNL was logged before new orders were processed rather than at the very end of the cycle, executing conversions at the start of each timestamp settled open inventory prior to snapshot logging, eliminating the artificial profit jumps entirely.

Although I figured the conversion execution steps in round 3, the opportunity for arbitrage substantially dropped after round 2. I wish I had spent more time on deeply investigating simulator's execution order instead of spending too much time to find predictive signals from sunlight and humidity which ultimately had no short term effect on ORCHIDs price movements.

For the Manual Trading Challenge, the objective was to maximize SeaShell payout through a sequence of up to 5 FX currency trades, starting and ending in SeaShells across four available currencies (Pizza, Wasabi, Snowballs, and SeaShells). Given the constrained search space of 5-step conversion paths, I implemented a complete brute-force simulation to evaluate every possible sequence combination. This exhaustive search revealed the optimal arbitrage cycle (SeaShells → Pizza → Wasabi → SeaShells → Pizza → SeaShells), guaranteeing the maximum possible return. ([round2.ipynb](./manual/round2.ipynb))


### Round 3
In the third round, 4 interrelated products were introduced: STRAWBERRIES, CHOCOLATES, ROSES and GIFT BASKET. Each GIFT BASKET comprised of 6 STRAWBERRIES, 4 CHOCOLATES and 1 ROSES. Although the trading environment did not support assembling or unbundling of baskets, I was able to find the fair value of basket given the individual component prices. GIFT BASKETS traded at a premium of $379.50 and whenever the market price of GIFT BASKETS diverged from it fair value (sum of component prices plus the premium) I short/long positions.

The Manual Trading Challenge presented a game-theoretic spatial optimisation problem set on a treasure map grid. Each tile on the map offered a base reward of 7.5K SeaShells multiplied by a tile-specific multiplier, but this total payout had to be shared equally among all players who targeted that same location. Because every additional player on a tile diluted the individual reward, even high-multiplier tiles suffered from severe diminishing returns if over-crowded.

### Round 4
In the fourth round, COCONUTs and COCONUT COUPONs were introduced. COCONUT COUPON is a call option where it will grant the right to buy COCONUT at a price of 10,000 at day 250. Because each round represented a single trading day, the option could not be held to expiration for physical settlement. Instead, using the Black-Scholes model and historical price of COCONUT to estimate the implied volatility, I derive a fair-value baseline and predict the expected price of the Coconut Coupons.

The Manual Trading Challenge extended the sequential auction from Round 1, bringing back the goldfish with reserve prices drawn from the same linearly increasing distribution (900 to 1,000 SeaShells). However, the second-stage bidding mechanism introduced a competitive crowd-dependence twist: a goldfish would accept a second bid if it met their reserve price and exceeded the average second bid across all participants in the archipelago. Bidding below the market average caused the acceptance probability to decay rapidly.

### Round 5
Round 5 introduced no new assets, but the identities of the market’s anonymous algorithmic bots were finally disclosed. By analysing trade volumes and PNL trajectories across all market participants, I identified a distinct counterparty—Rhianna—who traded ROSES with exceptional efficiency, consistently buying at local troughs and selling at peak prices. Recognising her strategic edge, I updated my algorithm to mirror her trades in real time.

For the final Manual Trading Challenge, we were given a news article detailing macro events across the archipelago. The objective was to analyse the news sentiment and decide how much capital to allocate toward long or short positions across the different products.

## Reflection
Competing over these 15 days was an exceptionally fun and rewarding experience, providing a deep dive into designing tailored trading strategies around unique product characteristics.

Round 2 remains a bittersweet turning point. Had I not been distracted by false environmental signals and instead applied a strict, methodical testing framework, securing a top-25 finish was well within reach.

My primary operational takeaway came down to infrastructure: discovering an open-source local backtesting tool late in the competition revealed just how much efficiency I lost by manually iterating parameters through the competition platform's simulator.
