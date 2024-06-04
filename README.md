# IMC Prosperity 2
in April 2024, I participated in the trading challenge Prosperity 2 hosted by IMC trading for the first time.

Prosperity 2 is a worldwide trading competition that goes for 15 days (although there were some issues with the system and extensions and ended up being 17 days), with 5 rounds of 3 days. each round, the participant works on their trading algorithm to trade new product(s) that are introduced each round and submits answers to a manual trading problem.

I participated as a solo team, and finished 199 out of 10000 teams. (team name: Koreant)

## Round results

<table>
    <thead>
        <tr>
            <th></th>
            <th colspan="4" style="text-align: center">Profit / loss</th>
            <th colspan="2" style="text-align: center">Leaderboard position</th>
            <th colspan="2" style="text-align: center">Visualizer links</th>
        </tr>
        <tr>
            <th></th>
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
            <td><a href="https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round1.py">1</a></td>
            <td>122,043</td>
            <td>92,367</td>
            <td>29,676</td>
            <td>122,043</td>
            <td>615</td>
            <td></td>
            <td>a</td>
        </tr>
        <tr>
            <td><a href="https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round2.py">2</a></td>
            <td>283,709</td>
            <td>113,938</td>
            <td>47,727</td>
            <td>161,665</td>
            <td>370</td>
            <td>+245</td>
            <td>a</td>
        </tr>
        <tr>
            <td><a href="https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round3.py">3</a></td>
            <td>453,483</td>
            <td>61,314</td>
            <td>108,460</td>
            <td>169,774</td>
            <td>314</td>
            <td>+56</td>
            <td>a</td>
        </tr>
        <tr>
            <td><a href="https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round4.py">4</a></td>
            <td>816,823</td>
            <td>99,152</td>
            <td>264,188</td>
            <td>363,340</td>
            <td>227</td>
            <td>+87</td>
            <td>a</td>
        </tr>
        <tr>
            <td><a href="https://github.com/jmerle/imc-prosperity-2/blob/master/src/submissions/round5.py">5</a></td>
            <td>1,042,480</td>
            <td>58,210</td>
            <td>167,446</td>
            <td>225,656</td>
            <td>199</td>
            <td>+28</td>
            <td>a</td>
        </tr>
    </tbody>
</table>

## Round summaries
### Round 1
starfruit AMETHYSTS
in round 1, two tradable products, starfruit and amethysts were introduced. amethysts had price fluctuating by +-3 from mean of 10000. I used market taking strategy where further the price from 10000 the more I traded. also market making strategy as long as I am not crossing the 10000 average I constantly traded 1 unit price better than the current price in the market.
for starfruit I found using moving average of last 5 timestamp predicted the next price quite well. and used similar strategy to see if the market is currently under/overpriced compared to the next predicted price and traded accordingly.

the manual trading of round 1 we were asked to give two price to trade with goldfish. where the each goldfish will have a ask price in their mind and I have two opportunity to give offer. gold fish only trades if my offer price is larger than its ask price and I have two chances to give this offer price. the information about the ask price was that it will be between 900 and 1000 and the probability linearly increases. I used linear algebra to solve this.

### Round 2
Orchid had been introduce, which had few interesting characteristics, first is that the production rate of orchid depended on the sunlight and humidity, second is that orchid could be traded with other island as well whilst paying for transportation and storage fees.
there was a pnl graph error for orchid where if you short large amount of orchid the graph showed that you had a huge profit but at the very last timestamp this was converted back to buy buying orchid and coming back to neutral position.
when i entered the discord chat I saw that some peoples pnl didnot have jumps like this but gradually increased, I thought it was something to do with conversion and tried to play around but didnt have enough time with work and solo team and realised in the later round how this was done. although I did find the conversion with other island cannot happen as soon as order is filled. but didnt find the below.
turns out the steps of process was actually,
conversion -> pnl recorded -> order filled
instead of
conversion -> order filled -> pnl recorded
therefore if the position was converted at start of every round you would have not seen any jupms in pnl graph

second manual trading round was to get the highest possible profit given foreign island currency exchange information, for this problem I used the bruteforce method to find out the best serires of exchange as there were only 5 island to exchange from therefore trying out every combination was the easiest to find the best sequence.

### Round 3
in third round, 5 tradable products were introduced which were strawberries, chocolates, roses and gift basket. gift basket consists of 6 strawberries, 4 chocolates and 1 rose. although we werent allow to buy the content of the product and combine to sell it as gift basket, I was able to find what the true value of basket should be given the content prices. basket were charging the premium of $379.5 and whenever the current price of contents + premium went over/under the basket price I short/long baskets.

third round manual trading was interesting problem, we were given a map of treasures as well as the  treasure hunt

### Round 4
in fourth round, coconut and coconut coupon was introduced, coconut coupon is a call option where it will give you the right to buy coconut at 10000 at day 250. each round is 1 trading day so we wouldnt get a chance to execute the option but using the black scholes formula and historical prices to estimate the implied volatility, I predicted the price of coconut coupon 

scuba gear trading with a little twist

### Round 5
no new product was introduced, but the anonymous bots in the market have been disclosed. I look through the volume traded and pnl for all market participants and found Rhianna trades ROSES profitably. Rhianna always bought roses at the lowest price and sold and highest price so i made the algo to copy Rhiannas trades.

for the manual trading we were given a news article about events and were made to decide how much capital to short or long in different products.

## Reflection
fun 15 days thinking and learning about trading strategies for given product. manual trading challenges also fun. round 2 is still painful, could have finished in top 25. if I didnt get distracted by false signals and had proper methodical way of testing. wish I had found the online open source tool earlier as I spend long time testing my algo on the competition website and if i found out the tool earlier would have saved a lot of my time spent in iterative optimising parameters.
