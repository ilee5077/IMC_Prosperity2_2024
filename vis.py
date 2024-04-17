# pip install -U prosperity2bt
# prosperity2bt C:\Users\ilee5\Documents\GitHub\python\vis.py 1

#orchid 11k
#round 3 400k on backtester
#strawberries nominal change mean std low do moving average and buy sell

import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import statistics
import math

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:


        # Orders to be placed on exchange matching engine
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'CHOCOLATE' : [], 'STRAWBERRIES' : [], 'ROSES' : [], 'GIFT_BASKET' : []}


        if state.traderData == '':
            mydata = {}
            mydata['position_limit'] = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100, 'CHOCOLATE' : 250, 'STRAWBERRIES' : 350, 'ROSES' : 60, "GIFT_BASKET" : 60}
            mydata['starfruit_cache'] = []
            mydata['starfruit_dim'] = 5
            mydata['last_position'] = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS' : 0}
            mydata['cpnl'] = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS' : 0}
            mydata['volume_traded'] = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS' : 0}
            mydata['running_buy_price'] = 0
            mydata['running_sell_price'] = 0
            mydata['running_buy_vol'] = 0
            mydata['running_sell_vol'] = 0
            mydata['OC_gap_average'] = 0
            mydata['cont_buy_basket_unfill'] = 0
            mydata['cont_sell_basket_unfill'] = 0
        else:
            mydata = jsonpickle.decode(state.traderData)


        #COMMENT BELOW WHEN SUBMITTING
        #product_keys = [#'AMETHYSTS','STARFRUIT','ORCHIDS',
        #'CHOCOLATE','STRAWBERRIES','ROSES','GIFT_BASKET']
        product_keys = ['ROSES']
        
        #Only method required. It takes all buy and sell orders for all symbols as an input,
        #and outputs a list of orders to be sent
        '''
        want_to_see_product = 'ORCHIDS'
        logger.print(f";timestamp;{state.timestamp}")
        #logger.print("LISTING")
        #logger.print(state.listings)
        logger.print(f";buyorders;{list(state.order_depths[want_to_see_product].buy_orders.items())};sellorders;{list(state.order_depths[want_to_see_product].sell_orders.items())};own_trades;{state.own_trades.get(want_to_see_product,"")};")
        logger.print(f";Baskprice;{state.observations.conversionObservations[want_to_see_product].askPrice};Bbidprice;{state.observations.conversionObservations[want_to_see_product].bidPrice};exportT;{state.observations.conversionObservations[want_to_see_product].exportTariff};importT;{state.observations.conversionObservations[want_to_see_product].importTariff};transportF;{state.observations.conversionObservations[want_to_see_product].transportFees};")
        logger.print(f";position;{state.position.get(want_to_see_product,"")}")
        logger.print(f";market_trades;{state.own_trades.get(want_to_see_product,"")}")
        logger.print(f";market_trades;{state.market_trades.get(want_to_see_product,"")}")
        '''

        '''
        result['AMETHYSTS'] = self.compute_orders_amethysts(state, acc_bid = 10000, acc_ask = 10000, POSITION_LIMIT = mydata['position_limit'])        

        buy_orders = list(state.order_depths['STARFRUIT'].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]

        sell_orders = list(state.order_depths['STARFRUIT'].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        mydata['starfruit_cache'].append((best_buy_pr+best_sell_pr)/2.0)
        nxt_SF_price = 0
        if len(mydata['starfruit_cache']) > mydata['starfruit_dim']:
            mydata['starfruit_cache'].pop(0)
            nxt_SF_price = int(round(sum(mydata['starfruit_cache'])/float(len(mydata['starfruit_cache']))))
            #nxt_SF_price = statistics.median(mydata['starfruit_cache'])
            gradient, intercept = np.polyfit(list(range(mydata['starfruit_dim'])),mydata['starfruit_cache'],1)
            nxt_SF_price2 = int(round(gradient * 6 + intercept,0))
            #logger.print(f"next price SF: {nxt_SF_price}, next price LR: {round(gradient * 6 + intercept,0)},history: {mydata['starfruit_cache']}, gradient: {gradient}")            
            # now we know the next SF price is going to be - so we can start trade!
            result['STARFRUIT'] = self.compute_orders_starfruit(state, acc_bid = int(round(nxt_SF_price-1,0)), acc_ask = math.floor(nxt_SF_price+1), POSITION_LIMIT = mydata['position_limit'])
            


        if state.position.get("ORCHIDS",0) > 0:
            mydata['running_buy_price'] += 0.1

        for product in state.own_trades.keys():
            if product == 'ORCHIDS':
                for trade in state.own_trades['ORCHIDS']:
                    if trade.timestamp != state.timestamp-100:
                        continue
                    
                    if trade.buyer == "SUBMISSION": 
                        mydata['running_buy_price'] = (mydata['running_buy_price'] * mydata['running_buy_vol'] + trade.quantity * trade.price)/(mydata['running_buy_vol'] + trade.quantity)
                        mydata['running_buy_vol'] += trade.quantity
                    else:
                        mydata['running_sell_price'] = (mydata['running_sell_price'] * mydata['running_sell_vol'] + trade.quantity * trade.price)/(mydata['running_sell_vol'] + trade.quantity)
                        mydata['running_sell_vol'] += trade.quantity
        

        if (mydata['running_buy_vol'] - mydata['running_sell_vol']) != 0:
            average_price = round((mydata['running_buy_vol'] * mydata['running_buy_price'] - mydata['running_sell_vol'] * mydata['running_sell_price']) / (mydata['running_buy_vol'] - mydata['running_sell_vol']),1)
        else:
            average_price = 0

        nxt_OC_price = self.next_orchid_price(state)
        result['ORCHIDS'], conversions = self.compute_orders_orchids(state, nxt_OC_price = nxt_OC_price, acc_bid = nxt_OC_price - 1, acc_ask = nxt_OC_price + 1, POSITION_LIMIT = mydata['position_limit'], avg_running_price = average_price)
        '''


        orders = self.compute_orders_basket(state.order_depths, state = state, POSITION_LIMIT = mydata['position_limit'], mydata = mydata)
        result['GIFT_BASKET'] += orders['GIFT_BASKET'] 
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['ROSES'] += orders['ROSES']

        conversions = None
        if conversions is not None:
            mydata['running_buy_price'] = 0
            mydata['running_sell_price'] = 0
            mydata['running_buy_vol'] = 0
            mydata['running_sell_vol'] = 0

        if state.position.get('ORCHIDS',0) > 0:
            mydata['cpnl']['ORCHIDS'] -= state.position.get('ORCHIDS',0) * 0.1
        
        # String value holding Trader state data required. 
        # It will be delivered as TradingState.traderData on next execution.
        #for key, item in mydata.items():
        #    logger.print(key,item)
        traderData = jsonpickle.encode(mydata) 



        result = {k: v for k, v in result.items() if k in (product_keys)}

        logger.print(f";order;{result}")






        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
        

    def compute_orders_amethysts(self, state: TradingState, acc_bid, acc_ask, POSITION_LIMIT):
        product = 'AMETHYSTS'
        
        orders: list[Order] = []

        # find buy orders
        buy_orders = list(state.order_depths[product].buy_orders.items())
        #logger.print(f"buy_orders: {buy_orders}")
        best_buy_pr, buy_vol = buy_orders[0]

        # find sell orders
        sell_orders = list(state.order_depths[product].sell_orders.items())
        #logger.print(f"sell_orders: {sell_orders}")
        best_sell_pr, sell_vol = sell_orders[0]

        pos_limt = POSITION_LIMIT[product]

        cpos= 0
        if product in state.position.keys():
            cpos = state.position[product]

        mx_with_buy = -1
        buy_cpos_update = cpos
        for ask, vol in sell_orders:
            # MATCHING ASK ORDERS
            # sell for less than the price we willing to buy OR
            # sell for same as we want to buy and our position is short
            if ((ask < acc_bid) or ((cpos<0) and (ask == acc_bid))) and buy_cpos_update < pos_limt:
                mx_with_buy = max(mx_with_buy, ask) # buy price
                order_for = min(-vol, pos_limt - cpos) # how many do we buy for
                buy_cpos_update += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2



        cpos= 0
        if product in state.position.keys():
            cpos = state.position[product]
        sell_cpos_update = cpos
        # short positioning
        for bid, vol in buy_orders:
            # MATCHING BUY ORDERS
            # someone is willing to buy more than what we ask
            # we are long and someone buying at price we think
            if ((bid > acc_ask) or ((cpos>0) and (bid == acc_ask))) and sell_cpos_update > -pos_limt:
                order_for = max(-vol, -pos_limt-sell_cpos_update)
                # order_for is a negative number denoting how much we will sell
                sell_cpos_update += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))



        if len(orders) == 0:
            arbi_order = self.check_for_arbi(state, product, best_buy_pr=best_buy_pr, best_sell_pr=best_sell_pr, best_buy_vol=buy_vol,best_sell_vol=sell_vol)
            for i in arbi_order:
                orders.append(i)
                if i.quantity > 0:
                    buy_cpos_update += i.quantity
                if i.quantity < 0:
                    sell_cpos_update += i.quantity


        undercut_buy = best_buy_pr + 1

        # buy orders
        if (buy_cpos_update < pos_limt) and (cpos < 0): # short position
            num = min(40, pos_limt - buy_cpos_update)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
            buy_cpos_update += num

        if (buy_cpos_update < pos_limt) and (cpos > 15): # long position
            num = min(40, pos_limt - buy_cpos_update)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
            buy_cpos_update += num

        if buy_cpos_update < pos_limt: # position between 1 to 14
            num = min(40, pos_limt - buy_cpos_update)
            orders.append(Order(product, min(undercut_buy, acc_bid-1), num))
            buy_cpos_update += num

        undercut_sell = best_sell_pr - 1
        # placing orders
        if (sell_cpos_update > -pos_limt) and (cpos > 0): # long position
            num = max(-40, -pos_limt-sell_cpos_update) # sell 40 or 
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            sell_cpos_update += num

        if (sell_cpos_update > -pos_limt) and (cpos < -15):
            num = max(-40, -pos_limt-sell_cpos_update)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            sell_cpos_update += num

        if sell_cpos_update > -pos_limt:
            num = max(-40, -pos_limt-sell_cpos_update)
            orders.append(Order(product, max(undercut_sell, acc_ask+1), num))
            sell_cpos_update += num



        return orders
    
    def compute_orders_starfruit(self, state: TradingState, acc_bid, acc_ask, POSITION_LIMIT):
        product = 'STARFRUIT'
        
        orders: list[Order] = []

        buy_orders = list(state.order_depths[product].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]

        sell_orders = list(state.order_depths[product].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        pos_limt = POSITION_LIMIT[product]

        cpos= 0
        if product in state.position.keys():
            cpos = state.position[product]

        mx_with_buy = -1
        buy_cpos_update = cpos
        for ask, vol in sell_orders:
            # MATCHING ASK ORDERS
            # sell for less than the price we willing to buy OR
            # sell for same as we want to buy and our position is short
            if ((ask <= acc_bid) or ((cpos<0) and (ask == acc_bid+1))) and buy_cpos_update < pos_limt:
                mx_with_buy = max(mx_with_buy, ask) # buy price
                order_for = min(-vol, pos_limt - cpos) # how many do we buy for
                buy_cpos_update += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid+acc_ask)/2

        cpos= 0
        if product in state.position.keys():
            cpos = state.position[product]
        sell_cpos_update = cpos
        # short positioning
        for bid, vol in buy_orders:
            # MATCHING BUY ORDERS
            # someone is willing to buy more than what we ask
            # we are long and someone buying at price we think
            if ((bid >= acc_ask) or ((cpos>0) and (bid+1 == acc_ask))) and sell_cpos_update > -pos_limt:
                order_for = max(-vol, -pos_limt-sell_cpos_update)
                # order_for is a negative number denoting how much we will sell
                sell_cpos_update += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
        
        if len(orders) == 0:
            arbi_order = self.check_for_arbi(state, product, best_buy_pr=best_buy_pr, best_sell_pr=best_sell_pr, best_buy_vol=buy_vol,best_sell_vol=sell_vol)
            for i in arbi_order:
                orders.append(i)
                if i.quantity > 0:
                    buy_cpos_update += i.quantity
                if i.quantity < 0:
                    sell_cpos_update += i.quantity

        
        # undercut
        if buy_cpos_update < pos_limt:
            num = pos_limt - buy_cpos_update
            orders.append(Order(product, min(best_buy_pr + 1, acc_bid), num))
            buy_cpos_update += num
        
        if sell_cpos_update > -pos_limt:
            num = -pos_limt-sell_cpos_update
            orders.append(Order(product, max(best_sell_pr - 1, acc_ask), num))
            sell_cpos_update += num



        return orders
    
    def check_for_arbi(self, state: TradingState, product, best_buy_pr, best_sell_pr, best_buy_vol, best_sell_vol):
        orders: list[Order] = []
        
        bought = []
        sold = []
        if product in state.own_trades:
            for i in list(state.own_trades[product]):
                if i.buyer == 'SUBMISSION':
                    bought.append(i)
                if i.seller == 'SUBMISSION':
                    sold.append(i)
                #logger.print(f"bought:{bought}")
                #logger.print(f"sold:{sold}")

        #buy arbitrage
        if len(bought) > 0:
            if best_sell_pr - best_buy_pr < best_buy_pr - bought[0].price:
                order_vol = min(bought[0].quantity,best_buy_vol,-best_sell_vol)
                orders.append(Order(product, best_buy_pr, order_vol))
                orders.append(Order(product, best_sell_pr, -order_vol))
        
        if len(sold) > 0:
            if best_sell_pr - best_buy_pr < sold[0].price - best_sell_pr:
                order_vol = max(-sold[0].quantity,-best_buy_vol,best_sell_vol)
                orders.append(Order(product, best_buy_pr, -order_vol))
                orders.append(Order(product, best_sell_pr, order_vol))
        
        #logger.print(f"arbi orders {product}:{orders}")
        return orders

    def next_orchid_price(self, state: TradingState):
        # end of timestamp dont trade!
        product = 'ORCHIDS'
        intercept = -0.0439889100189248
        #'ORCHIDS'    'SUNLIGHT'     'HUMIDITY'     'diffSUNLIGHT>2500'     'humdiff'
        coef = [0.999623941,    -0.0000250973316,    0.0067920161,    0.0000636415177,    -0.0110500552]

        observations = state.observations.conversionObservations[product]

        buy_orders = list(state.order_depths[product].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]

        sell_orders = list(state.order_depths[product].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]


        sunlight = observations.sunlight
        humidity = observations.humidity
        
        sundiff = max(sunlight-2500,0)
        humdiff = max(max(humidity-80,0),max(60-humidity,0))

        sum = 0
        sumvol = 0
        if len(buy_orders) > 1:
            for bid,bid_vol in buy_orders[:-1]:
                sum += bid*bid_vol
                sumvol += bid_vol
        else:
            bid,bid_vol = buy_orders[0]
            sum += bid*bid_vol
            sumvol += bid_vol
        if len(sell_orders) > 1:    
            for ask,ask_vol in sell_orders[:-1]:
                sum += ask*-ask_vol
                sumvol += -ask_vol
        else:
            ask,ask_vol = sell_orders[0]
            sum += ask*ask_vol
            sumvol += ask_vol

        mprice = sum/sumvol

        observed_values = []

        observed_values.append(mprice)
        observed_values.append(sunlight)
        observed_values.append(humidity)
        observed_values.append(sundiff)
        observed_values.append(humdiff)
        
        nxt_price = int(round(intercept + math.sumprod(coef,observed_values),0))
        logger.print(f";OC mprice;{(best_buy_pr+best_sell_pr)/2};wmprice;{mprice};nextprice;{nxt_price}")
        return nxt_price
    
    def compute_arb_conversion_order(self, state: TradingState, avg_running_price, acc_bid, acc_ask):
        product = 'ORCHIDS'
        observations = state.observations.conversionObservations[product]
        conversions = None
        #price we are paying to buy
        real_ask_price = observations.askPrice + observations.importTariff + observations.transportFees
        
        #price we are getting to sell
        real_bid_price = observations.bidPrice - observations.exportTariff - observations.transportFees
        
        if state.position.get(product,0) > 0:
            if (real_bid_price >= avg_running_price) and (real_bid_price > acc_ask):
                conversions = -state.position[product]
        
        if state.position.get(product,0) < 0:
            if (real_ask_price < avg_running_price) and (real_ask_price < acc_bid):
                conversions = -state.position[product]
        logger.print(f";real_bid;{real_bid_price};bid_profit;{- observations.exportTariff - observations.transportFees};real_ask;{real_ask_price};ask_profit;{observations.importTariff + observations.transportFees};avg_running_price;{avg_running_price};conversions;{conversions}")
        return conversions, real_ask_price, real_bid_price

    def compute_orders_orchids(self, state: TradingState, nxt_OC_price, acc_bid, acc_ask, POSITION_LIMIT, avg_running_price):
        product = 'ORCHIDS'
        
        #if state.position[product] != 0:
        #    conversion = self.compute_arb_conversion_order()
        conversions, real_ask_price, real_bid_price = self.compute_arb_conversion_order(state, avg_running_price = avg_running_price, acc_bid =acc_bid, acc_ask=acc_ask)

        orders: list[Order] = []

        buy_orders = list(state.order_depths[product].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]

        sell_orders = list(state.order_depths[product].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        pos_limt = POSITION_LIMIT[product]

        cpos= 0
        if product in state.position.keys() and conversions is None:
            cpos = state.position[product]

        mx_with_buy = -1
        buy_cpos_update = cpos
        for ask, vol in sell_orders:
            # MATCHING ASK ORDERS
            # sell for less than the price we willing to buy OR
            # sell for same as we want to buy and our position is short
            if ((ask < acc_bid) or ((cpos<0) and (ask == acc_bid)) or (real_bid_price > ask)) and buy_cpos_update < pos_limt:
            #if ((ask < acc_bid) or ((cpos<0) and (ask == acc_bid)) or (real_bid_price > ask + 1)) and buy_cpos_update < pos_limt:
                mx_with_buy = max(mx_with_buy, ask) # buy price
                order_for = min(-vol, pos_limt - cpos) # how many do we buy for
                buy_cpos_update += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        cpos= 0
        if product in state.position.keys() and conversions is None:
            cpos = state.position[product]
        sell_cpos_update = cpos
        # short positioning
        for bid, vol in buy_orders:
            # MATCHING BUY ORDERS
            # someone is willing to buy more than what we ask
            # we are long and someone buying at price we think
            if ((bid > acc_ask) or ((cpos>0) and (bid == acc_ask)) or (real_ask_price < bid))  and sell_cpos_update > -pos_limt:
            #if ((bid > acc_ask) or ((cpos>0) and (bid == acc_ask)) or (real_ask_price < bid - 1))  and sell_cpos_update > -pos_limt:
                order_for = max(-vol, -pos_limt-sell_cpos_update)
                # order_for is a negative number denoting how much we will sell
                sell_cpos_update += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))
        
        sum = 0
        sumvol = 0
        if len(buy_orders) > 1:
            for bid,bid_vol in buy_orders[:-1]:
                sum += bid*bid_vol
                sumvol += bid_vol
        else:
            bid,bid_vol = buy_orders[0]
            sum += bid*bid_vol
            sumvol += bid_vol
        if len(sell_orders) > 1:    
            for ask,ask_vol in sell_orders[:-1]:
                sum += ask*-ask_vol
                sumvol += -ask_vol
        else:
            ask,ask_vol = sell_orders[0]
            sum += ask*ask_vol
            sumvol += ask_vol

        mprice = int(round(sum/sumvol,0))


        observationB = state.observations.conversionObservations[product]
        undercut_sell = mprice + 1
        undercut_buy = mprice - 1    


        
        # buy is profitable
        if real_ask_price < best_sell_pr:
            if sell_cpos_update > -pos_limt:
                num = max(-200, -pos_limt-sell_cpos_update)

                if nxt_OC_price > mprice:
                    orders.append(Order(product, int(round(max(real_ask_price + 1, best_sell_pr - 1),0)), int(num*0.7)))
                    orders.append(Order(product, int(round(min(real_ask_price + 1, best_sell_pr - 1),0)), int(num*0.3)))
                else:
                    #conservative less tradesround(
                    orders.append(Order(product, int(round(max(real_ask_price + 1, best_sell_pr - 1),0)), int(num*0.3)))
                    #aggressive more tradesround(
                    orders.append(Order(product, int(round(min(real_ask_price + 1, best_sell_pr - 1),0)), int(num*0.7)))
                
                sell_cpos_update += num
        
        # sell is profitable
        if real_bid_price > best_buy_pr:
            if buy_cpos_update < pos_limt:
                num = min(200, pos_limt - buy_cpos_update)
                
                if nxt_OC_price > mprice:
                    orders.append(Order(product, int(round(min(real_bid_price - 1, best_buy_pr + 1),0)), int(num*0.3)))
                    orders.append(Order(product, int(round(max(real_bid_price - 1, best_buy_pr + 1),0)), int(num*0.7)))
                else:
                    orders.append(Order(product, int(round(min(real_bid_price - 1, best_buy_pr + 1),0)), int(num*0.7)))
                    orders.append(Order(product, int(round(max(real_bid_price - 1, best_buy_pr + 1),0)), int(num*0.3)))
                buy_cpos_update += num

        return orders, conversions
        
    def compute_orders_basket(self, order_depth, state: TradingState, POSITION_LIMIT, mydata):

        orders = {'STRAWBERRIES' : [], 'CHOCOLATE': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['STRAWBERRIES', 'CHOCOLATE', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = list(state.order_depths[p].sell_orders.items())
            obuy[p] = list(state.order_depths[p].buy_orders.items())

            best_sell[p] = next(iter(osell[p]))[0]
            best_buy[p] = next(iter(obuy[p]))[0]

            worst_sell[p] = next(reversed(osell[p]))[0]
            worst_buy[p] = next(reversed(obuy[p]))[0]

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p]:
                vol_buy[p] += vol 
                if vol_buy[p] >= POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p]:
                vol_sell[p] += -vol 
                if vol_sell[p] >= POSITION_LIMIT[p]/10:
                    break
        
        # mean:379.4904833333333;median381.0;std76.42310842343252
        res_buy = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 379.5
        res_sell = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 379.5

        trade_at = 76.4*0.5
        close_at = 76.4*(-1000)
        
        gb_pos = state.position.get('GIFT_BASKET',0)
        gb_neg = state.position.get('GIFT_BASKET',0)

        rs_pos = state.position.get('ROSES',0)
        rs_neg = state.position.get('ROSES',0)

        
        basket_buy_sig = 0
        basket_sell_sig = 0

        if state.position.get('GIFT_BASKET',0) == POSITION_LIMIT['GIFT_BASKET']:
            mydata['cont_buy_basket_unfill'] = 0
        if state.position.get('GIFT_BASKET',0) == -POSITION_LIMIT['GIFT_BASKET']:
            mydata['cont_sell_basket_unfill'] = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = state.position.get('GIFT_BASKET',0) + POSITION_LIMIT['GIFT_BASKET']
            mydata['cont_buy_basket_unfill'] = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                mydata['cont_sell_basket_unfill'] += 2
                gb_neg -= vol
                #uku_pos += vol
        elif res_buy < -trade_at:
            vol = POSITION_LIMIT['GIFT_BASKET'] - state.position.get('GIFT_BASKET',0)
            mydata['cont_sell_basket_unfill'] = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                mydata['cont_buy_basket_unfill'] += 2
                gb_pos += vol

        

        if res_sell > 76.4*0.7:
            vol = state.position.get('STRAWBERRIES',0) + POSITION_LIMIT['STRAWBERRIES']
            mydata['cont_buy_basket_unfill'] = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol)) 
                mydata['cont_sell_basket_unfill'] += 2
                gb_neg -= vol
                #uku_pos += vol
        elif res_buy < -76.4*0.7:
            vol = POSITION_LIMIT['STRAWBERRIES'] - state.position.get('STRAWBERRIES',0)
            mydata['cont_sell_basket_unfill'] = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol))
                mydata['cont_buy_basket_unfill'] += 2
                gb_pos += vol

        if res_sell > 76.4*0.6:
            vol = state.position.get('CHOCOLATE',0) + POSITION_LIMIT['CHOCOLATE']
            mydata['cont_buy_basket_unfill'] = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol)) 
                mydata['cont_sell_basket_unfill'] += 2
                gb_neg -= vol
                #uku_pos += vol
        elif res_buy < -76.4*0.6:
            vol = POSITION_LIMIT['CHOCOLATE'] - state.position.get('CHOCOLATE',0)
            mydata['cont_sell_basket_unfill'] = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol))
                mydata['cont_buy_basket_unfill'] += 2
                gb_pos += vol


        if res_sell > 76.4*0.3:
            vol = state.position.get('ROSES',0) + POSITION_LIMIT['ROSES']
            mydata['cont_buy_basket_unfill'] = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol)) 
                mydata['cont_sell_basket_unfill'] += 2
                gb_neg -= vol
                #uku_pos += vol
        elif res_buy < -76.4*0.3:
            vol = POSITION_LIMIT['ROSES'] - state.position.get('ROSES',0)
            mydata['cont_sell_basket_unfill'] = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol))
                mydata['cont_buy_basket_unfill'] += 2
                gb_pos += vol

        


        return orders