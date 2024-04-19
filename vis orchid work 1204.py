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
            mydata['tot_good_sl'] = 0
            mydata['prev_oc_p'] = 0
            mydata['cont_buy_basket_unfill'] = 0
            mydata['cont_sell_basket_unfill'] = 0
        else:
            mydata = jsonpickle.decode(state.traderData)


        #COMMENT BELOW WHEN SUBMITTING
        #product_keys = [#'AMETHYSTS','STARFRUIT','ORCHIDS',
        #'CHOCOLATE','STRAWBERRIES','ROSES','GIFT_BASKET']
        product_keys = ['AMETHYSTS','STARFRUIT','ORCHIDS','CHOCOLATE','STRAWBERRIES','ROSES','GIFT_BASKET']
                
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
            
        '''
        nxt_OC_price, mydata = self.next_orchid_price(state = state, mydata = mydata)
        result['ORCHIDS'], conversions = self.compute_orders_orchids(state, nxt_OC_price = nxt_OC_price, acc_bid = nxt_OC_price - 1, acc_ask = nxt_OC_price + 1, POSITION_LIMIT = mydata['position_limit'])
        '''
        result['ORCHIDS'], conversions = self.compute_orders_orchids2(state, POSITION_LIMIT = mydata['position_limit'])
        

        orders = self.compute_orders_basket(state.order_depths, state = state, POSITION_LIMIT = mydata['position_limit'], mydata = mydata)
        result['GIFT_BASKET'] += orders['GIFT_BASKET'] 
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['ROSES'] += orders['ROSES']


        # String value holding Trader state data required. 
        # It will be delivered as TradingState.traderData on next execution.
        #for key, item in mydata.items():
        #    logger.print(key,item)
        traderData = jsonpickle.encode(mydata) 



        result = {k: v for k, v in result.items() if k in (product_keys)}

        #logger.print(f";order;{result}")






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

    def next_orchid_price(self, state: TradingState, mydata):
        # end of timestamp dont trade!
        product = 'ORCHIDS'
        intercept = 1805.904358
        coef = [                    
        -0.011238 #SUNLIGHT                      
        ,      5.887798 #HUMIDITY                      
        ,     -5.015072 #hd_out                        
        ,      0.012797 #tot_good_sl                   
        ,  -1250.413591 #sl_prod_reduction             
        ,     -0.020060 #hm_prod_reduction             
        ,  -1156.001078 #total_prod_reduction_multi    
        ,     11.110230]#sl_more_than_7_flag2          


        observations = state.observations.conversionObservations[product]
        buy_orders = list(state.order_depths[product].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]
        sell_orders = list(state.order_depths[product].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        if state.timestamp == 0:
            mydata['tot_good_sl'] = 0
            mydata['prev_oc_p'] = (best_buy_pr+best_sell_pr)/2
        if observations.sunlight >= 2500:
            mydata['tot_good_sl'] += 1

        sl_more_than_7_flag = 0
        if mydata['tot_good_sl']/(state.timestamp/100+1) > 7/24:
            sl_more_than_7_flag = 1

        hm_out = max(observations.humidity-80,60-observations.humidity,0)
        sl_prod_reduction = max((2500 - observations.sunlight)/60.833 * 0.04,0)
        hm_prod_reduction = max(observations.humidity-80,60-observations.humidity,0)/5 * 0.02
        
        observed_values = [
            observations.sunlight
            ,observations.humidity
            ,hm_out
            ,mydata['tot_good_sl']
            ,sl_prod_reduction
            ,hm_prod_reduction
            ,(1-sl_prod_reduction)*(1-hm_prod_reduction)
            ,sl_more_than_7_flag
        ]

        nxt_price = intercept + math.sumprod(coef,observed_values)
        nxt_price = mydata['prev_oc_p']
        mydata['prev_oc_p'] = (best_buy_pr+best_sell_pr)/2
        return nxt_price, mydata
    
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

    def compute_orders_orchids(self, state: TradingState, nxt_OC_price, acc_bid, acc_ask, POSITION_LIMIT):
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}
        orders = []
        
        p = 'ORCHIDS'
        
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

        #mean:-1.6012354142836227e-11;median-0.04662620440399223;std3.202175387652208
        res = nxt_OC_price - (best_sell[p]+best_buy[p])/2

        trade_at = 1.45*0.5

        oc_pos = state.position.get(p,0)
        oc_neg = state.position.get(p,0)
        logger.print(f'res;{res};nxt_OC_price;{nxt_OC_price};mid_price;{mid_price[p]};')
        if res > trade_at: #buy
            vol = state.position.get(p,0) + POSITION_LIMIT[p]
            assert(vol >= 0)
            if vol > 0:
                orders.append(Order(p, best_buy[p], -vol))
                oc_neg -= vol
        elif res < -trade_at: # sell
            vol = POSITION_LIMIT[p] - state.position.get(p,0)
            assert(vol >= 0)
            if vol > 0:
                orders.append(Order(p, best_sell[p], vol))
                oc_pos += vol

        conversions = 0
        '''
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
        '''
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
    
    def compute_orders_orchids2(self, state: TradingState, POSITION_LIMIT):
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}
        orders = []
        
        p = 'ORCHIDS'
        observation = state.observations.conversionObservations[p]
        osell[p] = list(state.order_depths[p].sell_orders.items())
        obuy[p] = list(state.order_depths[p].buy_orders.items())

        best_sell[p] = next(iter(osell[p]))[0]
        best_buy[p] = next(iter(obuy[p]))[0]

        worst_sell[p] = next(reversed(osell[p]))[0]
        worst_buy[p] = next(reversed(obuy[p]))[0]

        conversions = -state.position.get(p,0)

        sellprice = observation.askPrice + observation.importTariff + observation.transportFees

        buyprice = observation.bidPrice + observation.exportTariff + observation.transportFees
        
        if state.position.get(p,0) >= 0:
            vol = state.position.get(p,0) + POSITION_LIMIT[p]
            #assert(vol >= 0)
            if vol > 0:
                orders.append(Order(p, min(sellprice + 1, best_buy[p]), -vol))
            
        if state.position.get(p,0) <= 0:
            vol = POSITION_LIMIT[p] - state.position.get(p,0)
            #assert(vol >= 0)
            if vol > 0:
                orders.append(Order(p, min(buyprice-1,best_sell[p]), vol))

        return orders, conversions