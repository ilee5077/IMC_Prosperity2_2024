#prosperity2bt C:\Users\ilee5\Documents\GitHub\imc-prosperity-2-backtester/example/koreant_algotrading.py 1
#prosperity2bt C:\Users\ilee5\Documents\GitHub\imc-prosperity-2-backtester/example/koreant_algotrading.py 1
#[pandas](https://pandas.pydata.org/)
#[NumPy](https://numpy.org/)
#[statistics](https://docs.python.org/3.9/library/statistics.html)
#[math](https://docs.python.org/3.9/library/math.html)
#[typing](https://docs.python.org/3.9/library/typing.html)
#[jsonpickle](https://pypi.org/project/jsonpickle/)

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import statistics
import math


class Trader:

    def compute_orders_amethysts(self, state: TradingState, acc_bid, acc_ask, POSITION_LIMIT):
        product = 'AMETHYSTS'
        
        orders: list[Order] = []

        # find buy orders
        buy_orders = list(state.order_depths[product].buy_orders.items())
        #print(f"buy_orders: {buy_orders}")
        best_buy_pr, buy_vol = buy_orders[0]

        # find sell orders
        sell_orders = list(state.order_depths[product].sell_orders.items())
        #print(f"sell_orders: {sell_orders}")
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
                #print(f"bought:{bought}")
                #print(f"sold:{sold}")

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
        
        #print(f"arbi orders {product}:{orders}")
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
        print(f";OC mprice;{(best_buy_pr+best_sell_pr)/2};wmprice;{mprice};nextprice;{nxt_price}")
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
        print(f";real_bid;{real_bid_price};bid_profit;{- observations.exportTariff - observations.transportFees};real_ask;{real_ask_price};ask_profit;{observations.importTariff + observations.transportFees};avg_running_price;{avg_running_price};conversions;{conversions}")
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
                    orders.append(Order(product, int(max(real_ask_price + 1, best_sell_pr - 1)), int(num*0.3)))
                    orders.append(Order(product, int(min(real_ask_price + 1, best_sell_pr - 1)), int(num*0.7)))
                else:
                    #conservative less trades
                    orders.append(Order(product, int(max(real_ask_price + 1, best_sell_pr - 1)), int(num*0.7)))
                    #aggressive more trades
                    orders.append(Order(product, int(min(real_ask_price + 1, best_sell_pr - 1)), int(num*0.3)))
                
                sell_cpos_update += num
        
        # sell is profitable
        if real_bid_price > best_buy_pr:
            if buy_cpos_update < pos_limt:
                num = min(200, pos_limt - buy_cpos_update)
                
                if nxt_OC_price > mprice:
                    orders.append(Order(product, min(real_bid_price - 1, best_buy_pr + 1), int(num*0.7)))
                    orders.append(Order(product, max(real_bid_price - 1, best_buy_pr + 1), int(num*0.3)))
                else:
                    orders.append(Order(product, min(real_bid_price - 1, best_buy_pr + 1), int(num*0.3)))
                    orders.append(Order(product, max(real_bid_price - 1, best_buy_pr + 1), int(num*0.7)))
                buy_cpos_update += num

        return orders, conversions

    def run(self, state: TradingState):
        
        # Orders to be placed on exchange matching engine
        result = {}

        if state.traderData == '':
            mydata = {}
            mydata['position_limit'] = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100}
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
        else:
            mydata = jsonpickle.decode(state.traderData)


        #COMMENT BELOW WHEN SUBMITTING
        #product_keys = [
            #'AMETHYSTS'
            #,'STARFRUIT'
        #       'ORCHIDS']
        
        #Only method required. It takes all buy and sell orders for all symbols as an input,
        #and outputs a list of orders to be sent
        want_to_see_product = 'ORCHIDS'
        print(f";timestamp;{state.timestamp}")
        #print("LISTING")
        #print(state.listings)
        print(f";buyorders;{list(state.order_depths[want_to_see_product].buy_orders.items())};sellorders;{list(state.order_depths[want_to_see_product].sell_orders.items())};own_trades;{state.own_trades.get(want_to_see_product,"")};")
        print(f";Baskprice;{state.observations.conversionObservations[want_to_see_product].askPrice};Bbidprice;{state.observations.conversionObservations[want_to_see_product].bidPrice};exportT;{state.observations.conversionObservations[want_to_see_product].exportTariff};importT;{state.observations.conversionObservations[want_to_see_product].importTariff};transportF;{state.observations.conversionObservations[want_to_see_product].transportFees};")
        print(f";position;{state.position.get(want_to_see_product,"")}")
        print(f";market_trades;{state.own_trades.get(want_to_see_product,"")}")
        print(f";market_trades;{state.market_trades.get(want_to_see_product,"")}")
        


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
            #print(f"next price SF: {nxt_SF_price}, next price LR: {round(gradient * 6 + intercept,0)},history: {mydata['starfruit_cache']}, gradient: {gradient}")            
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
        #conversions = None
        
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
        #    print(key,item)
        traderData = jsonpickle.encode(mydata) 



        #result = {k: v for k, v in result.items() if k in (product_keys)}

        print(f";order;{result}")
        return result, conversions, traderData
    