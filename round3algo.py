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
        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'CHOCOLATE' : [], 'STRAWBERRIES' : [], 'ROSES' : [], 'GIFT_BASKET' : [], 'COCONUT' : [], 'COCONUT_COUPON' : []}


        if state.traderData == '':
            mydata = {}
            mydata['position_limit'] = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100, 'CHOCOLATE' : 250, 'STRAWBERRIES' : 350, 'ROSES' : 60, "GIFT_BASKET" : 60, "COCONUT":300, "COCONUT_COUPON":600}
            

            traderData = jsonpickle.encode(mydata)
            mydata = jsonpickle.decode(traderData)
        else:
            mydata = jsonpickle.decode(state.traderData)
        
        orders = self.compute_orders_basket(state.order_depths, state = state, POSITION_LIMIT = mydata['position_limit'], mydata = mydata)
        result['GIFT_BASKET'] += orders['GIFT_BASKET'] 
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['ROSES'] += orders['ROSES']

        traderData = jsonpickle.encode(mydata)
        conversions = None
        logger.print(f";order;{result}")
		
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    

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
        premium = 379.5
        res_buy_gb = mid_price['GIFT_BASKET'] - (mid_price['STRAWBERRIES']*6 + mid_price['CHOCOLATE']*4 + mid_price['ROSES']) - premium
        res_sell_gb = mid_price['GIFT_BASKET'] - (mid_price['STRAWBERRIES']*6 + mid_price['CHOCOLATE']*4 + mid_price['ROSES']) - premium
        
        res_buy = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 379.5
        res_sell = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 - mid_price['CHOCOLATE']*4 - mid_price['ROSES'] - 379.5

        trade_at_gb = 76.4*0.5
        
        res_buy_cc = mid_price['CHOCOLATE'] - ((mid_price['GIFT_BASKET'] - 6*mid_price['STRAWBERRIES'] - mid_price['ROSES'])/4  )+ 94.873
        res_sell_cc = mid_price['CHOCOLATE'] - ((mid_price['GIFT_BASKET'] - 6*mid_price['STRAWBERRIES'] - mid_price['ROSES'])/4 )+ 94.873
        
        trade_at_cc = 19.106*0.1

        res_buy_sb = mid_price['STRAWBERRIES'] - ((mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES'])/6  )+ 63.248
        res_sell_sb = mid_price['STRAWBERRIES'] - ((mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['ROSES'])/6 )+ 63.248
        
        trade_at_sb = 12.737 * 1.5

        res_buy_rs = mid_price['ROSES'] - ((mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - 6*mid_price['STRAWBERRIES'])  )+ 379.5
        res_sell_rs = mid_price['ROSES'] - ((mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - 6*mid_price['STRAWBERRIES'])) + 379.5
        
        trade_at_rs = 76.4*1.5
        
        close_at = 76.4*(-1000)
        
        gb_pos = state.position.get('GIFT_BASKET',0)
        gb_neg = state.position.get('GIFT_BASKET',0)

        rs_pos = state.position.get('ROSES',0)
        rs_neg = state.position.get('ROSES',0)

        
        basket_buy_sig = 0
        basket_sell_sig = 0


        '''
        do_bask = 0
        if res_sell_gb > trade_at_gb:
            vol = state.position.get('GIFT_BASKET',0) + POSITION_LIMIT['GIFT_BASKET']
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                gb_neg -= vol
                #uku_pos += vol
        elif res_buy_gb < -trade_at_gb:
            vol = POSITION_LIMIT['GIFT_BASKET'] - state.position.get('GIFT_BASKET',0)
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                gb_pos += vol
        '''

        std = 76.4
        tot = mid_price['STRAWBERRIES']*6 + mid_price['CHOCOLATE']*4 + mid_price['ROSES']
        buy_cpos_gb = state.position.get('GIFT_BASKET',0)
        if (res_buy < -std*0.5) and (res_buy > -std*2) and (buy_cpos_gb < POSITION_LIMIT['GIFT_BASKET']): #coupon underpriced, buy now and sell later
            for ask, vol in osell['GIFT_BASKET']:
                if (ask <= tot + premium) and (buy_cpos_gb < POSITION_LIMIT['GIFT_BASKET']):
                    order_for = min(POSITION_LIMIT['GIFT_BASKET'] - buy_cpos_gb,-vol)
                    orders['GIFT_BASKET'].append(Order('GIFT_BASKET',ask,order_for))
                    buy_cpos_gb += order_for
        
        '''
        # one price version
        std = 76.4*0.5
        tot = mid_price['STRAWBERRIES']*6 + mid_price['CHOCOLATE']*4 + mid_price['ROSES']
        buy_cpos_gb = state.position.get('GIFT_BASKET',0)
        if (res_buy < -std) and (buy_cpos_gb < POSITION_LIMIT['GIFT_BASKET']): #coupon underpriced, buy now and sell later
            tot_order_for = 0
            for ask, vol in osell['GIFT_BASKET']:
                if (ask <= tot + premium) and (buy_cpos_gb < POSITION_LIMIT['GIFT_BASKET']):
                    order_for = min(POSITION_LIMIT['GIFT_BASKET'] - buy_cpos_gb,-vol)
                    buy_cpos_gb += order_for
                    tot_order_for += order_for
            orders['GIFT_BASKET'].append(Order('GIFT_BASKET',ask,tot_order_for))
        '''



        sell_cpos_gb = state.position.get('GIFT_BASKET',0)
        if (res_buy > std*0.5) and (res_buy < std*2.0)and (sell_cpos_gb > -POSITION_LIMIT['GIFT_BASKET']): #coupon overpriced, sell now and buy later
            for bid, vol in obuy['GIFT_BASKET']:
                if (bid >= tot + premium) and (sell_cpos_gb > -POSITION_LIMIT['GIFT_BASKET']):
                    order_for = min(POSITION_LIMIT['GIFT_BASKET'] + sell_cpos_gb,vol)
                    orders['GIFT_BASKET'].append(Order('GIFT_BASKET',bid,-order_for))
                    sell_cpos_gb += -order_for

        
        #market Make for GIFT BASKET
        # if the price is little bit off I am willing to buy/sell just a bit more
        if (abs(res_buy) >= std*2):
            if (res_buy <= -std*2) and (buy_cpos_gb < POSITION_LIMIT['GIFT_BASKET']): #coupon underpriced, buy at competitive price
                order_for = POSITION_LIMIT['GIFT_BASKET'] - buy_cpos_gb
                bid_price = math.floor(mid_price['GIFT_BASKET'])+1
                orders['GIFT_BASKET'].append(Order(p,bid_price,order_for))
                buy_cpos_gb += order_for

            if (res_buy >= std*2) and (sell_cpos_gb > -POSITION_LIMIT['GIFT_BASKET']): #coupon overpriced, sell at competitive price
                order_for = POSITION_LIMIT['GIFT_BASKET'] + sell_cpos_gb
                ask_price = math.ceil(mid_price['GIFT_BASKET'])-1
                orders['GIFT_BASKET'].append(Order(p,ask_price,-order_for))
                sell_cpos_gb += -order_for


        if res_sell_sb > trade_at_sb:
            vol = state.position.get('STRAWBERRIES',0) + POSITION_LIMIT['STRAWBERRIES']
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol)) 
                gb_neg -= vol
                #uku_pos += vol
        elif res_buy_sb < -trade_at_sb:
            vol = POSITION_LIMIT['STRAWBERRIES'] - state.position.get('STRAWBERRIES',0)
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol))
                gb_pos += vol
        

        if res_sell_cc > trade_at_cc:
            vol = state.position.get('CHOCOLATE',0) + POSITION_LIMIT['CHOCOLATE']
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol)) 
                gb_neg -= vol
                #uku_pos += vol
        elif res_buy_cc < -trade_at_cc:
            vol = POSITION_LIMIT['CHOCOLATE'] - state.position.get('CHOCOLATE',0)
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol))
                gb_pos += vol

        chocoprice = (mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES']*6 -mid_price['ROSES'] - 379.5)/4  
        logger.print(f'res;{res_buy_cc};chocopirce-mid;{chocoprice-mid_price["CHOCOLATE"]};tot;{chocoprice};mid_price;{mid_price["CHOCOLATE"]};orders;{orders["CHOCOLATE"]}')


        if res_sell_rs > trade_at_rs:
            vol = state.position.get('ROSES',0) + POSITION_LIMIT['ROSES']
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol)) 
                gb_neg -= vol
                #uku_pos += vol
        elif res_buy_rs < -trade_at_rs:
            vol = POSITION_LIMIT['ROSES'] - state.position.get('ROSES',0)
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol))
                gb_pos += vol

        


        return orders
    






