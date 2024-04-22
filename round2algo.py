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
            mydata['running_buy_price'] = 0
            mydata['running_sell_price'] = 0
            mydata['running_buy_vol'] = 0
            mydata['running_sell_vol'] = 0
            traderData = jsonpickle.encode(mydata)
            mydata = jsonpickle.decode(traderData)
        else:
            mydata = jsonpickle.decode(state.traderData)
        
        if state.position.get("ORCHIDS",0) > 0:
            mydata['running_buy_price'] += 0.1
        elif state.position.get("ORCHIDS",0) == 0:
            mydata['running_buy_price'] = 0
            mydata['running_sell_price'] = 0
            mydata['running_buy_vol'] = 0
            mydata['running_sell_vol'] = 0

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
        
        logger.print(f'runbuyp;{mydata['running_buy_price']};runbuyq;{mydata['running_buy_vol']};runselp;{mydata['running_sell_price']};runselq;{mydata['running_sell_vol']};')

        if (mydata['running_buy_vol'] - mydata['running_sell_vol']) != 0:
            average_price = round((mydata['running_buy_vol'] * mydata['running_buy_price'] - mydata['running_sell_vol'] * mydata['running_sell_price']) / (mydata['running_buy_vol'] - mydata['running_sell_vol']),1)
        else:
            average_price = 0




        result['ORCHIDS'], conversions = self.compute_orders_orchids_new(state = state, mydata = mydata, average_price = average_price)
        conversions = None
        traderData = jsonpickle.encode(mydata)
        logger.print(f";order;{result}")

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
    
    
    def compute_orders_orchids_new(self, state:TradingState, mydata, average_price):
        p = 'ORCHIDS'

        orders: list[Order] = []

        buy_orders = list(state.order_depths[p].buy_orders.items())
        best_buy_pr, buy_vol = buy_orders[0]

        sell_orders = list(state.order_depths[p].sell_orders.items())
        best_sell_pr, sell_vol = sell_orders[0]

        mid_price = (best_buy_pr + best_sell_pr)/2
        pos_limt = mydata['position_limit'][p]
       

        buy_from_south = state.observations.conversionObservations[p].askPrice + state.observations.conversionObservations[p].importTariff + state.observations.conversionObservations[p].transportFees
        
        sell_to_south = state.observations.conversionObservations[p].bidPrice - state.observations.conversionObservations[p].exportTariff - state.observations.conversionObservations[p].transportFees

        sell_cpos = state.position.get(p,0)
        # I sell and buy from south
        if (buy_from_south < best_buy_pr) and (sell_cpos > -pos_limt):
            for bid, vol in buy_orders:
                if buy_from_south < bid:
                    order_for = min(pos_limt + sell_cpos,vol)
                    orders.append(Order(p,bid,-order_for))
                    sell_cpos += -order_for

        buy_cpos = state.position.get(p,0)
        # I buy and sell to south
        if (sell_to_south > best_sell_pr) and (buy_cpos < pos_limt):
            for ask, vol in sell_orders:
                if sell_to_south > ask:
                    order_for = min(pos_limt - buy_cpos,-vol)
                    orders.append(Order(p,ask,order_for))
                    buy_cpos += order_for

        # MM
        # I sell and buy from south
        if (buy_from_south < mid_price) and (sell_cpos > -pos_limt):
            order_for = pos_limt + sell_cpos
            price = min(int(math.ceil(buy_from_south)), int(math.floor(mid_price)))
            orders.append(Order(p,price,-order_for))
        # I buy and sell to south
        if (sell_to_south > mid_price) and (buy_cpos < pos_limt):
            order_for = pos_limt - buy_cpos
            price = max(int(math.floor(sell_to_south)), int(math.ceil(mid_price)))
            orders.append(Order(p,price,order_for))

        #if mid_price > average_price # buy at lower than avg or sell at higher than avg
        if mid_price < average_price and state.position.get(p,0) > 0:
            for ask,vol in sell_orders:
                if (ask < average_price) and (buy_cpos < pos_limt):
                    order_for = min(pos_limt - buy_cpos, -vol)
                    orders.append(Order(p,ask,order_for))
                    buy_cpos += order_for
            for bid,vol in buy_orders:
                if (bid > average_price) and (sell_cpos > -pos_limt):
                    order_for = min(pos_limt + sell_cpos,vol)
                    orders.append(Order(p,bid,-order_for))
                    sell_cpos += -order_for

        #if mid_price < average_price # sell at lower than avg or buy at higher than avg
        if mid_price > average_price and state.position.get(p,0) < 0:
            for ask,vol in sell_orders:
                if (ask > average_price) and (buy_cpos < pos_limt):
                    order_for = min(pos_limt - buy_cpos, -vol)
                    orders.append(Order(p,ask,order_for))
                    buy_cpos += order_for
                    logger.print(Order(p,ask,order_for))
            for bid,vol in buy_orders:
                if (bid < average_price) and (sell_cpos > -pos_limt):
                    order_for = min(pos_limt + sell_cpos,vol)
                    orders.append(Order(p,bid,-order_for))
                    sell_cpos += -order_for
                    logger.print(Order(p,bid,-order_for))

        logger.print(f"buy_from_south:{buy_from_south},sell_to_south:{sell_to_south},midprice:{mid_price},bestbuy:{best_buy_pr},bestsell:{best_sell_pr};average_price;{average_price}")

        # need a function incorporting average price to find how much to convert
        #conversions = -state.position.get(p,0)
        conversions = self.compute_conversion(state, average_price, buy_from_south, sell_to_south)
        


        return orders, conversions

    def compute_conversion(self, state:TradingState, average_price, buy_from_south, sell_to_south):
        p = 'ORCHIDS'
        cpos = state.position.get(p,0)
        conversions = 0
        if cpos > 0: #sell to south
            if average_price < sell_to_south:
                conversions = -cpos
        if cpos < 0: #buy from south
            if average_price > buy_from_south:
                conversions = -cpos
        
        return conversions
