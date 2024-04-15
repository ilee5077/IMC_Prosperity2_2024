import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import pandas as pd
class Trader:
	def run(self, state):
		result = {}
		trader_data  = ''
		conversions = 0
		time = state.timestamp
		want_to_see_product = 'ORCHIDS'
		print('\n')
		print(f"timestamp: {state.timestamp}")
		#print("LISTING")
		#print(state.listings)
		print(f"buy_orders: {list(state.order_depths[want_to_see_product].buy_orders.items())}")
		print(f"sell_orders: {list(state.order_depths[want_to_see_product].sell_orders.items())}")
		if want_to_see_product in state.own_trades:
			print(f"own_trades: {state.own_trades[want_to_see_product]}")
		if want_to_see_product in state.position:
			print(f"position: {state.position[want_to_see_product]}")
		if want_to_see_product in state.market_trades:
			print(f"market_trades {state.market_trades[want_to_see_product]}")
		'''
		if time == 0:
			od = state.order_depths['ORCHIDS']
			buy_orders = list(od.buy_orders.items())
			buy_orders.sort(key = lambda x:x[0], reverse = True)
			sell_orders = list(od.sell_orders.items())
			sell_orders.sort(key = lambda x: x[0])
			best_bid = buy_orders[0][0]
			best_ask = sell_orders[0][0]
			result['ORCHIDS'] = [Order('ORCHIDS',best_bid,-2)]
			print(f"bidPrice:{state.observations.conversionObservations['ORCHIDS'].bidPrice}")
			print(f"askPrice:{state.observations.conversionObservations['ORCHIDS'].askPrice}")
			print(f"importTariff:{state.observations.conversionObservations['ORCHIDS'].importTariff}")
			print(f"exportTariff:{state.observations.conversionObservations['ORCHIDS'].exportTariff}")
			print(f"transportFees:{state.observations.conversionObservations['ORCHIDS'].transportFees}")
			conversions = 2
		if time == 100:
			
			print(f"bidPrice:{state.observations.conversionObservations['ORCHIDS'].bidPrice}")
			print(f"askPrice:{state.observations.conversionObservations['ORCHIDS'].askPrice}")
			print(f"importTariff:{state.observations.conversionObservations['ORCHIDS'].importTariff}")
			print(f"exportTariff:{state.observations.conversionObservations['ORCHIDS'].exportTariff}")
			print(f"transportFees:{state.observations.conversionObservations['ORCHIDS'].transportFees}")
		'''
		if time == 8000:
			result['ORCHIDS'] = [Order('ORCHIDS',1090,-100)]
			#result['ORCHIDS'].append(Order('ORCHIDS',1099,-98))
		print(result)
		return result, conversions, trader_data
		# pnl = qty*( local best_bid at ts 0) - qty*(conversion best_ask at ts = 100) - qty*(import tariff) - qty*(transport fees)
		# pnl = 2*1094 - 2*1099 - 2*(-5) - 2*(0.9) = -1.8 calculation explanation