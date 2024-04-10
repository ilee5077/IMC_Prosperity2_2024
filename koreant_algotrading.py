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
		cpos_update = cpos
		for ask, vol in sell_orders:
			# MATCHING ASK ORDERS
            # sell for less than the price we willing to buy OR
            # sell for same as we want to buy and our position is short
			if ((ask < acc_bid) or ((cpos<0) and (ask == acc_bid))) and cpos_update < pos_limt:
				mx_with_buy = max(mx_with_buy, ask) # buy price
				order_for = min(-vol, pos_limt - cpos) # how many do we buy for
				cpos_update += order_for
				assert(order_for >= 0)
				orders.append(Order(product, ask, order_for))

		mprice_actual = (best_sell_pr + best_buy_pr)/2
		mprice_ours = (acc_bid+acc_ask)/2

		undercut_buy = best_buy_pr + 1

		# buy orders
		if (cpos_update < pos_limt) and (cpos < 0): # short position
			num = min(40, pos_limt - cpos_update)
			orders.append(Order(product, min(undercut_buy + 1, acc_bid-1), num))
			cpos_update += num

		if (cpos_update < pos_limt) and (cpos > 15): # long position
			num = min(40, pos_limt - cpos_update)
			orders.append(Order(product, min(undercut_buy - 1, acc_bid-1), num))
			cpos_update += num

		if cpos_update < pos_limt: # position between 1 to 14
			num = min(40, pos_limt - cpos_update)
			orders.append(Order(product, min(undercut_buy, acc_bid-1), num))
			cpos_update += num

		cpos= 0
		if product in state.position.keys():
			cpos = state.position[product]
		cpos_update = cpos
		# short positioning
		for bid, vol in buy_orders:
			# MATCHING BUY ORDERS
			# someone is willing to buy more than what we ask
			# we are long and someone buying at price we think
			if ((bid > acc_ask) or ((cpos>0) and (bid == acc_ask))) and cpos_update > -pos_limt:
				order_for = max(-vol, -pos_limt-cpos_update)
				# order_for is a negative number denoting how much we will sell
				cpos_update += order_for
				assert(order_for <= 0)
				orders.append(Order(product, bid, order_for))

		undercut_sell = best_sell_pr - 1
		# placing orders
		if (cpos_update > -pos_limt) and (cpos > 0): # long position
			num = max(-40, -pos_limt-cpos_update) # sell 40 or 
			orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
			cpos_update += num

		if (cpos_update > -pos_limt) and (cpos < -15):
			num = max(-40, -pos_limt-cpos_update)
			orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
			cpos_update += num

		if cpos_update > -pos_limt:
			num = max(-40, -pos_limt-cpos_update)
			orders.append(Order(product, max(undercut_sell, acc_ask+1), num))
			cpos_update += num

		return orders
	
	def compute_orders_starfruit(self, state: TradingState, acc_bid, acc_ask, POSITION_LIMIT):
		product = 'STARFRUIT'
		
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
		cpos_update = cpos
		for ask, vol in sell_orders:
			# MATCHING ASK ORDERS
            # sell for less than the price we willing to buy OR
            # sell for same as we want to buy and our position is short
			if ((ask < acc_bid) or ((cpos<0) and (ask == acc_bid))) and cpos_update < pos_limt:
				mx_with_buy = max(mx_with_buy, ask) # buy price
				order_for = min(-vol, pos_limt - cpos) # how many do we buy for
				cpos_update += order_for
				assert(order_for >= 0)
				orders.append(Order(product, ask, order_for))

		mprice_actual = (best_sell_pr + best_buy_pr)/2
		mprice_ours = (acc_bid+acc_ask)/2

		undercut_buy = best_buy_pr + 1

		# buy orders
		if cpos_update < pos_limt:
			num = pos_limt - cpos_update
			orders.append(Order(product, min(undercut_buy, acc_bid), num))
			cpos_update += num

		cpos= 0
		if product in state.position.keys():
			cpos = state.position[product]
		cpos_update = cpos
		# short positioning
		for bid, vol in buy_orders:
			# MATCHING BUY ORDERS
			# someone is willing to buy more than what we ask
			# we are long and someone buying at price we think
			if ((bid > acc_ask) or ((cpos>0) and (bid == acc_ask))) and cpos_update > -pos_limt:
				order_for = max(-vol, -pos_limt-cpos_update)
				# order_for is a negative number denoting how much we will sell
				cpos_update += order_for
				assert(order_for <= 0)
				orders.append(Order(product, bid, order_for))

		undercut_sell = best_sell_pr - 1
		# placing orders
		if cpos_update > -pos_limt:
			num = -pos_limt-cpos_update
			orders.append(Order(product, max(undercut_sell, acc_ask), num))
			cpos_update += num

		return orders
	
	
	
	def run(self, state: TradingState):
		
		product_keys = [
			#'AMETHYSTS'
			'STARFRUIT']
		
		# Orders to be placed on exchange matching engine
		result = {}

		if state.traderData == '':
			mydata = {}
			mydata['position_limit'] = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
			mydata['starfruit_cache'] = []
			mydata['starfruit_dim'] = 5
		else:
			mydata = jsonpickle.decode(state.traderData)


		#Only method required. It takes all buy and sell orders for all symbols as an input,
		#and outputs a list of orders to be sent
		want_to_see_product = 'STARFRUIT'
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
		#print("traderData: " + state.traderData)
		#print("Observations: " + str(state.observations))

		

		#acc_bid = 9999, acc_ask = 10001
		result['AMETHYSTS'] = self.compute_orders_amethysts(state, acc_bid = 10000, acc_ask = 10000, POSITION_LIMIT = mydata['position_limit'])

		

		buy_orders = list(state.order_depths['STARFRUIT'].buy_orders.items())
		#print(f"buy_orders: {buy_orders}")
		best_buy_pr, buy_vol = buy_orders[0]

		# find sell orders
		sell_orders = list(state.order_depths['STARFRUIT'].sell_orders.items())
		#print(f"sell_orders: {sell_orders}")
		best_sell_pr, sell_vol = sell_orders[0]

		mydata['starfruit_cache'].append((best_buy_pr+best_sell_pr)/2.0)
		nxt_SF_price = 0
		if len(mydata['starfruit_cache']) > mydata['starfruit_dim']:
			mydata['starfruit_cache'].pop(0)
			nxt_SF_price = round(sum(mydata['starfruit_cache'])/float(len(mydata['starfruit_cache'])),1)
			gradient, intercept = np.polyfit(list(range(mydata['starfruit_dim'])),mydata['starfruit_cache'],1)
			nxt_SF_price2 = int(round(gradient * 6 + intercept,0))
			print(f"next price SF: {nxt_SF_price}, next price LR: {round(gradient * 6 + intercept,0)},history: {mydata['starfruit_cache']}, gradient: {gradient}")			
			# now we know the next SF price is going to be - so we can start trade!
			result['STARFRUIT'] = self.compute_orders_starfruit(state, acc_bid = int(math.floor(nxt_SF_price)), acc_ask = int(math.ceil(nxt_SF_price)), POSITION_LIMIT = mydata['position_limit'])
			
		#print(mydata['starfruit_cache'], nxt_SF_price)

		
		
		
		
		# String value holding Trader state data required. 
		# It will be delivered as TradingState.traderData on next execution.
		traderData = jsonpickle.encode(mydata) 
		
		# Sample conversion request. Check more details below. 
		conversions = 1


		result = {k: v for k, v in result.items() if k in (product_keys)}

		print(f"order: {result}")
		return result, conversions, traderData
	




