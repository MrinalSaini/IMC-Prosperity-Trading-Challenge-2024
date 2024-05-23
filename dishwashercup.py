import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import pandas as pd
import numpy as np
import math
from statistics import NormalDist

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
    buy_od = {} #buy side order depth for each product
    sell_od = {} #sell side order depth for each product
    pos_lim = {'AMETHYSTS':20, 'STARFRUIT':20, 'ORCHIDS':100, 'CHOCOLATE':350,'STRAWBERRIES':350,'ROSES':60,'GIFT_BASKET':60, 'COCONUT': 300, 'COCONUT_COUPON': 600}
    large_size = {'AMETHYSTS':2, 'STARFRUIT':3}
    pos = {}
    buys = {}
    sells = {}
    time = None

    mu_sf = 0
    curr_sf_price = None
    init_sf_price = None

    cur_humidity = None
    prev_humidity = None
    cur_exp_tar = None
    prev_exp_tar = None
    cur_sunlight = None

    cur_humidity = None
    prev_humidity = None
    cur_exp_tar = None
    prev_exp_tar = None


    REQ_HEDGE = {'CHOCOLATE':0,'STRAWBERRIES':0,'ROSES':0,'GIFT_BASKET':0}
    basket_std = 76.42
    
    rs_bid_max = -1
    rs_ask_min = 1e9

    olivia_buy_detect = False
    olivia_sell_detect = False
    pos_goal = 0

    prev_coco_price = 10_000
    prev_coup_price = 675


    def init_vars(self, state: TradingState):
        self.time = state.timestamp
        for product in state.order_depths.keys():
            cpos = state.position.get(product,0)
            self.pos[product] = cpos
            od = state.order_depths[product]
            buy_orders = list(od.buy_orders.items())
            buy_orders.sort(key = lambda x:x[0], reverse = True)
            sell_orders = list(od.sell_orders.items())
            sell_orders.sort(key = lambda x: x[0])
            self.buy_od[product] = buy_orders
            self.sell_od[product] = sell_orders


        if self.time!=0:
            self.curr_sf_price,self.init_sf_price,self.prev_humidity,self.prev_exp_tar, self.prev_imp_tar, self.exp_signal, self.imp_signal, self.hum_signal, self.REQ_HEDGE, self.pos_goal, self.prev_coco_price, self.prev_coup_price = jsonpickle.decode(state.traderData)
            # self.REQ_HEDGE, self.pos_goal = jsonpickle.decode(state.traderData)


    def blackscholes(self, day, state, s0):
        sigma = 0.16
        K = 10_000
        T = ((250-day) * 1_000_000 - state.timestamp) / 252_000_000
        d1 = (np.log(s0 / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = s0 * NormalDist().cdf(d1) - K * NormalDist().cdf(d2)
        return call_price   

    def get_osize(self, state, product):
        osize_a = self.pos_lim[product] + state.position.get(product,0) - self.sells.get(product,0)
        osize_b = self.pos_lim[product] - state.position.get(product,0) - self.buys.get(product,0)
        return osize_b, osize_a

    def sniper(self, state, product, fair_price, side = None):
        orders = []
        buy_orders = self.buy_od[product]
        sell_orders = self.sell_od[product]
        osize_bid, osize_ask = self.get_osize(state, product)
        
        if side != 'S': #exploit bid inefficiencies
            for prices,volumes in buy_orders:
                if prices >= fair_price:
                    sell_amt = min(volumes,osize_ask)
                    orders.append(Order(product, prices, -sell_amt))
                    self.sells[product] = self.sells.get(product,0) + sell_amt
                    osize_ask -= sell_amt
                    self.pos[product] -= sell_amt
                    if sell_amt == volumes:
                        buy_orders.pop(0)
                else:
                    break
            self.buy_od[product] = buy_orders

        if side != 'B': #exploit ask inefficiencies
            for prices,volumes in sell_orders:
                if prices <= fair_price:
                    buy_amt = min(abs(volumes),osize_bid)
                    orders.append(Order(product, prices, buy_amt ))
                    self.buys[product] = self.buys.get(product,0) + buy_amt
                    osize_bid -= buy_amt
                    self.pos[product] += buy_amt
                    if buy_amt == abs(volumes):
                        sell_orders.pop(0)
                else:
                    break
            self.sell_od[product] = sell_orders
        return orders

    def balancer(self, state, product, fair_price, side = 'None', tol = 0):
        orders = []
        buy_orders = self.buy_od[product]
        sell_orders = self.sell_od[product]
        osize_bid, osize_ask = self.get_osize(state, product)
        
        if side != 'S': #exploit bid inefficiencies
            for prices,volumes in buy_orders:
                if prices >= fair_price:
                    sell_amt = min(abs(self.pos[product]) - tol,volumes,osize_ask) # sell_amt = min(abs(self.pos[product]) + tol,volumes,osize_ask)
                    orders.append(Order(product, prices, -sell_amt))
                    self.sells[product] = self.sells.get(product,0) + sell_amt
                    osize_ask -= sell_amt
                    self.pos[product] -= sell_amt
                    if sell_amt == volumes:
                        buy_orders.pop(0)
                else:
                    break
            self.buy_od[product] = buy_orders

        if side != 'B': #exploit ask inefficiencies
            for prices,volumes in sell_orders:
                if prices <= fair_price:
                    buy_amt = min(abs(self.pos[product]) - tol, abs(volumes),osize_bid) # buy_amt = min(abs(self.pos[product]) + tol, abs(volumes),osize_bid)
                    orders.append(Order(product, prices, buy_amt ))
                    self.buys[product] = self.buys.get(product,0) + buy_amt
                    osize_bid -= buy_amt
                    self.pos[product] += buy_amt
                    if buy_amt == abs(volumes):
                        sell_orders.pop(0)
                else:
                    break
            self.sell_od[product] = sell_orders
        return orders


    def get_best_price(self, product, fair_price):

        large_size = self.large_size[product]

        if product == 'STARFRUIT':
            buy_orders = self.buy_od[product] if len(self.buy_od[product]) > 0 else [[int(self.curr_sf_price) - 3, 20]]
            sell_orders = self.sell_od[product] if len(self.sell_od[product]) > 0 else [[int(self.curr_sf_price) + 3, 20]]
            acc_bid = math.floor(fair_price) - 3
            acc_ask = math.ceil(fair_price) + 3
        
        if product == 'AMETHYSTS':
            buy_orders = self.buy_od[product] if len(self.buy_od[product]) > 0 else [[9995,20]]
            sell_orders = self.sell_od[product] if len(self.sell_od[product]) > 0 else [[10005,20]]
            acc_bid = math.floor(fair_price) - 3
            acc_ask = math.ceil(fair_price) + 3
        
        thresh = 5

        for prices,volumes in sell_orders:
            if prices <= fair_price + 2:
              if self.pos[product] < -thresh:
                acc_ask = prices
                break
            if prices > fair_price + 1: 
                if abs(volumes) < large_size + 1 and self.pos[product] < -thresh: # if abs(volumes) < large_size + 1 and self.pos[product] < -thresh:
                    acc_ask = prices
                else:
                    acc_ask = prices - 1
                break

        for prices, volumes in buy_orders:
            if prices >= fair_price - 2:
              if self.pos[product] > thresh:
                acc_bid = prices
                break
            if prices < fair_price - 1:
                if abs(volumes) < large_size + 1 and self.pos[product] > thresh:
                    acc_bid = prices
                else:
                    acc_bid = prices + 1
                break
            
        return int(acc_bid), int(acc_ask)

    def get_fair_price(self, product):
        if product == 'AMETHYSTS':
            return 10_000

        if product in ['STARFRUIT', 'ORCHIDS']:
            buy_orders = self.buy_od[product]
            sell_orders = self.sell_od[product]

            mx_buy_pr = max(buy_orders,key = lambda x:x[1])[0] if max(buy_orders,key = lambda x:x[1])[1] >= 20 else min(buy_orders,key = lambda x:x[0])[0] - 1
            mx_sell_pr = max(sell_orders,key = lambda x:-x[1])[0] if -max(sell_orders,key = lambda x:-x[1])[1] >= 20 else max(sell_orders, key = lambda x:x[0])[0] + 1 
            return (mx_buy_pr + mx_sell_pr)/2

        if product in ['CHOCOLATE', 'STRAWBERRIES','ROSES','GIFT_BASKET', 'COCONUT', 'COCONUT_COUPON']:
            buy_orders = self.buy_od[product]
            sell_orders = self.sell_od[product]

            best_bid = buy_orders[0][0] 
            best_ask = sell_orders[0][0]
            return (best_bid + best_ask)/2


    def SF_taker(self, state):

        product = 'STARFRUIT'
        self.curr_sf_price = self.get_fair_price(product)

        if self.time == 0:
            self.init_sf_price = self.curr_sf_price

        mu = 0
        fair_price = self.curr_sf_price*np.exp(mu)
        acc_ask = math.ceil(fair_price) + 1
        acc_bid = math.floor(fair_price) - 1 
        orders = []
        pos_tol = 12
        pos_tol_b = 2 
        offload_lim = 3
        
        orders = orders + self.sniper(state, product, acc_ask, 'B') + self.sniper(state, product, acc_bid, 'S')
        
        if self.pos[product] > offload_lim:
            orders = orders + self.balancer(state, product, math.ceil(fair_price), 'B', pos_tol_b)
        if self.pos[product] < -offload_lim:
            orders = orders + self.balancer(state, product, math.floor(fair_price), 'S', pos_tol_b)

        self.mu_sf = mu
        return orders

    def SF_maker(self, state):
        product = 'STARFRUIT'
        orders = []
        fair_price = self.curr_sf_price*np.exp(self.mu_sf)
        acc_bid, acc_ask = self.get_best_price(product, fair_price)

        acc_ask = max(math.ceil(fair_price),acc_ask)
        acc_bid = min(math.floor(fair_price),acc_bid)

        osize_bid, osize_ask = self.get_osize(state, product)
        orders.append(Order(product,acc_ask,-osize_ask))
        orders.append(Order(product,acc_bid, osize_bid))
        return orders

        
    def AM_taker(self, state):
        product = 'AMETHYSTS'
        fair_price = self.get_fair_price(product)
        orders = []
        pos_tol = 3
        pos_tol_b = 3
        orders = orders + self.sniper(state, product, fair_price + 2, 'B') + self.sniper(state, product, fair_price - 2, 'S')

        if self.pos[product] >= pos_tol:
            orders = orders + self.balancer(state, product,fair_price,'B', pos_tol_b) #we want to sell and offload inventory # orders = orders + self.balancer(state, product,fair_price,'B')

        elif self.pos[product] <= -pos_tol:
            orders = orders + self.balancer(state, product, fair_price, 'S', pos_tol_b) #we want to buy and offload inventory # orders = orders + self.balancer(state, product, fair_price, 'S')
        
        return orders

    def AM_maker(self, state):
        orders = []
        product = 'AMETHYSTS'
        fair_price = self.get_fair_price(product)
        acc_bid, acc_ask = self.get_best_price(product, fair_price)
        osize_bid, osize_ask = self.get_osize(state, product)
        
        orders.append(Order(product,acc_ask,-osize_ask))
        orders.append(Order(product,acc_bid, osize_bid))

        return orders

    def check_present_arb(self,state,product,c_bid,c_ask,import_tariff,export_tariff,transport_fees):
        orders = []
        buy_orders = self.buy_od[product]
        sell_orders = self.sell_od[product]
        c_buy_price = c_ask + import_tariff + transport_fees
        osize_bid, osize_ask = self.get_osize(state,product)
        conversions = 0
        if self.time == 0:
            orders.append(Order(product,buy_orders[0][0],-buy_orders[0][1]))
            return 0, orders
        if self.time == 999_900: #TODO: Set to 999_900 before submission
            for prices,volumes in sell_orders:
                if prices < c_buy_price:
                    buy_amt = min(abs(self.pos[product]) - tol, abs(volumes),osize_bid) # buy_amt = min(abs(self.pos[product]) + tol, abs(volumes),osize_bid)
                    orders.append(Order(product, prices, buy_amt ))
                    self.buys[product] = self.buys.get(product,0) + buy_amt
                    osize_bid -= buy_amt
                    self.pos[product] += buy_amt
                    if buy_amt == abs(volumes):
                        sell_orders.pop(0)
                else:
                    break
            conversions = -self.pos[product]
        else:
            if self.pos[product] == 0:
                for prices,volumes in buy_orders:
                    sell_amt = min(volumes,osize_ask) 
                    orders.append(Order(product, prices, -sell_amt))
                    self.sells[product] = self.sells.get(product,0) + sell_amt
                    osize_ask -= sell_amt
                    self.pos[product] -= sell_amt
                    if sell_amt == volumes:
                        buy_orders.pop(0)
                    break
            elif self.pos[product] < 0:
                for prices,volumes in buy_orders:
                    if prices > c_buy_price: #arb exists
                        sell_amt = min(volumes,osize_ask) 
                        conversions += sell_amt
                        orders.append(Order(product, prices, -sell_amt))
                        self.sells[product] = self.sells.get(product,0) + sell_amt
                        osize_ask -= sell_amt
                        self.pos[product] -= sell_amt
                        if sell_amt == volumes:
                            buy_orders.pop(0)
                    else:
                        break
            return conversions,orders

    def check_future_arb(self,state,product,c_bid,c_ask,import_tariff,export_tariff,transport_fees, margin):
        orders = []
        buy_orders = self.buy_od[product]
        sell_orders = self.sell_od[product]
        best_bid = buy_orders[0][0]
        c_buy_price = c_ask + import_tariff + transport_fees
        curr_position = self.pos[product]
        cspread = c_ask - c_bid

        osize_ask = 100
        fair_price = self.get_fair_price(product)
        conversions = -self.pos[product]

        # check inefficiencies
        for bid_price, bid_amount in buy_orders:
            if bid_price > math.ceil(c_buy_price):
                sell_amount = min(abs(bid_amount), osize_ask)
                curr_position -= sell_amount
                if sell_amount == 0:
                    break
                self.sells[product] = self.sells.get(product, 0) + sell_amount
                osize_ask -= sell_amount
                orders.append(Order(product, bid_price, -sell_amount))

        if math.ceil(c_bid)-1-c_buy_price >= margin:
            orders.append(Order(product, math.ceil(c_bid)-1, -osize_ask))
        elif math.floor(c_bid)-c_buy_price >= margin:
            orders.append(Order(product, math.floor(c_bid), -osize_ask))
        else:
            if margin == 0:
                orders.append(Order(product, math.ceil(c_bid) + margin, -osize_ask))
            
        return conversions, orders

    def update_hum_signal(self):

        delta_humidity = self.cur_humidity - self.prev_humidity

        THRESH = 90
        if self.cur_humidity >= THRESH:
            if delta_humidity > 0:
                self.hum_signal = 1
            if delta_humidity < 0:
                self.hum_signal = -1
        else:
            self.hum_signal = 0

    def update_exp_tar_signal(self):

        delta_export = self.cur_exp_tar - self.prev_exp_tar

        THRESH = 12
        if self.cur_exp_tar >= THRESH:
            if delta_export > 0:
                self.exp_signal = 1
            if delta_export < 0:
                self.exp_signal = -1
        else:
            self.exp_signal = 0

    def update_imp_tar_signal(self):

        delta_import = self.cur_imp_tar - self.prev_imp_tar

        THRESH = -2.6
        if self.cur_imp_tar <= THRESH:
            if delta_import < 0:
                self.imp_signal = -1
            if delta_import > 0:
                self.imp_signal = 1
        else:
            self.imp_signal = 0

    def get_arb(self,state):
        product = 'ORCHIDS'
        c_bid = state.observations.conversionObservations['ORCHIDS'].bidPrice
        c_ask = state.observations.conversionObservations['ORCHIDS'].askPrice
        import_tariff = state.observations.conversionObservations['ORCHIDS'].importTariff
        export_tariff = state.observations.conversionObservations['ORCHIDS'].exportTariff
        transport_fees = state.observations.conversionObservations['ORCHIDS'].transportFees
        humidity = state.observations.conversionObservations['ORCHIDS'].humidity
        sunlight = state.observations.conversionObservations['ORCHIDS'].sunlight
        self.cur_humidity = humidity
        self.cur_exp_tar = export_tariff
        self.cur_imp_tar = import_tariff
        if self.time == 0:
            self.prev_humidity = humidity
            self.prev_exp_tar = export_tariff
            self.prev_imp_tar = import_tariff
            self.exp_signal = 0
            self.imp_signal = 0
            self.hum_signal = 0

        # update the signals
        self.update_exp_tar_signal()
        self.update_imp_tar_signal()
        self.update_hum_signal()

        # aggregating the signals to margin
        margin = 0
        conf_short = False
        if self.exp_signal == 1:
            margin = 1
        elif self.exp_signal == -1:
            conf_short = True
        elif self.imp_signal == 1:
            margin = 1
        elif self.hum_signal == 1:
            margin = 1
        elif self.hum_signal == -1 and self.imp_signal == -1:
            conf_short = True
        else:
            margin = 0.3

        
        if conf_short:

            orders = list()

            osize_bid, osize_ask = self.get_osize(state,product)

            conversions, pres_arb_orders = self.check_present_arb(state,product,c_bid,c_ask,import_tariff,export_tariff,transport_fees)
            orders.extend(pres_arb_orders)

        else:
            conversions, orders = self.check_future_arb(state,product,c_bid,c_ask,import_tariff,export_tariff,transport_fees, margin)

        self.prev_humidity = self.cur_humidity
        self.prev_exp_tar = self.cur_exp_tar
        self.prev_exp_tar = self.cur_exp_tar
        self.prev_imp_tar = self.cur_imp_tar
        return conversions, orders

    def find_rhianna(self,state):
        product = 'ROSES'
        orders = []
        buy_orders = self.buy_od[product]
        sell_orders = self.sell_od[product]
        osize_bid, osize_ask = self.get_osize(state,product)

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]
        pos = self.pos[product]

        
        mt = state.market_trades
        for trade in mt.get('ROSES', []):
            if trade.seller=='Rhianna' and trade.timestamp == self.time - 100:
                # get to -limit
                self.pos_goal = -1

            if trade.buyer=='Rhianna' and trade.timestamp == self.time - 100:
                # get to +limit
                self.pos_goal = 1

        if self.pos_goal > 0:
            orders.append(Order(product,best_ask + 1,osize_bid))
        elif self.pos_goal < 0:
            orders.append(Order(product,best_bid - 1,-osize_ask))
        else:
            # back to zero if wrong position taken
            if pos > 0:
                qty = min(pos,osize_ask)
                orders.append(Order(product,best_bid - 1,-qty))
            elif pos < 0:
                qty = min(-pos,osize_bid)
                orders.append(Order(product,best_ask + 1,qty))                

        return orders


    def basket_strat(self, state, product):
        curr_position = self.pos[product]
        self.buys[product]=0
        self.sells[product]=0
        orders = []
        ch_price = self.get_fair_price('CHOCOLATE')
        sb_price = self.get_fair_price('STRAWBERRIES')
        rs_price = self.get_fair_price('ROSES')
        gb_price = self.get_fair_price('GIFT_BASKET')
        osize_bid, osize_ask = self.get_osize(state,product)

        buy_orders = self.buy_od['GIFT_BASKET']
        sell_orders = self.sell_od['GIFT_BASKET']
        best_bid, bbvol = buy_orders[0]
        best_ask, bavol = sell_orders[0]
        worst_bid = buy_orders[-1][0]
        worst_ask = sell_orders[-1][0]

        underlying = 6*sb_price + 4*ch_price + rs_price + 379
        gap = gb_price - underlying
        width = 25
        band = width - 15

        c = 0.1
        exponent = 2
        qty = round(c * (abs(gap) ** exponent))

        if gap >= width - band and curr_position > 0:
            sell_amt = min(qty,curr_position,osize_ask, bbvol)
            curr_position -= sell_amt
            orders.append(Order(product, best_bid, -sell_amt))

        elif gap >= width and curr_position <=0:
            sell_amt = min(qty,osize_ask)
            curr_position -= sell_amt
            orders.append(Order(product, best_bid - 1, -sell_amt))

        elif gap <= -width + band and curr_position < 0:
            buy_amt = min(qty,-curr_position,osize_bid, -bavol)
            curr_position += buy_amt
            orders.append(Order(product, best_ask, buy_amt))

        elif gap <= -width and curr_position >= 0:  
            buy_amt = min(qty,osize_bid)
            curr_position += buy_amt
            orders.append(Order(product, best_ask + 1, buy_amt))

        return orders

    def hedge(self, state, product):
        curr_position = self.pos[product]
        orders = []
        buy_orders = self.buy_od[product]
        sell_orders = self.sell_od[product]
        
        if self.REQ_HEDGE[product] > curr_position:
            for price, qty in sell_orders:
                buy_amt = min(-qty, self.REQ_HEDGE[product]-curr_position)
                orders.append(Order(product, price, buy_amt))
                curr_position += buy_amt
                if curr_position == self.REQ_HEDGE[product]: break       
        
        elif self.REQ_HEDGE[product] < curr_position:
            for price, qty in buy_orders:
                sell_amt = min(qty, -(self.REQ_HEDGE[product]-curr_position))
                orders.append(Order(product, price, -sell_amt))
                curr_position -= sell_amt
                if curr_position == self.REQ_HEDGE[product]: break
        return orders


    def coconut_n_coupon_pairs_trading(self, state):

        product = 'COCONUT_COUPON'
        hedge_product = 'COCONUT'

        curr_position_cc = self.pos[product]
        curr_position_c = self.pos[hedge_product]

        self.buys[product]=0
        self.sells[product]=0

        orders_cc = list()
        orders_c = list()

        osize_bid_c, osize_ask_c = self.get_osize(state,hedge_product)
        buy_orders_c = self.buy_od[hedge_product]
        sell_orders_c = self.sell_od[hedge_product]

        osize_bid_cc, osize_ask_cc = self.get_osize(state,product)
        buy_orders_cc = self.buy_od[product]
        sell_orders_cc = self.sell_od[product]

        #coco safety
        if len(buy_orders_c)!=0:
            best_bid_c, bbvol_c = buy_orders_c[0]
        else:
            best_bid_c, bbvol_c = [self.prev_coco_price - 1, 100]
        if len(sell_orders_c)!= 0:
            best_ask_c, bavol_c = sell_orders_c[0]
        else:    
            best_ask_c, bavol_c = [self.prev_coco_price + 1, 100]
        coco_fair_price = (best_bid_c + best_ask_c)/2

        #option safety
        if len(buy_orders_cc)!=0:
            best_bid_cc, bbvol_cc = buy_orders_cc[0]
        else:
            best_bid_cc, bbvol_cc = [self.prev_coup_price - 1, 40]
        if len(sell_orders_cc)!= 0:
            best_ask_cc, bavol_cc = sell_orders_cc[0]
        else:    
            best_ask_cc, bavol_cc = [self.prev_coup_price + 1, 40]
        coup_fair_price = (best_bid_cc + best_ask_cc)/2

        if self.time == 0:
            self.prev_coco_price = coco_fair_price
            self.prev_coup_price = coup_fair_price

        mean_iv = 8729
        iv_dev = (2*coup_fair_price - coco_fair_price) + mean_iv
        delta = 0.5

        day = 4 # TODO: CHANGE TO 4 FOR SUBMISSION
        bs_price = self.blackscholes(day, state, coco_fair_price)

        iv_width = 19
        misprice_tol = 10
            
        if (iv_dev >= iv_width) or (bs_price < coup_fair_price - misprice_tol):

            sell_amt = min(bbvol_cc,osize_ask_cc)
            buy_amt = round(min(-bavol_c,osize_bid_c, sell_amt*delta))

            orders_cc.append(Order(product, best_bid_cc, -sell_amt))
            orders_c.append(Order(hedge_product, best_ask_c, buy_amt))

        elif (iv_dev <= -iv_width) or (bs_price > coup_fair_price + misprice_tol):

            buy_amt = min(-bavol_cc,osize_bid_cc)
            sell_amt = round(min(bbvol_c,osize_ask_c, buy_amt*delta))

            orders_cc.append(Order(product, best_ask_cc, buy_amt))
            orders_c.append(Order(hedge_product, best_bid_c, -sell_amt))

        self.prev_coco_price = coco_fair_price
        self.prev_coup_price = coup_fair_price
        
        return orders_cc, orders_c


    def run(self, state: TradingState):
        result = {}
        conversions = 0
        trader_data = ''
        self.init_vars(state)

        self.buys = {}
        self.sells = {}
        hedge = False
        antihedge = False
        result['ROSES'] = self.find_rhianna(state)
        result['GIFT_BASKET'] = self.basket_strat(state, "GIFT_BASKET")


        result['AMETHYSTS'] = self.AM_taker(state) + self.AM_maker(state)
        result['STARFRUIT'] = self.SF_taker(state) + self.SF_maker(state)

        conversions, result['ORCHIDS'] = self.get_arb(state)

        result['COCONUT_COUPON'], result['COCONUT'] = self.coconut_n_coupon_pairs_trading(state)

        trader_data = jsonpickle.encode([self.curr_sf_price,self.init_sf_price,self.prev_humidity,self.prev_exp_tar,self.prev_imp_tar, self.exp_signal, self.imp_signal, self.hum_signal, self.REQ_HEDGE, self.pos_goal, self.prev_coco_price, self.prev_coup_price]) 

        # trader_data = jsonpickle.encode([self.REQ_HEDGE, self.pos_goal]) 

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data