class GridTradingStrategy:
    def __init__(self, initial_price, grid_size, num_grids, stop_loss, take_profit):
        self.initial_price = initial_price
        self.grid_size = grid_size
        self.num_grids = num_grids
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.current_price = initial_price

    def set_grid_levels(self):
        self.buy_levels = [self.initial_price - i * self.grid_size for i in range(1, self.num_grids + 1)]
        self.sell_levels = [self.initial_price + i * self.grid_size for i in range(1, self.num_grids + 1)]

    def execute_trade(self, action):
        if action == 'buy':
            self.current_price = self.buy_levels.pop(0)
            print(f"Buying at {self.current_price}")
        elif action == 'sell':
            self.current_price = self.sell_levels.pop(0)
            print(f"Selling at {self.current_price}")

    def risk_management(self):
        if self.current_price - self.initial_price >= self.take_profit:
            print("Take Profit Hit. Exiting position.")
            return True
        elif self.initial_price - self.current_price >= self.stop_loss:
            print("Stop Loss Hit. Exiting position.")
            return True
        return False

# 示例用法
initial_price = 1.1200
grid_size = 0.0010
num_grids = 5
stop_loss = 0.0020
take_profit = 0.0015

grid_trader = GridTradingStrategy(initial_price, grid_size, num_grids, stop_loss, take_profit)
grid_trader.set_grid_levels()

for _ in range(num_grids):
    grid_trader.execute_trade('buy')
    if grid_trader.risk_management():
        break

    grid_trader.execute_trade('sell')
    if grid_trader.risk_management():
        break
