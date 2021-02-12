import random, copy, math, time

log_2 = {0: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8, 512: 9, 1024: 10, 2048: 11, 4096: 12, 8192: 13, 16384: 14, 32768: 15, 65536: 16}
exp_2 = {0: 0, 1: 2, 2: 4, 3: 8, 4: 16, 5: 32, 6: 64, 7: 128, 8: 256, 9: 512, 10: 1024, 11: 2048, 12: 4096, 13: 8192, 14: 16384, 15: 32768, 16: 65536}

class Board():
    def __init__(self):
        self.board = [0] * 16    # stores 16 blocks of the board in a array 
        self.score = 0    # current score of the game
   
        # state -> binary representation of a row of a board
        # next_state -> binary representation of a row after a move is applied
        self.next_state = [] 
        self.reward = []    # increment in the score after going in the next state
        self.moved = []    # tells whether states change after performing the move
        
        # 65535 is total number of states possible.
        for x in range(65535):
            next_state, reward, moved = self.slide(self.to_row(x))
            #print(self.to_row(x), next_state, reward, moved)

            self.next_state.append(next_state)
            self.reward.append(reward)
            self.moved.append(moved)

        self.possible_moves = self.get_all_move()
    
    def to_row(self, x):
        '''
        converts the state to a row of a board
        where Empty,2,4,8... mapped to 0,1,2,3...
        respectively.
        '''
        row = [0] * 4

        for i in range(4):
            row[3-i] = x % 16
            x = x // 16

        return row

    def slide(self, row):
        '''
        implements left move on a row of board
        and also calculate the score i.e sum of the merged tiles
        and lastly any of block is moved or not.
        '''
        
        new_row = []
        score = 0
        moved = False

        flag = 1
        for y in range(4):
            if row[y] != 0:
                if len(new_row) == 0:
                    new_row.append(row[y])
                else:
                    if flag == 1 and row[y] == new_row[-1]:
                        new_row[-1] += 1
                        score += exp_2[new_row[-1]]
                        flag = 0
                    else:
                        new_row.append(row[y])
                        if flag == 0: flag = 1

        for y in range(4):
            try:
                if row[y] != new_row[y]:
                    moved = True
                row[y] = new_row[y]
            except:
                if row[y] != 0:
                    moved = True
                row[y] = 0

        return row, score, moved

    def add_number(self):
        empty_space_list = []
        for i in range(16):
            if self.board[i] == 0:
                empty_space_list.append(i)

        x = random.choice(empty_space_list)
        y = random.choice([2,2,2,2,2,2,2,2,2,4])

        self.board[x] = y

    def move(self, action):
        if action == 'l':
            for x in range(4):
                state = (log_2[self.board[4*x]]<<12) + (log_2[self.board[4*x+1]]<<8) + (log_2[self.board[4*x+2]]<<4) + (log_2[self.board[4*x+3]])

                next_row = self.next_state[state]
                self.score += self.reward[state]

                for y in range(4):
                    self.board[4*x+y] = exp_2[next_row[y]]

        elif action == 'r':
            for x in range(4):
                state = (log_2[self.board[4*x+3]]<<12) + (log_2[self.board[4*x+2]]<<8) + (log_2[self.board[4*x+1]]<<4) + (log_2[self.board[4*x]])

                next_row = self.next_state[state]
                self.score += self.reward[state]

                for y in range(4):
                    self.board[4*x+y] = exp_2[next_row[3-y]]

        elif action == 'u':
            for y in range(4):
                state = (log_2[self.board[y]]<<12) + (log_2[self.board[4+y]]<<8) + (log_2[self.board[8+y]]<<4) + (log_2[self.board[12+y]])

                next_row = self.next_state[state]
                self.score += self.reward[state]

                for x in range(4):
                    self.board[4*x+y] = exp_2[next_row[x]]

        elif action == 'd':
            for y in range(4):
                state = (log_2[self.board[12+y]]<<12) + (log_2[self.board[8+y]]<<8) + (log_2[self.board[4+y]]<<4) + (log_2[self.board[y]])

                next_row = self.next_state[state]
                self.score += self.reward[state]

                for x in range(4):
                    self.board[4*x+y] = exp_2[next_row[3-x]]
                    
        else:
            print("Invalid Move")

        return action, self.score, self.is_over()

    def get_all_move(self):
        self.possible_moves = []

        for action in ['l', 'r', 'u', 'd']:
            flag = 0
            if action == 'l':
                for x in range(4):  
                    state = (log_2[self.board[4*x]]<<12) + (log_2[self.board[4*x+1]]<<8) + (log_2[self.board[4*x+2]]<<4) + (log_2[self.board[4*x+3]])
                    #print(action, self.to_row(state), self.moved[state])
                    if self.moved[state]:
                        flag = 1
                        break
     
            elif action == 'r':
                for x in range(4):
                    state = (log_2[self.board[4*x+3]]<<12) + (log_2[self.board[4*x+2]]<<8) + (log_2[self.board[4*x+1]]<<4) + (log_2[self.board[4*x]])
                    #print(action, self.to_row(state), self.moved[state])
                    if self.moved[state]:
                        flag = 1
                        break

            elif action == 'u':
                for y in range(4):
                    state = (log_2[self.board[y]]<<12) + (log_2[self.board[4+y]]<<8) + (log_2[self.board[8+y]]<<4) + (log_2[self.board[12+y]])
                    #print(action, self.to_row(state), self.moved[state])
                    if self.moved[state]:
                        flag = 1
                        break

            else:
                for y in range(4):
                    state = (log_2[self.board[12+y]]<<12) + (log_2[self.board[8+y]]<<8) + (log_2[self.board[4+y]]<<4) + (log_2[self.board[y]])
                    #print(action, self.to_row(state), self.moved[state])
                    if self.moved[state]:
                        flag = 1
                        break
                    
            #print(flag)
            if flag != 0:
                self.possible_moves.append(action)

        return self.possible_moves

    def is_over(self):
        if len(self.get_all_move()) == 0:
            return True
        else:
            return False

    def reset(self):
        self.board = [0] * 16

        self.score = 0
        
        self.add_number()
        self.add_number()
        self.possible_moves = self.get_all_move()

    def draw_board(self):
        print('---------')
        for x in range(4):
            print(str(self.board[4*x])+'\t'+str(self.board[4*x+1])+'\t'+str(self.board[4*x+2])+'\t'+str(self.board[4*x+3]))
        print('---------')

class ExpectiMax:
    def __init__(self, depth):
        self.depth = depth
        self.stats = [0] * depth

    def expectimax(self, board, depth, maximize=True):
        if board.is_over():
            return (-1, -10000)
        
        if depth==0:
            return (-1, board.score + self.heuristic(board))

        if maximize:
            score = -100000
            best_move = -1
            possible_move = board.possible_moves.copy()
            for x in possible_move:
                prev_board = board.board.copy()
                prev_score = board.score
                
                board.move(x)
                _, val = self.expectimax(board, depth-1, False)

                if val > score:
                    score = val
                    best_move = x

                self.stats[depth-1] += 1

                board.board = prev_board.copy()
                board.score = prev_score

            return (best_move, score)

        else:
            # early stopping
            if (self.depth - depth >= 5) and (board.board.count(0) >= 3):
                return (-1, board.score + self.heuristic(board))
        
            if (self.depth - depth >= 3) and (board.board.count(0) >= 6):
                return (-1, board.score + self.heuristic(board))
            
            possible_move = []
            for i in range(16):
                if board.board[i] == 0:
                        possible_move.append(i)

            score = 100000
            for x in possible_move:                
                board.board[x] = 2
                _, val1 = self.expectimax(board, depth-1, True)

                board.board[x] = 4
                _, val2 = self.expectimax(board, depth-1, True)

                val = 0.9 * val1 + 0.1 * val2
                score = min(score, val)
                
                self.stats[depth-1] += 1

                board.board[x] = 0

            return (-1, score)

    def heuristic(self, board):
        point = 0

        for i in range(4):
            for j in range(4): 
                x = board.board[4*i + j]
                if i > 0:
                    y = board.board[4*i+j-4]
                    point -= 2.5 * abs(x - y)
                if j > 0:
                    y = board.board[4*i+j-1]
                    point -= 2.5 * abs(x - y)
                
                if x == 0:
                    point += 0.05

        # monotonicity 
        monotonic_up = 0
        monotonic_down = 0
        monotonic_left = 0
        monotonic_right = 0

        for x in range(4):
            current = 0
            next = current + 1
            while next < 4:
                while ((board.board[x + 4 * next] == 0) and (next < 3)):
                    next += 1

                current_cell = board.board[x + 4 * current]
                current_value = log_2[current_cell]

                next_cell = board.board[x + 4 * next]
                next_value = log_2[next_cell]

                if current_value < next_value:
                    monotonic_up += (next_value - current_value)
                
                if next_value < current_value:
                    monotonic_down += (current_value - next_value)
                
                current = next
                next += 1

        for x in range(4):
            current = 0
            next = current + 1
            while next < 4:
                while ((board.board[4 * x + next] == 0) and (next < 3)):
                    next += 1

                current_cell = board.board[4 * x + current]
                current_value = log_2[current_cell]

                next_cell = board.board[4 * x + next]
                next_value = log_2[next_cell]

                if current_value < next_value:
                    monotonic_left += (next_value - current_value)
                
                if next_value < current_value:
                    monotonic_right += (current_value - next_value)
                
                current = next
                next += 1

        point += (max(monotonic_up, monotonic_down) + max(monotonic_left, monotonic_right))

        return point

score_list = []
depth = 7
board = Board()
expectimax = ExpectiMax(depth)

for _ in range(1):
    board.reset()
    board.draw_board()
    over = False
    score = 0
    # for _ in range(10):
    while True:
        start_time = time.time()
        action, score = expectimax.expectimax(board, depth)
        end_time = time.time()

        _, _, over = board.move(action)
        if over: break

        board.add_number()
        
        print("Move: {} \t Score: {} \t Speed: {s:.3f}K \t Time: {t:.3f}s".format(action, board.score, s = (sum(expectimax.stats) * 0.001) / (end_time - start_time + 1e-6),  t = (end_time - start_time)))
        
        # board.draw_board()
        
        expectimax.stats = [0] * depth
    print(board.score)
    score_list.append(board.score)

print(sum(score_list) / len(score_list))
board.draw_board()