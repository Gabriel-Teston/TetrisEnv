import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'console'],
                'video.frames_per_second':350}

    def __init__(self, size=(20,10)):
        self.board = self.Board()
        self.init_pos = np.array([0,5])
        self.current_piece = self.Piece(self.init_pos)
        self.next_piece = self.Piece(self.init_pos)
        self.time_step = 0
        self.done = False
        self.action_shape = 2
        self.state_shape = 11
        self._max_episode_steps = 100000
        self.screen = None


    def step(self, action):
        self.time_step += 1
        rot, mov_x = action.astype(int)
        reward = 0
        for r in range(abs(rot)):
            self.current_piece.rotate(rot/abs(rot))
            if self.board.hit(self.current_piece):
                self.current_piece.rotate(-rot/abs(rot))
        for x in range(abs(mov_x)):
            self.current_piece.move(np.array([0, mov_x/abs(mov_x)]))
            if self.board.hit(self.current_piece):
                self.current_piece.move(np.array([0, -mov_x/abs(mov_x)]))
        self.current_piece.move(np.array([1, 0]))
        if self.board.hit(self.current_piece):
            self.current_piece.move(np.array([-1, 0]))
            self.board.blit(self.current_piece)
            self.current_piece = self.next_piece
            self.next_piece = self.Piece(self.init_pos)
            reward += 1
        reward += self.board.clean()
        height = self.board.check()
        self.done = self.board.done
        return self.get_state(), reward, self.done, None
        
    def reset(self):
        self.board = self.Board()
        self.current_piece = self.Piece(self.init_pos)
        self.next_piece = self.Piece(self.init_pos)
        self.time_step = 0
        self.done = False
        return self.get_state()
    
    def render(self, mode='human', close=False):
        """
        This function renders the current game state in the given mode.
        """
        if mode == 'console':
            print(self.board.blit(self.current_piece, ret=True))
        elif mode == "human":
            try:
                import pygame
                from pygame import gfxdraw
            except ImportError as e:
                raise error.DependencyNotInstalled(
                    "{}. (HINT: install pygame using `pip install pygame`".format(e))
            if close:
                pygame.quit()
            else:
                if self.screen is None:
                    pygame.init()
                    self.screen = pygame.display.set_mode(
                        (round(self.board.size[1]*10), round(self.board.size[0]*10)))
                clock = pygame.time.Clock()
                
                board = self.board.blit(self.current_piece, ret=True)
                # Draw old bubbles
                self.screen.fill((255, 255, 255))
                for row in range(board.shape[0]):
                    for column in range(board.shape[1]):
                        if board[row, column]:
                            pygame.draw.rect(self.screen, (0,0,0), pygame.Rect(column*10, row*10, 9, 9))
                            #pygame.box(self.screen, pygame.Rect(column*10, row*10, 9, 9), (0,0,0))

                pygame.display.update()
        else:
            raise error.UnsupportedMode("Unsupported render mode: " + mode)
        
        
        
        
        
        
        
        
        """if mode == 'human':
            board = self.board.blit(self.current_piece, ret=True)
            if not self.visual:
                self.visual = self.visual_render()
                pyglet.app.run()
            
        if mode == "rgb_array":
            board = self.board.blit(self.current_piece, ret=True)
            rgb_board = board.reshape([board.shape[0], board.shape[1],1])
            plt.imshow(board)
            
            return cv2.resize(np.concatenate([rgb_board for z in range(3)], axis=2).astype("uint8"), (2000,1000))"""
    
    
        
    
    def get_state(self):
        simbol = self.current_piece.piece_simbol
        next_simbol = self.next_piece.piece_simbol
        rot = self.current_piece.rot
        #y, x 
        pos = self.current_piece.pos
        height = self.board.height
        state = np.concatenate([[simbol], [next_simbol], [rot], pos, height]).astype(float)
        return state
        
    def sample(self):
        return 2*np.random.sample([2])-1
    
    class action_space:
        shape = [2]
        high = [10]
        low = [-10]
        def sample():
            a = 20*np.random.sample([2])-10
            return a
        
    class observation_space:
        shape = [15]
        _max_episode_steps = 100000
    
    class Piece:
        pieces = {
            #"l"
            0:np.array([[1, 0],
                          [1, 0],
                          [1, 1]]),
            #"j"
            1:np.array([[0, 1],
                          [0, 1],
                          [1, 1]]),
            #"i"
            2:np.array([[1],
                          [1],
                          [1],
                          [1]]),
            #"t"
            3:np.array([[0, 1],
                          [1, 1],
                          [0, 1]]),
            #"o"
            4:np.array([[1, 1],
                          [1, 1]]),
            #"s"
            5:np.array([[1, 0],
                          [1, 1],
                          [0, 1]]),
            #"z"
            6:np.array([[0, 1],
                          [1, 1],
                          [1, 0]]),
        }
        
        def __init__(self, pos):
            self.piece_simbol = np.random.choice([key for key in TetrisEnv.Piece.pieces.keys()])
            self.data = self.pieces[self.piece_simbol]
            self.pos = np.array(pos)
            self.rot = 0
            
        def rotate(self, direction):
            self.data = self.data.T[::-1,:] if direction  == -1 else (self.data.T[:,::-1] if direction else self.data)
            self.rot += (direction % 4)
            
        def move(self, direction):
            self.pos += direction.astype(int)
            
    class Board:
        def __init__(self, size=(20,10)):
            self.size = size
            self.data = np.zeros(self.size)
            self.score = 0
            self.done = False
            self.height = np.zeros([self.size[1]])

        def hit(self, piece):
            piece_end = piece.pos+piece.data.shape
            if (piece.pos < 0).any() or ((piece_end) > self.data.shape).any():
                return True
            if (piece.data+self.data[piece.pos[0]:piece_end[0], 
                                     piece.pos[1]:piece_end[1]] > 1).any():
                return True
            return False
            
        def blit(self, piece, ret=False):
            if ret:
                board_copy = np.copy(self.data)
                piece_end = piece.pos+piece.data.shape
                for y in range(self.data.shape[0]):
                    for x in range(self.data.shape[1]):
                        if (np.array([y,x]) >= piece.pos).all() and (np.array([y,x]) < piece_end).all():
                            board_copy[y,x] += piece.data[y-piece.pos[0], x-piece.pos[1]]
                return board_copy
            else:
                self.score += 1
                piece_end = piece.pos+piece.data.shape
                for y in range(self.data.shape[0]):
                    for x in range(self.data.shape[1]):
                        if (np.array([y,x]) >= piece.pos).all() and (np.array([y,x]) < piece_end).all():
                            self.data[y,x] += piece.data[y-piece.pos[0], x-piece.pos[1]]
            
        def clean(self):
            count = 0
            buffer = []
            for y in range(self.size[0]-1, -1, -1):
                if (self.data[y,:] > 0).all():
                    count += 1
                else:
                    buffer.append([self.data[y,:]])

            for i in range(count):
                 buffer.append(np.zeros([1, self.size[1]]))
            self.score += count*10
            self.data = np.concatenate(buffer[::-1], axis=0)#.reshape(self.size)
            return count*10
        
        def check(self):
            height = []
            for x in range(self.size[1]):
                y = 0
                while self.data[y,x] == 0 and y < self.size[0] - 1:
                    y += 1
                height.append(self.size[0] - y)
            height = np.array(height)
            if (height >= self.size[0]).any():
                self.done = True
            self.height = height
            return height
            
        def add(self, atom, pos):
            self.data.insert(pos, atom)
            self.size += 1
            if atom.plus and self.size > 2:
                self.ajust(pos)
            
        def ajust(self, pos):
            pass
                    
        def neigbors(pos):
            
            return left,right