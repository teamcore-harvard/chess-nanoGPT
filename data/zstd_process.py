# data format example
# [Event "Rated Bullet game"]
# [Site "https://lichess.org/iML6DQBA"]
# [Date "2023.01.01"]
# [Round "-"]
# [White "dida0617"]
# [Black "Wizard_of_North"]
# [Result "1-0"]
# [UTCDate "2023.01.01"]
# [UTCTime "00:00:42"]
# [WhiteElo "1877"]
# [BlackElo "1936"]
# [WhiteRatingDiff "+7"]
# [BlackRatingDiff "-6"]
# [ECO "A00"]
# [Opening "Van Geet Opening"]
# [TimeControl "30+0"]
# [Termination "Time forfeit"]
# 1. Nc3 { [%clk 0:00:30] } 1... Nf6 { [%clk 0:00:30] } 2. Nf3 { [%clk 0:00:30] } 2... e6 { [%clk 0:00:30] } 3. d4 { [%clk 0:00:30] } 3... d5 { [%clk 0:00:29] } 4. e4 { [%clk 0:00:30] } 4... dxe4 { [%clk 0:00:28] } 5. Ng5 { [%clk 0:00:30] } 5... h6 { [%clk 0:00:27] } 6. Ngxe4 { [%clk 0:00:29] } 6... Nxe4 { [%clk 0:00:27] } 7. Nxe4 { [%clk 0:00:29] } 7... Be7 { [%clk 0:00:26] } 8. Qf3 { [%clk 0:00:28] } 8... Bf6 { [%clk 0:00:25] } 9. c3 { [%clk 0:00:26] } 9... O-O { [%clk 0:00:24] } 10. Bd3 { [%clk 0:00:25] } 10... c5 { [%clk 0:00:22] } 11. Nxf6+ { [%clk 0:00:24] } 11... Qxf6 { [%clk 0:00:21] } 12. Qxf6 { [%clk 0:00:23] } 12... gxf6 { [%clk 0:00:20] } 13. dxc5 { [%clk 0:00:23] } 13... b6 { [%clk 0:00:18] } 14. cxb6 { [%clk 0:00:22] } 14... axb6 { [%clk 0:00:18] } 15. Be3 { [%clk 0:00:21] } 15... Bb7 { [%clk 0:00:18] } 16. O-O { [%clk 0:00:19] } 16... Nc6 { [%clk 0:00:18] } 17. Bxb6 { [%clk 0:00:18] } 17... Ne5 { [%clk 0:00:17] } 18. Bc2 { [%clk 0:00:16] } 18... Rad8 { [%clk 0:00:16] } 19. Bb3 { [%clk 0:00:14] } 19... Rb8 { [%clk 0:00:15] } 20. Bc7 { [%clk 0:00:13] } 20... Rbc8 { [%clk 0:00:13] } 21. Bxe5 { [%clk 0:00:12] } 21... fxe5 { [%clk 0:00:12] } 22. Rae1 { [%clk 0:00:12] } 22... f6 { [%clk 0:00:12] } 23. f4 { [%clk 0:00:10] } 23... Kf7 { [%clk 0:00:11] } 24. fxe5 { [%clk 0:00:10] } 24... f5 { [%clk 0:00:10] } 25. g4 { [%clk 0:00:08] } 25... Kg6 { [%clk 0:00:08] } 26. gxf5+ { [%clk 0:00:08] } 26... exf5 { [%clk 0:00:08] } 27. Be6 { [%clk 0:00:06] } 27... Rg8 { [%clk 0:00:07] } 28. Bxc8 { [%clk 0:00:04] } 28... Kh5+ { [%clk 0:00:05] } 29. Kf2 { [%clk 0:00:04] } 29... Rg2+ { [%clk 0:00:04] } 30. Ke3 { [%clk 0:00:03] } 30... Rxh2 { [%clk 0:00:03] } 31. e6 { [%clk 0:00:03] } 31... Rh3+ { [%clk 0:00:02] } 32. Kd4 { [%clk 0:00:03] } 32... Bc6 { [%clk 0:00:02] } 33. Kc5 { [%clk 0:00:03] } 33... Be8 { [%clk 0:00:01] } 34. e7 { [%clk 0:00:02] } 34... Rh4 { [%clk 0:00:00] } 35. Be6 { [%clk 0:00:02] } 35... Rc4+ { [%clk 0:00:00] } 36. Kxc4 { [%clk 0:00:01] } 1-0

import os
import re
import time
from typing import Optional, List
from torch.utils.data import IterableDataset
import tqdm
import zstandard
import io
import chess.pgn
import itertools
from chess.pgn import SkipType, BaseVisitor, SKIP
import itertools
import numpy as np
from tokenizers import (
    Encoding,
    Tokenizer,
    models,
    pre_tokenizers,
)

import tqdm
import re
import zstandard
import io
import time 


# def create_tokenizer() -> Tokenizer:
#     s = '''
#     1. e2e4P c7c6p 2. d2d4P d7d5p 3. b1d2N d5e4p 4. d2e4N g8f6n 5. e4f6N e7f6p 6. g1f3N f8d6b 7. f1d3B e8g8k 8. e1g1K c8g4b 9. c1e3B f8e8r 10. h2h3P g4h5b 11. d3e2B b8d7n 12. f3d2N h5e2b 13. d1e2Q d6f4b 14. e2f3Q f4e3b 15. f2e3P d8c7q 16. d2e4N e8e7r 17. e4g3N a8e8r 18. g3f5N e7e6r 19. a1e1R g7g6p 20. f5g3N e6e3r 21. e1e3R e8e3r 22. f3e3Q d7b6n 23. f1e1R b6d5n 24. e3e8Q g8g7k 25. e8e4Q c7g3q 26. e1e3R d5e3n 0-1
#     '''

#     tokenizer = Tokenizer(models.Model())
#     # tokenizer.normalizer = normalizers.Sequence(
#     #     [normalizers.NFD(), normalizers.Lowercase(),
#     #      normalizers.Replace(Regex('\d+\.'), '[MOVE]')

#     #      ]
#     # )

#     tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
#         [pre_tokenizers.WhitespaceSplit()]
#     )

#     board_positions = list(itertools.product('abcdefgh', '12345678'))
#     pieces = ['K', 'Q', 'R', 'B', 'N', 'P', 'k', 'q', 'r', 'b', 'n', 'p'] 
#     tokenizer.add_tokens(
#         [
#             *[''.join(x) for x in board_positions],
#             *pieces
#         ]
#     )

#     tokenizer.add_special_tokens(
#         ["[MOVE]", "[UNK]", "[MASK]"]
#     )

#     print('Tokenizer vocab size:', tokenizer.get_vocab_size())
#     print('Testing encoding...')
#     encoding: Encoding = tokenizer.encode(s)
#     print(encoding)
#     print('Passed!')
#     return tokenizer

def process_wrapper():
    vocab = '#+-.0123456789;=BKNOQRabcdefghx '
    del_chars = ''.join(c for c in map(chr, range(1114111)) if not c in vocab)
    del_map = str.maketrans('', '', del_chars)
    def process(game_str):
        res = {}

        for g in game_str.split('\n'):
            if g.startswith('['):
                k, v = g[1:-1].split(' "')
                res[k] = v[:-1]
            elif g.startswith('1. '):
                no_brackets_string = re.sub(r'\{.*?\}', '', g) # , flags=re.DOTALL
                no_brackets_string = no_brackets_string.translate(del_map)
                remove_dots = re.sub(r'\b\d+\.\.\. ', '', no_brackets_string)
                remove_game_result = re.sub(r'1-0|0-1|1/2-1/2', '', remove_dots)[:-2]
                remove_spaces = re.sub(r"(\d+)\.\s+", r"\1.", remove_game_result)
                remove_double_spaces = re.sub(r"  ", r" ", remove_spaces)
                res['transcript'] = remove_double_spaces
                
        return res
    return process

    
class StreamingPGNDataset(IterableDataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.process = process_wrapper()


    def read_game(self):
        dctx = zstandard.ZstdDecompressor()        
        with open(self.file_path, 'rb') as pgn_file:
            stream_reader = dctx.stream_reader(pgn_file)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
            
            fg = ""
            while True:
                for i in text_stream:
                    yield i                        
                    # fg += i
                    if i.startswith('1. '):
                        game = self.process(fg)
                        fg = ""
                        yield game
                        
                raise StopIteration

                
            # parse txt 
    def __iter__(self):
        return iter(self.read_game())


    
class StreamingBlockPGNDataset(StreamingPGNDataset):
    def __init__(self, file_path, transform=None, block_size=1024):
        self.file_path = file_path
        self.transform = transform
        self.process = process_wrapper()
        self.block_size = block_size
        self.tokenizer = {'vocab_size': 32, 'itos': {0: ' ', 1: '#', 2: '+', 3: '-', 4: '.', 5: '0', 6: '1', 7: '2', 8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 13: '8', 14: '9', 15: ';', 16: '=', 17: 'B', 18: 'K', 19: 'N', 20: 'O', 21: 'Q', 22: 'R', 23: 'a', 24: 'b', 25: 'c', 26: 'd', 27: 'e', 28: 'f', 29: 'g', 30: 'h', 31: 'x'}, 'stoi': {' ': 0, '#': 1, '+': 2, '-': 3, '.': 4, '0': 5, '1': 6, '2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, ';': 15, '=': 16, 'B': 17, 'K': 18, 'N': 19, 'O': 20, 'Q': 21, 'R': 22, 'a': 23, 'b': 24, 'c': 25, 'd': 26, 'e': 27, 'f': 28, 'g': 29, 'h': 30, 'x': 31}}


    def read_game_block(self):
        ds = iter(self.read_game())
        game = None
        full_block = ""
        while True:
            if game is not None: # use the previous game that was cut off in the last block
                full_block += f";{game['WhiteElo']} {game['BlackElo']} {game['transcript']}"
                
            while len(full_block) < self.block_size:
                game = next(ds)                
                full_block += f";{game['WhiteElo']} {game['BlackElo']} {game['transcript']}"
                
            # add np array
            out = full_block[:self.block_size]            
            full_block = ""            
            yield np.array([self.tokenizer['stoi'][c] for c in out], dtype=np.uint8)
            
                
    def __iter__(self):
        return iter(self.read_game_block())
        
if __name__ == '__main__':
    
    ############
    # StreamingPGNDataset
    ############
    ds = StreamingPGNDataset('/home/ezipe/git/transcendence/lichess-2023-janoct/lichess_db_standard_rated_2023-01.pgn.zst')    
    # ds = StreamingPGNDataset('/home/ezipe/git/transcendence/lichess-2023-janoct/test.pgn.zst')    
    kk = tqdm.tqdm()
    itr = iter(ds)
    while True:
    # next(itr)
        z = next(itr)
        kk.update()
    print(z)
    
    t = time.time()
    itrs = 10000
    for i in tqdm.tqdm(range(itrs)):
        z = next(itr)
    end = time.time() - t
    
    print(f'There are approximately 100M games per month...so this would take approximately {int(end * 1e8 / itrs / 60)} minutes to process the full month.')

    ############
    # StreamingBlockPGNDataset
    ############

    ds_block = StreamingBlockPGNDataset('/home/ezipe/git/transcendence/lichess-2023-janoct/lichess_db_standard_rated_2023-01.pgn.zst')
    itr_block = iter(ds_block)
    # next(itr_block)
    z = next(itr_block)
    print(z)
    
    t = time.time()
    itr_blocks = 1000000000000
    for i in tqdm.tqdm(range(itr_blocks)):
        z = next(itr_block)
    end = time.time() - t
    
    print(f'There are approximately 100M games per month...so this would take approximately {int(end * 1e8 / itr_blocks / 60)} minutes to process the full month.')


                
#####
#%%
# debug = False
# fg = ""
# pbar = tqdm.tqdm(desc="Processing games")
# dctx = zstandard.ZstdDecompressor()        
# pgn_file = open('/home/ezipe/git/transcendence/lichess-2023-janoct/lichess_db_standard_rated_2023-01.pgn.zst', 'rb')
# stream_reader = dctx.stream_reader(pgn_file)
# text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')        
# for i in text_stream:
#     fg += i
#     if i.startswith('1. '):
#         # process
#         if debug:
            
#             p = process(fg)
#             # if p['Result'] == '1/2-1/2':
#             print(p['transcript'])
#             # print(i)
#                 # print('\n\n')
#             time.sleep(.15)
#         else:
#             p = process(fg)
#         # end process        
#         fg = ""
#         pbar.update()

