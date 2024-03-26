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
import torch
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
import random


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



def calculate_split(total_splits, total_len, index):
    # Calculate the length of each split
    split_length = total_len // total_splits
    
    # Calculate the start and end indices of the split
    start_index = index * split_length
    end_index = start_index + split_length
    
    # Adjust the end index if the split is not evenly divided
    if index == total_splits - 1:
        end_index = total_len
    
    return start_index, end_index


    
class StreamingPGNDataset(IterableDataset):
    def __init__(self, file_paths, seed=42):
        self.set_file_paths(file_paths, seed)
        
        self.process = process_wrapper()
        
    def set_file_paths(self, file_paths, seed):
        self.file_paths = file_paths
        self.rng = random.Random(seed)
        self.rng.shuffle(self.file_paths)        

    def read_game(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None: #multiprocessing
            assert worker_info.num_workers <= len(self.file_paths), f'Num workers {worker_info.num_workers} greater than number of files {len(self.file_paths)}.'
            start, end = calculate_split(worker_info.num_workers, len(self.file_paths), worker_info.id)
            self.file_paths = self.file_paths[start:end]
            
        def game_generator(path):
            dctx = zstandard.ZstdDecompressor()        
            with open(path, 'rb') as pgn_file:
                stream_reader = dctx.stream_reader(pgn_file)
                text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
                
                fg = ""
                for i in text_stream:
                    # yield i                        
                    fg += i
                    if i.startswith('1. '):
                        game = self.process(fg)
                        fg = ""
                        yield game
        
        gen = [game_generator(file) for file in self.file_paths]
        i = 0
        while len(gen) > 0:
            try:
                game = next(gen[i % len(gen)])
                # if game.get('WhiteElo') is None or game.get('BlackElo') is None or game.get('transcript') is None:
                #     continue
            except StopIteration:
                del gen[i % len(gen)]
                continue
            
            yield game            
                
            # parse txt 
    def __iter__(self):
        return self.read_game()


    
class StreamingBlockPGNDataset(StreamingPGNDataset):
    def __init__(self, file_paths, seed=42, block_size=1024):
        self.set_file_paths(file_paths, seed)
        self.process = process_wrapper()
        self.block_size = block_size
        self.tokenizer = {'vocab_size': 32, 'itos': {0: ' ', 1: '#', 2: '+', 3: '-', 4: '.', 5: '0', 6: '1', 7: '2', 8: '3', 9: '4', 10: '5', 11: '6', 12: '7', 13: '8', 14: '9', 15: ';', 16: '=', 17: 'B', 18: 'K', 19: 'N', 20: 'O', 21: 'Q', 22: 'R', 23: 'a', 24: 'b', 25: 'c', 26: 'd', 27: 'e', 28: 'f', 29: 'g', 30: 'h', 31: 'x'}, 'stoi': {' ': 0, '#': 1, '+': 2, '-': 3, '.': 4, '0': 5, '1': 6, '2': 7, '3': 8, '4': 9, '5': 10, '6': 11, '7': 12, '8': 13, '9': 14, ';': 15, '=': 16, 'B': 17, 'K': 18, 'N': 19, 'O': 20, 'Q': 21, 'R': 22, 'a': 23, 'b': 24, 'c': 25, 'd': 26, 'e': 27, 'f': 28, 'g': 29, 'h': 30, 'x': 31}}


    def read_game_block(self):
        ds = self.read_game()
        game = None
        full_block = ""
        while True:
            if game is not None: # use the previous game that was cut off in the last block
                full_block += f";{game['WhiteElo']} {game['BlackElo']} {game['transcript']}"
                
            while len(full_block) < self.block_size:
                game = next(ds)   
                # try:                             
                full_block += f";{game['WhiteElo']} {game['BlackElo']} {game['transcript']}"
                # except KeyError:
                #     print(game)
                #     raise KeyError
                
            # add np array
            out = full_block[:self.block_size]            
            full_block = ""            
            yield np.array([self.tokenizer['stoi'][c] for c in out], dtype=np.int64)
            
                
    def __iter__(self):
        return self.read_game_block()

# TODO add cycle across multiple files and huggingface dataloader (latter for mp)
# takes around 30 minutes to cycle through a single dataset

if __name__ == '__main__':
    
    ############
    # StreamingPGNDataset
    ############
    ds = StreamingPGNDataset(['/mnt/data/lichess_2023_janoct_shards/data/lichess_db_standard_rated_2023-01.pgn.00.zst'])
    # ds = StreamingPGNDataset('/home/ezipe/git/transcendence/lichess-2023-janoct/test.pgn.zst')    
    for k in tqdm.tqdm(ds):
        pass
    
    # t = time.time()
    # itrs = 10000
    # for i in tqdm.tqdm(range(itrs)):
    #     z = next(itr)
    # end = time.time() - t
    
    # print(f'There are approximately 100M games per month...so this would take approximately {int(end * 1e8 / itrs / 60)} minutes to process the full month.')

    ############
    # StreamingBlockPGNDataset
    ############

    data_dir = '/mnt/data/lichess_2023_janoct_shards/data/'
    ds_block = torch.utils.data.DataLoader(StreamingBlockPGNDataset([os.path.join(data_dir, k) for k in os.listdir(data_dir)]), num_workers=47, batch_size=1024)
    itr_block = iter(ds_block)
    # next(itr_block)
    z = next(itr_block)
    print(z)
    print(z.shape)
    
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

