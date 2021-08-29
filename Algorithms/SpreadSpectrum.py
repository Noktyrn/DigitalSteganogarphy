import numpy as np
from Utils.utils import cut_image_into_blocks, concatenate_image
from Utils.utils import count_BER


def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def add_bits_into_seq(bits, seq, block, G=2):
    MIN_LEN = 4
    
    m = [-1 if bit == 0 else 1 for bit in bits]
    
    n = len(block)
    emb_len = len(seq) // len(bits)
    
    if emb_len >= MIN_LEN:
        new_seq = [chunk for chunk in get_chunks(seq, emb_len)]
        
        for r_idx in range(n):
            for c_idx in range(n):
                f_idx = r_idx * n + c_idx
                chunk_idx = f_idx // emb_len
                coef_idx = f_idx % emb_len
                
                block[r_idx][c_idx] += new_seq[chunk_idx][coef_idx] * m[chunk_idx] * G
                block[r_idx][c_idx] = np.clip(block[r_idx][c_idx], 0, 255)  

        return block
    else:
        raise RuntimeError("The embedding sequence with len = {} is too short, need minimum {}".format(emb_len, MIN_LEN))


def get_bits_from_seq(seq, block, emb_len):
    temp_block = [c for c in get_chunks(block.reshape(64), emb_len)]
    temp_seq = [c for c in get_chunks(seq, emb_len)]
    
    res = []
    for i in range(len(temp_block)):
        bit = np.correlate(temp_block[i], temp_seq[i])
        if bit > 0:
            res.append(1)
        else:
            res.append(0)
    return res


def insert_message_SpreadSpectrum(image, message):
    MAX_LEN=16
    
    #step1
    blocks = cut_image_into_blocks(image)

    seqs = [[[-1 if i == 0 else 1 for i in np.random.randint(0,2,64)] for _ in range(len(blocks))] for _ in range(len(blocks))]
    
    n = len(blocks)
    m = len(blocks[0])
    emb_len = len(message) // (m*n)

    if emb_len <= MAX_LEN:
        message_chunks = [c for c in get_chunks(message, emb_len)]
        for i in range(len(message_chunks)):
            m = message_chunks[i]
            c_idx = i % n
            r_idx = i // n
            block = blocks[r_idx][c_idx]
            seq = seqs[r_idx][c_idx]
            
            blocks[r_idx][c_idx] = add_bits_into_seq(m, seq, block)
        return concatenate_image(blocks), {'seqs': seqs, 'emb_len':emb_len}
    else:
        raise RuntimeError("The embedding sequence with len = {} is too long, need max {}".format(emb_len, MAX_LEN))


def extract_message_SpreadSpectrum(im, seqs, emb_len):
    res = []

    blocks = cut_image_into_blocks(im)
    
    for r_idx in range(len(blocks)):
        for c_idx in range(len(blocks[0])):
            block = blocks[r_idx][c_idx]
            seq = seqs[r_idx][c_idx]
            
            res.extend(get_bits_from_seq(seq, block, emb_len))
    
    return res


def insert_message_iSpreadSpectrum(image, message, threshold=10):
    MAX_LEN=16
    
    #step1
    blocks = cut_image_into_blocks(image)

    seqs = [[[-1 if i == 0 else 1 for i in np.random.randint(0,2,64)] for _ in range(len(blocks))] for _ in range(len(blocks))]
    
    n = len(blocks)
    m = len(blocks[0])
    emb_len = len(message) // (m*n)

    if emb_len <= MAX_LEN:
        message_chunks = [c for c in get_chunks(message, emb_len)]
        for i in range(len(message_chunks)):
            m = message_chunks[i]
            c_idx = i % n
            r_idx = i // n
            block = blocks[r_idx][c_idx]
            seq = seqs[r_idx][c_idx]
            
            temp_block = block
            best_block = temp_block
            t = 0
            best_ber = 1
            
            while t < threshold and best_ber != 0:
                temp_block = add_bits_into_seq(m, seq, temp_block)
                ext_mes = get_bits_from_seq(seq, temp_block, emb_len)
                cur_ber = count_BER(ext_mes, m)
                if cur_ber < best_ber:
                    best_ber = cur_ber
                    best_block = temp_block
                
                t += 1
            blocks[r_idx][c_idx] = best_block
        return concatenate_image(blocks), {'seqs': seqs, 'emb_len':emb_len}
    else:
        raise RuntimeError("The embedding sequence with len = {} is too long, need max {}".format(emb_len, MAX_LEN))
