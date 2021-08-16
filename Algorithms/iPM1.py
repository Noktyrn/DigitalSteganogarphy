from math import floor
import sys
sys.path.append("..")
from Utils.utils import cut_image_into_blocks, concatenate_image, dct2d, idct2d, count_BER
from numpy.random import randint
from Algorithms.PM1 import extract_message_PM1, insert_message_PM1

def get_block_indexes_structure_from_route(route):
    i = 0
    res_coefs = dict()
    res_idxs = dict()
    b_ord = []
    for i in route[0]:
        if i not in b_ord:
            b_ord.append(i)
    
    for b in set(route[0]):
        res_coefs[b] = []
        res_idxs[b] = []
    
    for i in range(len(route[0])):
        res_coefs[route[0][i]].append(route[1][i])
        res_idxs[route[0][i]].append(i)
    
    return {'block_order': b_ord, 'coefs': res_coefs, 'indexes': res_idxs}


def get_mes_from_block(block, coefs):
    mes = []
    n = len(block)
    
    dct_block = dct2d(block)
    
    for i in coefs:
        row = i // n
        col = i % n
        
        c = floor(dct_block[row][col])
        
        if (c < 0 and c % 2 == 1) or (c > 0 and c % 2 == 0):
            mes.append(0)
        else:
            mes.append(1)
    
    return mes
        
        
def embed_mes_into_block(block, mes, coefs):
    dct_block = dct2d(block)
    n = len(block)
    
    for i in range(len(coefs)):
        c_ind = coefs[i]
        
        row = c_ind // n
        col = c_ind % n
        
        c_orig = dct_block[row][col]
        c = floor(c_orig)
        
        if mes[i] == 0:
            if (c < 0 and c % 2 == 1) or (c > 0 and c % 2 == 0):
                continue
        else:
            if (c < 0 and c % 2 == 0) or (c > 0 and c % 2 == 1):
                continue
                
        r = 0
        
        if abs(c) > 1:
            r = randint(0, 2)
        elif c == 1:
            r = 0
        else:
            r = 1
        
        if r == 1:
            dct_block[row][col] = c_orig - 1
        else:
            dct_block[row][col] = c_orig + 1
    
    return idct2d(dct_block)
        

def insert_message_iPM1(image, message):
    THRESHOLD=10

    emb, d = insert_message_PM1(image, message)
    U = d['route']
    res = extract_message_PM1(emb, U)

    ber = count_BER(message, res)
    
    if ber != 0:
        block_dict = get_block_indexes_structure_from_route(U)
        blocks = cut_image_into_blocks(emb)
        n = len(blocks)
        
        for b_ind in block_dict['block_order']:
                
            indexes = block_dict['indexes'][b_ind]
            coefs = block_dict['coefs'][b_ind]
            
            part_of_message_to_embed = [message[i] for i in indexes]
            
            row_b = b_ind // n
            col_b = b_ind % n
            
            b = blocks[row_b][col_b]
            b_temp = b
            ber = 1
            
            t = 0
            n_mes = []
            
            while ber != 0 and t <= THRESHOLD:
                b_temp = embed_mes_into_block(b_temp, part_of_message_to_embed, coefs)
                n_mes = get_mes_from_block(b_temp, coefs)
                
                ber_temp = count_BER(n_mes, part_of_message_to_embed)
                if ber_temp < ber:
                    b = b_temp
                
                ber = ber_temp
                t += 1

            blocks[row_b][col_b] = b
            
            n_mes = get_mes_from_block(blocks[row_b][col_b], coefs)
        
    emb = concatenate_image(blocks)
    
    return emb, {'route': U}
