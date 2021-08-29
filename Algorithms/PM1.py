from math import floor
import sys
sys.path.append("..")
from Utils.utils import cut_image_into_blocks, get_dct_coefs, get_image_from_dct_coefs, concatenate_image, dct2d, idct2d, count_BER
from numpy.random import randint

def count_v_i(block, n=8):
    v = 0
    
    for i in range(2, n*n+1):
        row = (i-1)// n
        col = (i-1) % n
        if block[row][col] != 0:
            v += 1
    
    return v

def g_j(j):
    if j <= 7:
        return 0
    elif j<= 31:
        return 1
    elif j <= 41:
        return 2
    else:
        return 3

def count_w_i(block, n=8):
    w = 0
    
    for i in range(2, n*n+1):
        row = (i-1)// n
        col = (i-1) % n
        if block[row][col] != 0:
            w += g_j(i)
    
    return w

def insert_message_PM1(image, message):
    L = len(message)
    
    #step1
    blocks = cut_image_into_blocks(image)
    dct_blocks = get_dct_coefs(blocks)
    
    #step2
    V = []
    W = []
    for row in dct_blocks:
        for block in row:
            V.append(count_v_i(block))
            W.append(count_w_i(block))

    #step3
    K = len(V)
    S = [i for i in range(K)]
    

    #step4
    flag = True
    while flag:
        flag = False
        for p in range(0, K-1):
            for q in range(p+1, K):
                if not(V[S[p]] > V[S[q]] or (V[S[p]] == V[S[q]] and W[S[p]] >= W[S[q]])):
                    S[p], S[q] = S[q], S[p]
                    flag = True

    #step5
    U = [[0 for i in range(L)] for _ in range(2)]
    n = 0
    f = 63
    y = 0
    
    #step6
    while y < L and f >= 0:
        row_c = f // 8
        col_c = f % 8
        
        row_b = S[n] // 64
        col_b = S[n] % 64
        
        c = dct_blocks[row_b][col_b][row_c][col_c]
        
        if floor(c) != 0:
            U[0][y] = S[n]
            U[1][y] = f
            y += 1
        
        n += 1
        if n >= K:
            n = 1
            f -= 1
    
    if f < 0:
        raise RuntimeError("Can embed only {} bits, ran out of non-zero DCT coefficients".format(y))

    #step7
    y = 0

    #step8
    while y < L:
        row_b = U[0][y] // 64
        col_b = U[0][y] % 64

        row_c = U[1][y] // 8
        col_c = U[1][y] % 8

        c_orig = dct_blocks[row_b][col_b][row_c][col_c]
        c = floor(c_orig)
        #step8.1
        if message[y] == 0:
            #step8.3
            if (c < 0 and c % 2 == 1) or (c > 0 and c % 2 == 0):
                y += 1
                continue
        else:
            #step8.2
            if (c < 0 and c % 2 == 0) or (c > 0 and c % 2 == 1):
                y += 1
                continue

        #step8.4
        r = 0

        if abs(c) > 1:
            r = randint(0, 2)
        #step8.5
        elif c == 1:
            r = 0
        else:
            r = 1

        #step8.6
        if r == 1:
            dct_blocks[row_b][col_b][row_c][col_c] = c_orig - 1
        else:
            dct_blocks[row_b][col_b][row_c][col_c] = c_orig + 1

        y += 1

    new_image = get_image_from_dct_coefs(dct_blocks)
    return concatenate_image(new_image), {'route': U}

def extract_message_PM1(im, route):
    dct_blocks = get_dct_coefs(cut_image_into_blocks(im))
    mes = []
    
    for i in range(len(route[0])):
        b = route[0][i]
        
        row_b = b // 64
        col_b = b % 64
        
        c = route[1][i]
        
        row_c = c // 8
        col_c = c % 8
        
        c = floor(dct_blocks[row_b][col_b][row_c][col_c])
        
        if (c < 0 and c % 2 == 1) or (c > 0 and c % 2 == 0):
            mes.append(0)
        else:
            mes.append(1)
    
    return mes

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
        

def insert_message_iPM1(image, message, threshold=10):
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
            
            while ber != 0 and t <= threshold:
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