from math import floor
import sys
sys.path.append("..")
from Utils.utils import cut_image_into_blocks, get_dct_coefs, get_image_from_dct_coefs, concatenate_image
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