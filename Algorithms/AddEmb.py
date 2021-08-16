import numpy as np
from Utils.utils import cut_image_into_blocks, get_dct_coefs, quantize_dct_blocks, qm

def cost(u, v):
    """
    This function returns the distortion cost using formula stated in the corresponding article
    """
    c = lambda x: 1/np.sqrt(2) if x == 0 else 1
    res = 0
    u += 1
    v += 1
    for x in range(1, 9):
        for y in range(1, 9):
            res += np.square((0.25*c(u)*c(v)*qm[u-1][v-1]*np.cos((2*x+1)*u*3.14/16)*np.cos((2*y+1)*v*3.14/16)))
    
    return res

def dist_func(u, v, blocks_vector):
    """
    Returns the value of distortion for given positon on DCT coefficients according to the formula from article

    WARNING:
    Probably will have to change comparison by removing np.round operation
    """
    p = 1
    q = 1
    for b in blocks_vector:
        if np.round(b[u][v]) == 0 or np.round(b[u][v]) == 1:
            q += 1
        else:
            p += 1
    
    res = 0.5
    res += p/q
    return res * cost(u, v)

def insert_message_AddEmb(image, message):
    L = len(message)
    
    #step1
    blocks = cut_image_into_blocks(image)
    dct_blocks = quantize_dct_blocks(get_dct_coefs(blocks))
    
    #step2
    b_vector = dct_blocks.flatten(1)
    print(b_vector.shape)

    #step3
    Q = [[] for _ in len(b_vector[0])]
    for i in range(len(b_vector)):
        u = i // 8
        v = i %  8
        Q[u].append(dist_func(u, v, b_vector))
    
    
    return Q

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

def extract_message_AddEmb(im, route):
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


print("Cost values for 0,0 is {}, for 7,7 is {}".format(cost(0,0), cost(7,7)))
