import numpy as np
from Utils.utils import count_BER, cut_image_into_blocks, get_dct_coefs, quantize_dct_blocks, qm, concatenate_image, get_image_from_dct_coefs, dequantize_dct_blocks

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

def create_emb_map(positions, n=8):
    res = [[0 for _ in range(n)] for _ in range(n)]
    for p in positions:
        res[p[0]][p[1]] = 1
    return res


def embed_message_into_block(point, mes, positions, b):
    for p in positions:
        I = np.round(b[p[0]][p[1]])
        if point >= len(mes):
            break
        if I < 0:
            b[p[0]][p[1]] -= 1
        elif I == 0:
            b[p[0]][p[1]] -= mes[point]
            point += 1
        elif I == 1:
            b[p[0]][p[1]] += mes[point]
            point += 1
        else:
            b[p[0]][p[1]] += 1
            
    return point, b


def get_message_from_block(mes, positions, b, emb_l):
    for p in positions:
        if emb_l <= 0:
            break
        I = np.round(b[p[0]][p[1]])
        if I == 0 or I == 1:
            emb_l -= 1
            mes.append(0)
        elif I == -1 or I == 2:
            emb_l -= 1
            mes.append(1)
    
    return mes, b, emb_l


def get_capacity(positions, block_vector):
    cap = 0
    
    for b in block_vector:
        for p in positions:
            if np.round(b[p[0]][p[1]]) == 0 or np.round(b[p[0]][p[1]]) == 1:
                cap += 1
    
    return cap
    

def insert_message_AddEmb(image, message, k=8):
    l = len(message)

    #step1
    blocks = cut_image_into_blocks(image)
    dct_blocks = quantize_dct_blocks(get_dct_coefs(blocks))
    
    #step2
    b_vector = dct_blocks.reshape(4096, 8, 8)
    
    n = len(b_vector[0])
    num_blocks = len(dct_blocks)

    #step3
    Q = [[] for _ in range(n)]
    for i in range(num_blocks):
        u = i // n
        v = i %  n
        Q[u].append(dist_func(u, v, b_vector))
    
    #step4
    k_coords = []
    for i in range(n):
        for j in range(n):
            k_coords.append([i, j])
    
    k_mins = sorted(k_coords, key=lambda x: Q[x[0]][x[1]])

    embedding_positions = k_mins[:k]
    cap = get_capacity(embedding_positions, b_vector)

    if cap >= l:
        point = 0
        for idx in range(len(b_vector)):
            point, temp_b = embed_message_into_block(point, message, embedding_positions, b_vector[idx])
            b_vector[idx] = temp_b
            if point >= l:
                break

        new_image = concatenate_image(get_image_from_dct_coefs(dequantize_dct_blocks(b_vector.reshape(64, 64, 8, 8))))
        return new_image, {'route': embedding_positions, 'mes_len':l}
    else:
        raise RuntimeError("Capacity {} can't fit the message with size {}".format(cap, l))

def extract_message_AddEmb(im, route, mes_len):
    #step1
    blocks = cut_image_into_blocks(im)
    dct_blocks = quantize_dct_blocks(get_dct_coefs(blocks))
    
    #step2
    b_vector = dct_blocks.reshape(4096, 8, 8)
    
    extracted_message = []

    for idx in range(len(b_vector)):
        extracted_message, temp_b_2, mes_len = get_message_from_block(extracted_message, route, b_vector[idx], mes_len)
        if mes_len <= 0:
            break
    
    return extracted_message

def insert_message_iAddEmb(image, message, k=1, threshold=10):
    l = len(message)

    #step1
    blocks = cut_image_into_blocks(image)
    dct_blocks = quantize_dct_blocks(get_dct_coefs(blocks))
    
    #step2
    b_vector = dct_blocks.reshape(4096, 8, 8)
    
    n = len(b_vector[0])
    num_blocks = len(dct_blocks)

    #step3
    Q = [[] for _ in range(n)]
    for i in range(num_blocks):
        u = i // n
        v = i %  n
        Q[u].append(dist_func(u, v, b_vector))
    
    #step4
    k_coords = []
    for i in range(n):
        for j in range(n):
            k_coords.append([i, j])
    
    k_mins = sorted(k_coords, key=lambda x: Q[x[0]][x[1]])

    embedding_positions = k_mins[:k]
    cap = get_capacity(embedding_positions, b_vector)

    if cap >= l:
        point = 0
        for idx in range(len(b_vector)):
            t = 0
            cur_ber = 1
            best_ber = 1
            
            temp_point = point
            best_point = temp_point

            temp_b = b_vector[idx]
            best_b = temp_b

            while t < threshold and cur_ber != 0:
                temp_point, temp_b = embed_message_into_block(point, message, embedding_positions, b_vector[idx])
                old_mes = message[point:temp_point]
                new_mes, _, _ = get_message_from_block([], embedding_positions, b_vector[idx], temp_point-point)
                cur_ber = count_BER(old_mes, new_mes)

                if cur_ber < best_ber:
                    best_ber = cur_ber
                    best_b = temp_b
                    best_point = temp_point
            
            point = best_point
            b_vector[idx] = best_b
            if point >= l:
                break

        new_image = concatenate_image(get_image_from_dct_coefs(dequantize_dct_blocks(b_vector.reshape(64, 64, 8, 8))))
        return new_image, {'route': embedding_positions, 'mes_len':l}
    else:
        raise RuntimeError("Capacity {} can't fit the message with size {}".format(cap, l))
