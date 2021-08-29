import numpy as np
import math
from Utils.utils import count_BER


def cut_image_into_pairs(image):
    temp_image = image.reshape(len(image) * len(image[0]))
    res = []
    for i in range(0, len(temp_image) - 1, 2):
        res.append([temp_image[i],temp_image[i+1]])
    return res


def concat_image_from_pairs(pairs):
    res = []
    for x, y in pairs:
        res.append(x)
        res.append(y)
    
    n = int(np.sqrt(len(res)))
    return np.asarray(res).reshape(n, n)


def get_n(dif):
    range_table = [[0, 7], [8, 15], [16, 31], [32, 63],[64, 127], [128, 255]]
    for r in range_table:
        if r[0] <= dif <= r[1]:
            return math.floor(np.log2(r[1]-r[0]+1))


def to_bin(dec):
    res = []
    while dec > 0:
        mod = dec % 2
        dec //= 2
        res.insert(0, int(mod))
    
    return res


def to_dec(b):
    dec = 0
    n = len(b) - 1
    for i in range(len(b)):
        dec += (b[i] * 2**(n-i))
    
    return dec


def embed_into_pair(message, p1, p2, point):
    if point >= len(message):
        return p1, p2, point
    
    d = p2-p1
    n = get_n(abs(int(d)))
    bits = []
    if point + n >= len(message):
        bits = message[point:]
    else:
        bits = message[point:point+n]

    point += n
    b = to_dec(bits)
    l = 2**n
    d_t = l + b

    if d < 0:
        d_t *= -1

    if d % 2 == 1:
        p1 -= math.ceil((d_t - d)/2)
        p2 += math.floor((d_t - d)/2)
    else:
        p1 -= math.floor((d_t - d)/2)
        p2 += math.ceil((d_t - d)/2)
    
    return p1, p2, point


def get_mes_from_pair(p1, p2):
    d = abs(p2-p1)
    n = get_n(d)
    l = 2**n
    part = to_bin(d - l)
    
    if len(part) < n:
        for _ in range(n-len(part)):
            part.insert(0, 0)
    
    return part


def insert_message_PVD(image, message):
    pairs = cut_image_into_pairs(image)
    point = 0
    l = 0
    d = {}
    for i in range(len(pairs)):
        point_old = point
        pairs[i][0], pairs[i][1], point = embed_into_pair(message, pairs[i][0], pairs[i][1], point)
        d[l] = message[point_old:point]
        l+=1
        if point >= len(message):
            break
    
    return concat_image_from_pairs(pairs), {'pair_num': l, 'd':d}


def insert_message_iPVD(image, message, threshold=10):
    pairs = cut_image_into_pairs(image)
    point = 0
    l = 0
    d = {}
    for i in range(len(pairs)):
        best_ber = 1
        best_pair = [pairs[i][0], pairs[i][1]]
        point_old = point
        t = 0
        
        while best_ber != 0 and t < threshold:
            pairs[i][0], pairs[i][1], point = embed_into_pair(message, pairs[i][0], pairs[i][1], point_old)
            d[l] = message[point_old:point]
            ext_mes = get_mes_from_pair(pairs[i][0], pairs[i][1])
            
            ber = count_BER(d[l], ext_mes)
            if ber < best_ber:
                best_ber = ber
                best_pair = [pairs[i][0], pairs[i][1]]
            l+=1
            t += 1
        
        pairs[i][0], pairs[i][1] = best_pair[0], best_pair[1]
        
        if point >= len(message):
            break
    
    return concat_image_from_pairs(pairs), {'pair_num': l}


def extract_message_PVD(image, pair_num, d):
    pairs = cut_image_into_pairs(image)
    res = []
    i = 0
    for p in pairs[:pair_num]:
        mes_part = get_mes_from_pair(p[0], p[1])

        if len(mes_part) != len(d[i]):
            print("Got {}, needed to get {}, iter {}".format(mes_part, d[i], i))
        else:
            for j in range(len(d[i])):
                if d[i][j] != mes_part[j]:
                    print("Got {}, needed to get {}, iter {}".format(mes_part, d[i], i))

        res.extend(mes_part)
        i+=1
    
    return res