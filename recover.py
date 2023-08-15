import numpy as np
import cv2
from queue import Queue
from main import read_img, img2arr, arr2img, bin2int

def de_enhance(arr):
    width, height = arr.shape[0], arr.shape[1]
    bookkeeping = np.array([], dtype=int)
    # 读取32个LSB
    pqrs = np.array([], dtype=int)
    for h in range(height-32, height):
        pqrs = np.concatenate((pqrs, [arr[width-1,h]%2]), axis=0)

    # 最后一次
    x_max_l = bin2int(pqrs[:8])
    y_max_l = bin2int(pqrs[8:16])
    x_max_r = bin2int(pqrs[16:24])
    y_max_r = bin2int(pqrs[24:32])
    extracted_bits = np.array([], dtype=int)
    for w in range(width):
        for h in range(0, height, 2):
            if w == width-1 and h >= height-32:
                break

            if arr[w,h] == x_max_l or arr[w,h] == x_max_l-1:
                extracted_bits = np.concatenate((extracted_bits, [x_max_l-arr[w,h]]), axis=0)
                arr[w,h] = x_max_l
            elif arr[w,h] < x_max_l-1:
                arr[w,h] += 1
            elif arr[w,h] == x_max_r or arr[w,h] == x_max_r+1:
                extracted_bits = np.concatenate((extracted_bits, [arr[w,h]-x_max_r]), axis=0)
                arr[w,h] = x_max_r
            elif arr[w,h] > x_max_r+1:
                arr[w,h] -= 1

    for w in range(width):
        for h in range(1, height, 2):
            if w == width-1 and h >= height-32:
                break

            if arr[w,h] == y_max_l or arr[w,h] == y_max_l-1:
                extracted_bits = np.concatenate((extracted_bits, [y_max_l-arr[w,h]]), axis=0)
                arr[w,h] = y_max_l
            elif arr[w,h] < y_max_l-1:
                arr[w,h] += 1
            elif arr[w,h] == y_max_r:
                extracted_bits = np.concatenate((extracted_bits, [arr[w,h]-y_max_r]), axis=0)
                arr[w,h] = y_max_r
            elif arr[w,h] > y_max_r+1:
                arr[w,h] -= 1
    # 恢复32个LSB
    pqrs, S, lsb, extracted_bits = extracted_bits[:32], extracted_bits[32:40], extracted_bits[40:72], extracted_bits[72:]
    x_max_l = bin2int(pqrs[:8])
    y_max_l = bin2int(pqrs[8:16])
    x_max_r = bin2int(pqrs[16:24])
    y_max_r = bin2int(pqrs[24:32])
    S = bin2int(S)
    bookkeeping = np.concatenate((extracted_bits, bookkeeping), axis=0)
    for i in range(32):
        arr[width-1,height-32+i] = arr[width-1,height-32+i] - arr[width-1, height-32+i]%2 + lsb[i]

    # 前S-1次
    for iteration in range(S-2):
        extracted_bits = np.array([], dtype=int)
        for w in range(width):
            for h in range(0, height, 2):
                if arr[w,h] == x_max_l or arr[w,h] == x_max_l-1:
                    extracted_bits = np.concatenate((extracted_bits, [x_max_l-arr[w,h]]), axis=0)
                    arr[w,h] = x_max_l
                elif arr[w,h] < x_max_l-1:
                    arr[w,h] += 1
                elif arr[w,h] == x_max_r or arr[w,h] == x_max_r+1:
                    extracted_bits = np.concatenate((extracted_bits, [arr[w,h]-x_max_r]), axis=0)
                    arr[w,h] = x_max_r
                elif arr[w,h] > x_max_r+1:
                    arr[w,h] -= 1

        for w in range(width):
            for h in range(1, height, 2):
                if arr[w,h] == y_max_l or arr[w,h] == y_max_l-1:
                    extracted_bits = np.concatenate((extracted_bits, [y_max_l-arr[w,h]]), axis=0)
                    arr[w,h] = y_max_l
                elif arr[w,h] < y_max_l-1:
                    arr[w,h] += 1
                elif arr[w,h] == y_max_r:
                    extracted_bits = np.concatenate((extracted_bits, [arr[w,h]-y_max_r]), axis=0)
                    arr[w,h] = y_max_r
                elif arr[w,h] > y_max_r+1:
                    arr[w,h] -= 1

        pqrs, extracted_bits = extracted_bits[:32], extracted_bits[32:]
        x_max_l = bin2int(pqrs[:8])
        y_max_l = bin2int(pqrs[8:16])
        x_max_r = bin2int(pqrs[16:24])
        y_max_r = bin2int(pqrs[24:32])
        bookkeeping = np.concatenate((extracted_bits, bookkeeping), axis=0)

    # 第一次
    extracted_bits = np.array([], dtype=int)
    for w in range(width):
        for h in range(0, height, 2):
            if arr[w,h] == x_max_l or arr[w,h] == x_max_l-1:
                extracted_bits = np.concatenate((extracted_bits, [x_max_l-arr[w,h]]), axis=0)
                arr[w,h] = x_max_l
            elif arr[w,h] < x_max_l-1:
                arr[w,h] += 1
            elif arr[w,h] == x_max_r or arr[w,h] == x_max_r+1:
                extracted_bits = np.concatenate((extracted_bits, [arr[w,h]-x_max_r]), axis=0)
                arr[w,h] = x_max_r
            elif arr[w,h] > x_max_r+1:
                arr[w,h] -= 1

    for w in range(width):
        for h in range(1, height, 2):
            if arr[w,h] == y_max_l or arr[w,h] == y_max_l-1:
                extracted_bits = np.concatenate((extracted_bits, [y_max_l-arr[w,h]]), axis=0)
                arr[w,h] = y_max_l
            elif arr[w,h] < y_max_l-1:
                arr[w,h] += 1
            elif arr[w,h] == y_max_r:
                extracted_bits = np.concatenate((extracted_bits, [arr[w,h]-y_max_r]), axis=0)
                arr[w,h] = y_max_r
            elif arr[w,h] > y_max_r+1:
                arr[w,h] -= 1

    bookkeeping = np.concatenate((extracted_bits, bookkeeping), axis=0)

    # 分离bookkeeping和embedding_bits
    length, bookkeeping = bookkeeping[:20], bookkeeping[20:]
    length = bin2int(length)
    bookkeeping, embedding_bits = bookkeeping[:length], bookkeeping[length:]

    return arr, S, bookkeeping, embedding_bits
    

def de_preprocessing(arr, bookkeeping, S):
    width, height = arr.shape[0], arr.shape[1]
    for iteration in range(S):
        y_min_r, bookkeeping = bin2int(bookkeeping[-8:]), bookkeeping[:-8]
        sign, bookkeeping = bookkeeping[-1], bookkeeping[:-1]
        # 没有合并过
        if sign == 1:
            for w in range(width):
                for h in range(1, height, 2):
                    if arr[w,h] >= y_min_r:
                        arr[w,h] += 1
        # 合并过
        else:
            for w in range(width):
                for h in range(1, height, 2):
                    if arr[w,h] >= y_min_r:
                        arr[w,h] += 1
            for w in range(width-1, -1, -1):
                for h in range(height-1, -1, -2):
                    if arr[w,h] == y_min_r-1:
                        item, bookkeeping = bookkeeping[-1], bookkeeping[:-1]
                        if item == 1:
                            arr[w,h] = y_min_r
        
        y_min_l, bookkeeping = bin2int(bookkeeping[-8:]), bookkeeping[:-8]
        sign, bookkeeping = bookkeeping[-1], bookkeeping[:-1]
        # 没有合并过
        if sign == 1:
            for w in range(width):
                for h in range(1, height, 2):
                    if arr[w,h] <= y_min_l:
                        arr[w,h] -= 1
        # 合并过
        else:
            for w in range(width-1, -1, -1):
                for h in range(1, height, 2):
                    if arr[w,h] <= y_min_l:
                        arr[w,h] -= 1
            for w in range(width-1, -1, -1):
                for h in range(height-1, -1, -2):
                    if arr[w,h] == y_min_l+1:
                        item, bookkeeping = bookkeeping[-1], bookkeeping[:-1]
                        if item == 1:
                            arr[w,h] = y_min_l

        x_min_r, bookkeeping = bin2int(bookkeeping[-8:]), bookkeeping[:-8]
        sign, bookkeeping = bookkeeping[-1], bookkeeping[:-1]
        # 没有合并过
        if sign == 1:
            for w in range(width):
                for h in range(0, height, 2):
                    if arr[w,h] >= x_min_r:
                        arr[w,h] += 1
        # 合并过
        else:
            for w in range(width):
                for h in range(0, height, 2):
                    if arr[w,h] >= x_min_r:
                        arr[w,h] += 1
            for w in range(width-1, -1, -1):
                for h in range(height-2, -1, -2):
                    if arr[w,h] == x_min_r-1:
                        item, bookkeeping = bookkeeping[-1], bookkeeping[:-1]
                        if item == 1:
                            arr[w,h] = x_min_r
        
        x_min_l, bookkeeping = bin2int(bookkeeping[-8:]), bookkeeping[:-8]
        sign, bookkeeping = bookkeeping[-1], bookkeeping[:-1]
        # 没有合并过
        if sign == 1:
            for w in range(width):
                for h in range(0, height, 2):
                    if arr[w,h] <= x_min_l:
                        arr[w,h] -= 1
        # 合并过
        else:
            for w in range(width):
                for h in range(0, height, 2):
                    if arr[w,h] <= x_min_l:
                        arr[w,h] -= 1
            for w in range(width-1, -1, -1):
                for h in range(height-2, -1, -2):
                    if arr[w,h] == x_min_l+1:
                        item, bookkeeping = bookkeeping[-1], bookkeeping[:-1]
                        if item == 1:
                            arr[w,h] = x_min_l
    
    return arr
                

if __name__ == "__main__":
    # 参数设置
    path = 'en_kodim23.png'
    ref_path = 'kodim23.png'
    # 读取图片
    img = read_img(path=path, mode=cv2.IMREAD_GRAYSCALE)
    arr = img2arr(img=img)

    # 取消增强
    arr, S, bookkeeping, embedding_bits = de_enhance(arr=arr)

    # 取消预处理
    arr = de_preprocessing(arr=arr, bookkeeping=bookkeeping, S=S)
    
    # 对比原图
    ref_img = read_img(ref_path, cv2.IMREAD_GRAYSCALE)
    ref_arr = img2arr(ref_img)
    print(np.array_equal(ref_arr, arr))
    img = arr2img(arr)
    cv2.imshow('recover', img)
    cv2.waitKey(0)