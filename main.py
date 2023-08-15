import cv2
import numpy as np
import random
from queue import Queue
from recover import *

def read_img(path, mode):
    return cv2.imread(path, mode)

def save_img(save_path, img):
    img = arr2img(arr=arr)
    # cv2.imshow('rslt', img)
    # cv2.waitKey(0)
    cv2.imwrite(save_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def arr2img(arr):
    return np.uint8(arr)

def img2arr(img):
    return np.asarray(img)

def calc_2D_hist(arr, width, height):
    histogram = np.zeros(shape=(256,256),dtype=np.int32)

    for i in range(width):
        for j in range(0, height, 2):
            index1 = arr[i,j]
            index2 = arr[i,j+1]
            histogram[index2,index1] += 1

    return histogram

def de_calc_2D_hist(arr, width, height, hist):
    # 不计算最后32个像素的直方图
    for j in range(height-32, height, 2):
        index1 = arr[width-1,j]
        index2 = arr[width-1,j+1]
        hist[index2,index1] -= 1

    return hist

def preprocessing(arr, hist, margin):
    # 图像的大小
    width, height = arr.shape[0], arr.shape[1]
    # 左像素的一维直方图
    x_hist = np.sum(hist, axis=0)
    # 右像素的一维直方图
    y_hist = np.sum(hist, axis=1)
    # 四个角的最矮bin
    x_min_l = np.where(x_hist[margin:128] == np.min(x_hist[margin:128]))[0].min() + margin
    x_min_r = np.where(x_hist[128:256-margin] == np.min(x_hist[128:256-margin]))[0].max() + 128
    y_min_l = np.where(y_hist[margin:128] == np.min(y_hist[margin:128]))[0].min() + margin
    y_min_r = np.where(y_hist[128:256-margin] == np.min(y_hist[128:256-margin]))[0].max() + 128
    # bookkeeping
    bookkeeping = np.array([], dtype=int)

    # 是空bin的情况-----------------------------------x_min_l-------------------------------------------
    if x_hist[x_min_l] == 0:
        bookkeeping = np.concatenate((bookkeeping, [1]), axis=0)
        for w in range(width):
            for h in range(0, height, 2):
                if arr[w, h] < x_min_l:
                    arr[w,h] += 1
    
    # 非空bin                
    else:
        # 合并_修改图像
        for w in range(width):
            for h in range(0, height, 2):
                if arr[w,h] == x_min_l + 1:
                    bookkeeping = np.concatenate((bookkeeping, [0]), axis=0)
                if arr[w,h] == x_min_l:
                    arr[w,h] += 1
                    bookkeeping = np.concatenate((bookkeeping, [1]), axis=0)
        # 移动bins_修改图像
        for w in range(width):
            for h in range(0, height, 2):
                if arr[w, h] < x_min_l:
                    arr[w,h] += 1
        bookkeeping = np.concatenate((bookkeeping, [0]), axis=0)
    bookkeeping = np.concatenate((bookkeeping, int2bin(x_min_l, 8)), axis=0)
    # 是空bin的情况-----------------------------------x_min_r-------------------------------------------
    if x_hist[x_min_r] == 0:
        bookkeeping = np.concatenate((bookkeeping, [1]), axis=0)
        # 修改图像
        for w in range(width):
            for h in range(0, height, 2):
                if arr[w, h] > x_min_r:
                    arr[w,h] -= 1
        
    # 非空bin
    else:
        # 合并_修改图像
        for w in range(width):
            for h in range(0, height, 2):
                if arr[w,h] == x_min_r - 1:
                    bookkeeping = np.concatenate((bookkeeping, [0]), axis=0)
                if arr[w,h] == x_min_r:
                    arr[w,h] -= 1
                    bookkeeping = np.concatenate((bookkeeping, [1]), axis=0)
        # 移动bins_修改图像
        for w in range(width):
            for h in range(0, height, 2):
                if arr[w, h] > x_min_r:
                    arr[w,h] -= 1
        bookkeeping = np.concatenate((bookkeeping, [0]), axis=0)
    bookkeeping = np.concatenate((bookkeeping, int2bin(x_min_r, 8)), axis=0)
    # 是空bin的情况-----------------------------------y_min_l-------------------------------------------
    if y_hist[y_min_l] == 0:
        bookkeeping = np.concatenate((bookkeeping, [1]), axis=0)
        # 修改图像  
        for w in range(width):
            for h in range(1, height, 2):
                if arr[w,h] < y_min_l:
                    arr[w,h] += 1
    # 非空bin
    else:
        # 合并_修改图像
        for w in range(width):
            for h in range(1, height, 2):
                if arr[w,h] == y_min_l + 1:
                    bookkeeping = np.concatenate((bookkeeping, [0]), axis=0)
                if arr[w,h] == y_min_l:
                    arr[w,h] += 1
                    bookkeeping = np.concatenate((bookkeeping, [1]), axis=0)
        # 移动bins_修改图像
        for w in range(width):
            for h in range(1, height, 2):
                if arr[w,h] < y_min_l:
                    arr[w,h] += 1
        bookkeeping = np.concatenate((bookkeeping, [0]), axis=0)
    bookkeeping = np.concatenate((bookkeeping, int2bin(y_min_l, 8)), axis=0)
    # 是空bin的情况-----------------------------------y_min_r-------------------------------------------
    if y_hist[y_min_r] == 0:
        bookkeeping = np.concatenate((bookkeeping, [1]), axis=0)
        # 修改图像
        for w in range(width):
            for h in range(1, height, 2):
                if arr[w,h] > y_min_r:
                    arr[w,h] -= 1
    # 非空bin
    else:
        # 合并_修改图像
        for w in range(width):
            for h in range(1, height, 2):
                if arr[w,h] == y_min_r - 1:
                    bookkeeping = np.concatenate((bookkeeping, [0]), axis=0)
                if arr[w,h] == y_min_r:
                    arr[w,h] -= 1
                    bookkeeping = np.concatenate((bookkeeping, [1]), axis=0)
        # 移动bins_修改图像
        for w in range(width):
            for h in range(1, height, 2):
                if arr[w,h] > y_min_r:
                    arr[w,h] -= 1
        bookkeeping = np.concatenate((bookkeeping, [0]), axis=0)
    bookkeeping = np.concatenate((bookkeeping, int2bin(y_min_r, 8)), axis=0)

    return arr, bookkeeping  # 返回bookkeeping arr


def int2bin(num, n):
    binary_str = bin(num)[2:]  # 将整数转换为二进制字符串，去掉前缀 '0b'
    binary_str = binary_str.zfill(n)  # 在字符串前面填充0，确保长度为n
    binary_array = np.array([int(bit) for bit in binary_str])  # 将二进制字符串转换为NumPy数组
    return binary_array

def bin2int(arr):
    binary_str = ''.join([str(bit) for bit in arr])  # 将二进制数组转换为二进制字符串
    decimal_value = int(binary_str, 2)  # 将二进制字符串转换为十进制整数
    return decimal_value

def enhance(arr, S, bookkeeping):
    width, height = arr.shape[0], arr.shape[1]
    # bookkeeping的长度
    length = int2bin(len(bookkeeping), 20)
    bookkeeping = np.concatenate((length, bookkeeping), axis=0)
    pqrs = Queue()
    idx_bk = 0
    idx_emb = 0
    # 前S-1轮次
    for iteration in range(S-1):
        hist = calc_2D_hist(arr=arr, width=width, height=height)
        # 左像素的一维直方图
        x_hist = np.sum(hist, axis=0)
        # 右像素的一维直方图
        y_hist = np.sum(hist, axis=1)
        # 四个角的最高bin
        x_max_l = np.where(x_hist[:128] == np.max(x_hist[:128]))[0].min()
        x_max_r = np.where(x_hist[128:] == np.max(x_hist[128:]))[0].max() + 128
        y_max_l = np.where(y_hist[:128] == np.max(y_hist[:128]))[0].min()
        y_max_r = np.where(y_hist[128:] == np.max(y_hist[128:]))[0].max() + 128
                    
        # 嵌入
        for w in range(width):
            for h in range(0, height, 2):
                if arr[w,h] == x_max_l:
                    if not pqrs.empty():
                        arr[w,h] -= pqrs.get()
                    elif idx_bk < len(bookkeeping):
                        arr[w,h] -= bookkeeping[idx_bk]
                        idx_bk += 1
                    else:
                        arr[w,h] -= random.randint(0, 1)
                        idx_emb += 1
                elif arr[w,h] == x_max_r:
                    if not pqrs.empty():
                        arr[w,h] += pqrs.get()
                    elif idx_bk < len(bookkeeping):
                        arr[w,h] += bookkeeping[idx_bk]
                        idx_bk += 1
                    else:
                        arr[w,h] += random.randint(0, 1)
                        idx_emb += 1
                elif arr[w,h] < x_max_l:
                    arr[w,h] -= 1
                elif arr[w,h] > x_max_r:
                    arr[w,h] += 1
                
        for w in range(width):
            for h in range(1, height, 2):
                if arr[w,h] == y_max_l:
                    if not pqrs.empty():
                        arr[w,h] -= pqrs.get()
                    elif idx_bk < len(bookkeeping):
                        arr[w,h] -= bookkeeping[idx_bk]
                        idx_bk += 1
                    else:
                        arr[w,h] -= random.randint(0, 1)
                        idx_emb += 1
                elif arr[w,h] == y_max_r:
                    if not pqrs.empty():
                        arr[w,h] += pqrs.get()
                    elif idx_bk < len(bookkeeping):
                        arr[w,h] += bookkeeping[idx_bk]
                        idx_bk += 1
                    else:
                        arr[w,h] += random.randint(0, 1)
                        idx_emb += 1
                elif arr[w,h] < y_max_l:
                    arr[w,h] -= 1
                elif arr[w,h] > y_max_r:
                    arr[w,h] += 1

        # pqrs记录
        p = int2bin(x_max_l, 8)
        for item in p:
            pqrs.put(item)
        q = int2bin(y_max_l, 8)
        for item in q:
            pqrs.put(item)
        r = int2bin(x_max_r, 8)
        for item in r:
            pqrs.put(item)
        s = int2bin(y_max_r, 8)
        for item in s:
            pqrs.put(item)

    # 最后一轮次
    hist = calc_2D_hist(arr, width, height)
    hist = de_calc_2D_hist(arr, width, height, hist)
    # 左像素的一维直方图
    x_hist = np.sum(hist, axis=0)
    # 右像素的一维直方图
    y_hist = np.sum(hist, axis=1)
    # 四个角的最高bin
    x_max_l = np.where(x_hist[:128] == np.max(x_hist[:128]))[0].min()
    x_max_r = np.where(x_hist[128:] == np.max(x_hist[128:]))[0].max() + 128
    y_max_l = np.where(y_hist[:128] == np.max(y_hist[:128]))[0].min()
    y_max_r = np.where(y_hist[128:] == np.max(y_hist[128:]))[0].max() + 128
    # 参数记录（S  最后32个像素的LSB）
    for item in int2bin(S, 8):
        pqrs.put(item)
    for h in range(height-32, height):
        pqrs.put(arr[width-1,h]%2)
    # 嵌入
    for w in range(width):
        for h in range(0, height, 2):
            if w == width-1 and h >= height-32:
                break

            if arr[w,h] == x_max_l:
                if not pqrs.empty():
                    arr[w,h] -= pqrs.get()
                elif idx_bk < len(bookkeeping):
                    arr[w,h] -= bookkeeping[idx_bk]
                    idx_bk += 1
                else:
                    arr[w,h] -= random.randint(0, 1)
                    idx_emb += 1
            elif arr[w,h] == x_max_r:
                if not pqrs.empty():
                    arr[w,h] += pqrs.get()
                elif idx_bk < len(bookkeeping):
                    arr[w,h] += bookkeeping[idx_bk]
                    idx_bk += 1
                else:
                    arr[w,h] += random.randint(0, 1)
                    idx_emb += 1
            elif arr[w,h] < x_max_l:
                arr[w,h] -= 1
            elif arr[w,h] > x_max_r:
                arr[w,h] += 1
    for w in range(width):
        for h in range(1, height, 2):
            if w == width-1 and h >= height-32:
                break
            if arr[w,h] == y_max_l:
                if not pqrs.empty():
                    arr[w,h] -= pqrs.get()
                elif idx_bk < len(bookkeeping):
                    arr[w,h] -= bookkeeping[idx_bk]
                    idx_bk += 1
                else:
                    arr[w,h] -= random.randint(0, 1)
                    idx_emb += 1
            elif arr[w,h] == y_max_r:
                if not pqrs.empty():
                    arr[w,h] += pqrs.get()
                elif idx_bk < len(bookkeeping):
                    arr[w,h] += bookkeeping[idx_bk]
                    idx_bk += 1
                else:
                    arr[w,h] += random.randint(0, 1)
                    idx_emb += 1
            elif arr[w,h] < y_max_l:
                arr[w,h] -= 1
            elif arr[w,h] > y_max_r:
                arr[w,h] += 1
    
    
    # pqrs记录
    p = int2bin(x_max_l, 8)
    for item in p:
        pqrs.put(item)
    q = int2bin(y_max_l, 8)
    for item in q:
        pqrs.put(item)
    r = int2bin(x_max_r, 8)
    for item in r:
        pqrs.put(item)
    s = int2bin(y_max_r, 8)
    for item in s:
        pqrs.put(item)

    # 替换LSB
    for h in range(height-32, height):
        item = pqrs.get()
        arr[width-1,h] = arr[width-1,h] - (arr[width-1,h]%2) + item

    return arr, idx_emb, idx_bk

if __name__ == "__main__":
    # 参数设置
    path = 'kodim23.png'
    save_path = 'en_kodim23.png'
    S = 10

    # 读取图像
    img = read_img(path=path, mode=cv2.IMREAD_GRAYSCALE)
    width, height = img.shape[0], img.shape[1]
    arr = img2arr(img=img)
    ref = arr
    hist = calc_2D_hist(arr=arr, width=width, height=height)
    bookkeeping = np.array([], dtype=int)

    # 图像预处理
    print("-----------------preprocessing-------------------------")
    for margin in range(S):
        arr, new_bk = preprocessing(arr=arr, hist=hist, margin=margin)
        bookkeeping = np.concatenate((bookkeeping, new_bk), axis=0)
        hist = calc_2D_hist(arr=arr, width=width, height=height)

    # RDHCE
    print("-----------------ennhancement-------------------------")
    arr, idx_emb, idx_bk = enhance(arr=arr, S=S, bookkeeping=bookkeeping)
    print("length of bookkeeping:{0}".format(len(bookkeeping) + 20))
    print("embeded pure bits:{0}".format(idx_emb))
    print("all bits:{0}".format(idx_bk + idx_emb + 32 * S + 8 + 32))

    # 图像保存
    print("-----------------save-------------------------")
    save_img(save_path=save_path, img=img)

    arr, S, bookkeeping, _ = de_enhance(arr)
    arr = de_preprocessing(arr, bookkeeping, S)

    print(np.array_equal(arr, ref))