import numpy as np 
import pickle

# with open('mask_extra.pkl', 'rb') as f:
#     mask_extra = pickle.load(f)
#     mask_extra = np.concatenate([mask_extra, np.zeros((mask_extra.shape[0], 7,mask_extra.shape[2]), dtype=bool)], axis=1).astype(bool)
mask_extra = None 

def clean_data(inpt, targs, missing_ratio=0.8, mask_ratio=0.8):
    # 随机按照mask进行掩膜，保证数据缺失特征相似
    if mask_extra is not None:
        B, N = inpt.shape[0], mask_extra.shape[0]
        selected_mask_indices = np.random.choice(N, size=B, replace=True)
        selected_masks = mask_extra[selected_mask_indices]  # shape = (B, C1, C2)
        inpt[selected_masks] = np.nan

    # 按照mask_ratio掩膜最后一天的数据(索引0)
    mask_end = np.random.choice([True,False], size=inpt.shape[0], p=[mask_ratio,1-mask_ratio])
    inpt[mask_end,:,0] = np.nan 


    # 去除全部缺失的数据
    cho_idx = np.ones(inpt.shape[0], dtype=bool)
    inpt_isnan = np.isnan(inpt)
    inpt_isnan = inpt_isnan.sum(axis=2).max(axis=1) == inpt.shape[2]
    cho_idx[inpt_isnan] = False

    # 进行一次整理
    inpt = inpt[cho_idx]
    targs = {vari:values[cho_idx] for vari, values in targs.items()}

    # 应该存在目标值
    cho_idx = np.ones(inpt.shape[0], dtype=bool)
    for vari, values in targs.items():
        targs_isnan = np.isnan(values)|np.isinf(values)
        cho_idx[targs_isnan] = False
    
    return inpt[cho_idx], {vari:values[cho_idx] for vari,values in targs.items()}

if __name__ == '__main__':
    from dataset_psave import multi_ds
    import config
    save_rpath = config.Phase1['save_rpath']
    ds = multi_ds(save_rpath, [], None)
    print(len(ds))
    print(ds.__getitem__(1)[0].shape)
    print(mask_extra.shape)