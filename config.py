model = dict(Lx=21,
             d_model=256,
             seq_len=30,
             encoder_num=6,
             decoder_num=6,
             heads=4)

Phase1 = dict(save_rpath="/mnt/oc_data/",
              load_checkpoint=None,
              batch_size=16,
              epochs=30,
              lr=0.0002,
              weight_decay=1e-5,
              sub_batchs=100,
              save_model_path='saved_model-split [m1].pth')

Phase2 = dict(inner_path='./_P2temp/',
              cv_num=4,
              epochs=(100, 500, 200),
              sub_max=3000,
              lr=(0.001, 0.0002, 0.0001),
              weight_decay=(5e-6, 5e-6, 5e-6),
              ckpt='saved_model-split [m1].pth',
              sp_varis=["secchi_depth","pp"],
              save_model_path='saved_model-split [m2].pth',
              save_mat='data_m2.mat')

Phase3 = dict(device='cuda',
              sp_varis=["pp","secchi_depth"],
              batch_num=1,
              ckpt='saved_model-split [m1].pth',
              lr=0.001,
              weight_decay=1e-5,
              epochs=200,
              save_model_path='saved_model-split [m3].pth',
              save_mat='data_m3.mat',
              seed=None)

Generate = dict(rpath='/mnt/Aqua-MODIS/',
                app=['secchi_depth','pp'],
                # app=['secchi_depth','npp','pp','tchl_a','fuco','spm','kd490','z_01','a443'],
                # app=['chlor_a','pic','poc','a_443','bb_443','Kd_490','Ratio-547_443'],
                ckpt2='saved_model-split [m2].pth',
                ckpt3='saved_model-split [m3].pth',
                batch_num=100,
                save_rpath='/mnt/OC_generated/',
                bt='2020-01-01',
                et='2020-01-30',
                device='cuda:0')


