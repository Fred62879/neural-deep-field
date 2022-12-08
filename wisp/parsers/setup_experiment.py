

def setup_ablation(config):
    cho = config['abl_cho']
    config['loss_cho'] = ['l1','l1','l1', 'l1','l1','l1', 'l2','l2','l2', 'l2','l2','l2'][cho]
    config['weight_train'] = [True,True,True, False,False,False, True,True,True, False,False,False][cho]
    config['output_norm_cho'] = ['identity','arcsinh','sinh','identity','arcsinh','sinh','identity','arcsinh','sinh','identity','arcsinh','sinh'][cho]
    config['infer_output_norm_cho'] = ['identity','identity','sinh','identity','identity','sinh','identity','identity','sinh','identity','identity','sinh'][cho]

def setup_experiment(config):
    ''' array job, experiment setups, redefine corresponding argument '''

    if config['gaus_cho'] is not None:
        cho = config['gaus_cho']
        #sigmas = [1,15,30, 1,15,30, 1,15,30]
        #omegas = [100,100,100, 150,150,150, 200,200,200]
        sigmas = [1,1,1,1,1,  5,5,5,5,5,  10,10,10,10,10]
        omegas = [1,2,3,5,10, 1,2,3,5,10, 1,2,3,5,10]
        config['gaus_sigma'] = float(sigmas[cho])
        config['gaus_omega'] = float(omegas[cho])

    if config['fourier_mfn_cho'] is not None:
        cho = config['fourier_mfn_cho']
        scales = [5,5,5,5, 10,10,10,10]
        omegas = [50,100,150,200, 50,100,150,200]
        #scales = [5,5,5, 10,10,10]
        #omegas = [200,250,300, 200,250,300]
        config['mfn_w_scale'] = scales[cho]
        config['mfn_omega'] = omegas[cho]

    if config['gabor_mfn_cho'] is not None:
        cho = config['gabor_mfn_cho']
        #scales = [5,5,5, 10,10,10]
        #omegas = [200,250,300, 200,250,300]
        config['mfn_w_scale'] = 10 #scales[cho]
        config['mfn_omega'] = 250 #omegas[cho]
        config['mfn_alpha'] = [1,4,6][cho]
        config['mfn_beta'] = [1,1,1][cho]

    if config['inpaint_cho'] != 'no_inpaint' and \
       config['sample_ratio_cho'] is not None:

        ratios = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
        config['sample_ratio'] = ratios[config['sample_ratio_cho']]

    # redefine experiment id
    if config['para_nms'] is not None:
        eid = ''
        for para_nm in config['para_nms']:
            eid += str(config[para_nm]) + '_'
        config['experiment_id'] = eid[:-1]

def setup_experiments(config):
    if config['abl_cho'] is not None:
        setup_ablation(config)
    setup_experiment(config)
