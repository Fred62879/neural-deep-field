import os
import torch
import pickle
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from os.path import join
from astropy.visualization import ZScaleInterval
from wisp.utils.numerical import calculate_sam_spectrum


#######################
# basic plotting funcs
#######################

def plot_save(fname, x, y):
    plt.plot(x, y)
    plt.savefig(fname);
    plt.close()

def plot_latent_embedding(model_id, smpl_latent_dir, out_dir,
                          model_dict=None, plot_latent_only=False):

    # plot latent variables only
    latent = np.load(join(smpl_latent_dir, str(model_id)+'.npy'))
    latent = latent.reshape((-1,latent.shape[-1]))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(latent[:,0],latent[:,1],latent[:,2],marker='v',color='orange')
    out_fn = join(out_dir, str(model_id)+'_latent.png')
    plt.savefig(out_fn)
    plt.close()

    if plot_latent_only: return

    assert(model_dict is not None)
    for n,p in model_dict.items():
        if 'codebook' in n:
            embd = p.T
            break
    embd = np.array(embd.cpu())

    # plot embddings only
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embd[:,0],embd[:,1],embd[:,2],marker='o',color='blue')
    out_fn = join(out_dir, str(model_id)+'_embd.png')
    plt.savefig(out_fn)
    plt.close()

    # plot embeddings with latent variables
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embd[:,0],embd[:,1],embd[:,2],marker='o',color='blue')
    ax.scatter(latent[:,0],latent[:,1],latent[:,2],marker='v',color='orange')
    out_fn = join(out_dir, str(model_id)+'.png')
    plt.savefig(out_fn)
    plt.close()

def plot_embd_map(embd_ids, embd_map_fn):
    num_ids = len(set(list(embd_ids.flatten())))
    embd_ids = np.clip(embd_ids, 0, 255)
    plt.imshow(embd_ids, cmap='gray',origin='lower')
    plt.savefig(embd_map_fn)
    plt.close()

def plot_zscale(ax, data, vmin, vmax):
    ax.axis('off')
    ax.imshow(data, cmap='gray', interpolation='none', vmin=vmin,
              vmax=vmax, alpha=1.0, aspect='equal',origin='lower')

def plot_one_row(fig, r, c, lo, img, vmins, vmaxs, num_bands, cal_z_range=False):
    if cal_z_range:
        vmins, vmaxs = [], []
        for i in range(num_bands):
            vmin, vmax = ZScaleInterval(contrast=.25).get_limits(img[i])
            vmins.append(vmin);vmaxs.append(vmax)

    for i in range(num_bands):
        ax = fig.add_subplot(r, c, lo+i+1)
        plot_zscale(ax, img[i], vmins[i], vmaxs[i])

    if cal_z_range:
        return vmins,vmaxs

def plot_gt_recon(gt, recon, fn, cal_z_range=False):
    n = 1
    nchls = gt.shape[0]
    fig = plt.figure(figsize=(20,2*n+2))
    r, c = n+1, nchls
    vmins, vmaxs = plot_one_row(fig, r, c, 0, gt, [], [], True, nchls)
    plot_one(fig, r, c, nchls, recon, vmins, vmaxs, nchls, cal_z_range=cal_z_range)
    fig.tight_layout()
    plt.savefig(fn)
    plt.close()

def plot_horizontally(img, png_fname, zscale_ranges=None):
    """ Plot multiband image horizontally
        @Param
          img: multiband image [nbands,sz,sz]
          zscale_ranges: min and max value for zscaling [2,nbands]
    """
    if zscale_ranges is None:
        vmins, vmaxs = [], []
        cal_z_range = True
    else:
        (vmins, vmaxs) = zscale_ranges
        cal_z_range = False

    num_bands = img.shape[0]
    fig = plt.figure(figsize=(3*num_bands + 1,3))
    plot_one_row(fig, 1, num_bands, 0, img, vmins, vmaxs, num_bands, cal_z_range=cal_z_range)
    fig.tight_layout()
    plt.savefig(png_fname)
    plt.close()

def sdss_rgb(imgs, bands, scales=None, m = 0.02):
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)

    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    return rgb

def generate_gaus_img(sz):
    # Initializing value of x-axis and y-axis
    x, y = np.meshgrid(np.linspace(-1,1,sz), np.linspace(-1,1,sz))
    dst = np.sqrt(x*x+y*y)
    sigma, muu = 1, 0.000
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    return gauss

def plotTrans(l, t):
    colors = ['r','g','b','c','m']
    lbs=['g','r','i','z','y']
    for i in range(5):
        plt.plot(l, t[i], colors[i], label=lbs[i])
        plt.scatter(l, t[i], label=lbs[i])
    plt.legend(loc="upper right")
    plt.show()

def plot_trans(wave_fn, trans_fn):
    #wave_fn='../../data/pdr3_input/10/transmission/full_wave.npy'
    #trans_fn='../../data/pdr3_input/10/transmission/full_trans.npy'
    a=np.load(wave_fn)
    b=np.load(trans_fn)
    lbs =['g', 'r', 'i', 'z', 'y', 'nb387', 'nb816', 'nb921','u','u*']
    colors = ['green','red','blue','gray','yellow','gray','red','blue','yellow','blue']
    styles=['solid','solid','solid','solid','solid','dashed','dashed','dashed','dashdot','dashdot']
    for i in range(10):
	    plt.plot(a,b[i],color=colors[i],label=lbs[i],linestyle=styles[i])
    #plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0)
    plt.legend(loc='upper right')
    plt.show()

def plotLoss(lossfn, nepoch, intvl, title):
    loss=np.load(lossfn)
    x = np.arange(0, nepoch + 1, intvl)
    plt.plot(x, loss, 'r')
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('ncc')
    plt.show()

def plotLoss(n, lossfn, nepoch, intvl, title):
    colors=['g','r','b','c','m']
    lbs = ['g','r','i','z','y']
    loss=np.load(lossfn).T
    x = np.arange(0, nepoch + 1, intvl)
    for i in range(n):
        plt.plot(x, loss[i], colors[i], label=lbs[i])
    plt.title(title);plt.xlabel('epochs');plt.ylabel('loss')
    plt.legend(loc="upper right");plt.show()

def plotImageHist(img, bins=10, prob=True, fn=None):
    lo, hi = np.min(img), np.max(img)
    rg = hi - lo
    hist, bin_edges = np.histogram(img, bins=bins)
    bin_edges = np.array([ (a + b) / 2 for a, b in zip(bin_edges[:-1], bin_edges[1:]) ])
    if prob: hist = hist/np.sum(hist)
    plt.figure()#plt.xlim(lo - rg/10, hi + rg/10)
    plt.ylim(0, 30)
    plt.plot(bin_edges, hist)
    if fn is not None: plt.savefig(fn);plt.close()
    else: plt.show()

def plotArr(arr, lb, color):
    x = np.arange(0, len(arr))
    if arr.ndim == 1:
        plt.plot(x, arr, color, label=lb)
    else:
        for sub in arr:
            plt.plot(x, sub, color, label=lb)

def plotGray(arr, fn):
    ndim = arr.ndim
    if ndim == 2: plt.imshow(arr, cmap='gray')
    else:
        if ndim == 3:
            nchl, _, _ = arr.shape
            n = 1
        elif ndim == 4: n, nchl, _, _ = arr.shape

        fig = plt.figure(figsize=(8, 8))
        for i in range(n):
            for j in range(nchl):
                ax = fig.add_subplot(n, nchl, i * nchl + j + 1)
                curimg = arr[j] if ndim == 3 else arr[i][j]
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                show = ax.imshow(curimg, cmap='gray')
    plt.savefig(fn)
    plt.close()

def batchPlot(path, option=0, vmins=None, vmaxs=None, fixedCutoutImgId=None):
    files = os.listdir(path)
    for fn in files:
        if 'npy' not in fn or 'spectra' in fn: continue

        cfn = join(path, fn)
        arr = np.load(cfn) #.transpose((1, 0, 2, 3))
        outfn = cfn.replace('.npy', '.png')

        if option == 2:   # plot gray image
            plotGray(arr, outfn)
        elif vmins is None or vmaxs is None:
            plotZscale(arr, outfn)
        elif option == 0: # plot fixed cutout
            plotZscale(arr, outfn, vmins[fixedCutoutImgId], vmaxs[fixedCutoutImgId])
        elif option == 1: # entire reconstruced image, fn be like x_y.npy/x.npy
            imgId = int(fn[0])
            plotZscale(arr, fileName=outfn, vmins=vmins[imgId], vmaxs=vmaxs[imgId])

def getBound(img):
    vmin, vmax = ZScaleInterval(contrast=.25).get_limits(img)
    return vmin, vmax

def getBounds(dataDir, groups):
    n, chl = groups.shape
    vmins, vmaxs = np.zeros((n, chl)), np.zeros((n, chl))
    for (i, group) in enumerate(groups):
        for (j, fn) in enumerate(group):
            id = 0 if 'Mega-u' in fn else 1
            hdu_list = fits.open(join(dataDir, fn), memmap=True)
            data = hdu_list[id].data
            lo, hi = getBound(data)
            vmins[i][j], vmaxs[i][j] = lo, hi
    return vmins, vmaxs

def plotZscaleOne(img, vmin=None, vmax=None, fn=None):
    if not vmin or not vmax: vmin, vmax = getBound(img)
    plt.imshow(img, cmap='gray', interpolation='none', vmin=vmin,
               vmax=vmax, alpha=1.0, aspect='equal')
    plt.axis('off')
    if fn is None: plt.show()
    else: plt.savefig(fn, bbox_inches=0)

# vmins/vmaxs: nchl x 1
def plotZscale(imgs, fileName=None, vmins=None, vmaxs=None):
    if type(imgs).__name__ == 'Tensor':
        imgs = imgs.detach().numpy()

    dim = imgs.ndim
    if dim == 4:
        n, nchl, _, _ = imgs.shape
    elif dim == 3:
        nchl, _, _ = imgs.shape
        n = 1

    fig = plt.figure(figsize=(8, 8))
    for i in range(n):
        for j in range(nchl):
            ax = fig.add_subplot(n, nchl, i * nchl + j + 1)
            curimg = imgs[j] if dim == 3 else imgs[i][j]
            if vmins is None or vmaxs is None:
                vmin, vmax = ZScaleInterval(contrast=.25).get_limits(curimg)
            else: vmin, vmax = vmins[j], vmaxs[j]

            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            show = ax.imshow(curimg, cmap='gray', interpolation='none',
                             vmin=vmin, vmax=vmax, alpha=1.0)
    if fileName: fig.savefig(fileName)
    plt.close(fig)

def stackPlot(dir, nimg, model_id):
    imgs = []
    for i in range(nimg):
        imgs.append(np.load(join(dir, str(i)+'_'+str(model_id)+'.npy')))
    imgs = np.array(imgs) # nimg x nchl x 4100 x 4100

def plot_sample_histogram(counts, distrib, wave, trans, args):
    if args.mc_cho == 1:
        res, num_bands = [], args.num_bands
        for i in range(num_bands):
            cur_counts = counts[i].detach().cpu().numpy()
            res.append(cur_counts)
            plt.plot(wave[i], counts[i])
        with open(args.counts_fn+'txt', 'wb') as fp:
            pickle.dump(res, fp)

    elif args.mc_cho == 2:
        counts = counts.detach().cpu().numpy()
        # np.mean(counts)/nepochs =~ nsmpl_per_epoch/full_nsmpl
        np.save(args.counts_fn+'npy', counts)
        plt.plot(wave, counts)

    plt.savefig(args.sample_hist_fn)
    plt.close()

def plot_grad_flow(named_parameters, gradFileName=None):
    layers, ave_grads = [], []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            #if 'decoder' in n: continue
            layers.append(n[:20])
            #if 'codebook' in n:
            #    print('**grad**', p.grad)
            ave_grads.append(p.grad.detach().cpu().abs().mean())

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, fontsize=8, rotation=30)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers");plt.ylabel("average gradient")
    plt.title("Gradient flow");plt.grid(True)
    if gradFileName: plt.savefig(gradFileName)

def plot_mask(sz, dim, mask, fn):
    fig = plt.figure(figsize=(2*dim,4))
    mask = mask.reshape(sz, sz, dim)
    for i in range(dim):
        ax = fig.add_subplot(1, dim, i+1)
        ax.axis('off')
        #ax.patch.set_edgecolor('black')
        #ax.patch.set_linewidth('1')
        ax.imshow(mask[:,:,i], cmap='gray')#, vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(fn)
    plt.close()

def batch_plot_mask(sz, dim, path):
    #batch_plot_mask(64, 10, '../data/pdr3_output/sampled_id/spectral1')
    for fn in os.listdir(path):
        if not str(sz) in fn or not 'npy' in fn: continue
        mask_fn = join(path, fn)
        plot_mask(sz, dim, np.load(mask_fn), mask_fn[:-4]+'.png')

def hist(fig, bins, img, r, c, id, lo, hi, nm, clip_v2, w, xhi, yhi):
    if clip_v2:
        v1, v2 = ZScaleInterval(contrast=.25).get_limits(img)
        selected = [img<=v2]
        img = img[selected]
        if w is not None:
            img *= (w[selected])

    ax = fig.add_subplot(r,c,id)
    hist, bin_edges = np.histogram(img, bins=bins)
    bin_edges = np.array([ (a + b) / 2
                           for a, b in zip(bin_edges[:-1], bin_edges[1:]) ])
    rg = hi - lo
    if xhi is not None: ax.set_xlim(lo - rg/10, xhi)
    if yhi is not None: ax.set_ylim(0, yhi)
    ax.title.set_text(nm)
    ax.plot(bin_edges, hist)

def batch_hist(gt, imgs, bins, chnl, nms, xhi=None, yhi=None, clip_v2=False, weight_fn=None):
    # same resolution (for large value)
    #nms=['iu','s2s','pnp-dip']
    #batch_hist(gt, imgs, 10, 9, nms, 300, 30)

    n = len(imgs) + 1
    r, c = 1, n
    fig = plt.figure(figsize=(3*n,3))

    if weight_fn is not None:
        w = np.load(weight_fn)
        n = w.shape[0]
        sz = int(np.sqrt(n))
        w = w.reshape((sz,sz,-1))[...,chnl]
    else: w = None

    if gt.ndim == 3: gt = gt[chnl]
    lo, hi = np.min(gt), np.max(gt)
    hist(fig, bins, gt, r, c, 1, lo, hi, 'gt', clip_v2, w, xhi, yhi)

    for i, img in enumerate(imgs):
        if img.ndim == 3 and img.shape[0] != 1:
            img = img[chnl]
        hist(fig, bins, img, r, c, i+2, lo, hi, nms[i], clip_v2, w, xhi, yhi)
    plt.show()

def batch_hist(gt_fn, img_fns, bins, chnl, nms, xhi=None, yhi=None, clip_v2=False, weight_fn=None):
    # same resolution (for large value)
    #nms=['iu','s2s','pnp-dip']
    #batch_hist(gt_fn, img_fns, 10, 9, nms, 300, 30)
    #img_fns=['../../data/pdr3_output/iu_output/10/trail_7250/siren_179_15_0/recons/10.npy','../../data/pdr3_output/s2s_output/10/trail_7250/179_2/recons/10.npy','../../data/pdr3_output/pnp_dip_output/10/trail_7250/179_2/recons/10.npy']

    n = len(img_fns) + 1
    r, c = 1, n
    fig = plt.figure(figsize=(3*n,3))

    if weight_fn is not None:
        w = np.load(weight_fn)
        n = w.shape[0]
        sz = int(np.sqrt(n))
        w = w.reshape((sz,sz,-1))[...,chnl]
    else: w = None

    gt = np.load(gt_fn)[chnl]
    lo, hi = np.min(gt), np.max(gt)
    hist(fig, bins, gt, r, c, 1, lo, hi, 'gt', clip_v2, w, xhi, yhi)

    for i, img_fn in enumerate(img_fns):
        img = np.load(img_fn)
        if img.shape[0] != 1:
            img = img[chnl]
        hist(fig, bins, img, r, c, i+2, lo, hi, nms[i], clip_v2, w, xhi, yhi)
    plt.show()

def batch_hist(gt_fn, img_fns, bins, yhi, chnl, nms):
    # diff resolution
    #nms=['iu','s2s','pnp-dip']
    #batch_hist(gt_fn, img_fns, 40, 30, 9, nms)

    n = len(img_fns) + 1
    r, c = 1, n
    fig = plt.figure(figsize=(3*n,3))

    gt = np.load(gt_fn)[chnl]
    hist(fig, bins, gt, r, c, 1,'gt')

    for i, fn in enumerate(img_fns):
        img = np.load(fn)[chnl]
        hist(fig, img, r, c, i+2, nms[i])
    plt.show()

def heat_range(arr,lo,hi):
    ''' only show heats within range '''
    my_cmap = copy(plt.cm.YlGnBu)
    my_cmap.set_over("blue")
    my_cmap.set_under("white")
    g = sns.heatmap(arr, vmin=lo, vmax=hi, xticklabels=False, yticklabels=False, cmap=my_cmap) #, linewidths=1.0, linecolor="grey")
    #g.set_xticklabels(labels, rotation=rotation)
    g.tick_params(left=False, bottom=False)
    #g.set_title("Semantic Textual Similarity")

# plot residual heat map
def heat(fig, arr, r, c, i):
    ax = fig.add_subplot(r, c, i)
    ax.axis('off')
    img=ax.imshow(arr, cmap='viridis', origin='lower')
    plt.colorbar(img,ax=ax)

def heat_all(resid, fn, los=None, his=None):
    nbands = len(resid)
    fig = plt.figure(figsize=(20,5))
    r, c = 1, nbands
    for i, band in enumerate(resid):
        if los is not None and his is not None:
            band = np.clip(fig, los[i], his[i])
        heat(fig, band, r, c, i+1)

    fig.tight_layout()
    plt.savefig(fn)
    plt.close()

def batch_heat(arrs, lo=None, hi=None, heat_range=True):
    #names=['ours','s2s','pnp-dip']

    def heat_for_range(fig, arr, r, c, id, name):
        ax = fig.add_subplot(r,c,id)
        ax.axis('off')
        g = sns.heatmap(arr, vmin=lo, vmax=hi, xticklabels=False, yticklabels=False, cmap=my_cmap)
        g.tick_params(left=False, bottom=False)
        g.set_title(name)

    def heat(fig, arr, r, c, id):
        ax = fig.add_subplot(r,c,id)
        ax.axis('off')
        im = ax.imshow(arr, cmap='viridis',origin='lower')
        #im1 = ax1.imshow(m1, interpolation='None')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    my_cmap = copy(plt.cm.YlGnBu)
    #my_cmap = copy(plt.cm.BuPu)
    my_cmap.set_over("blue")
    my_cmap.set_under("white")

    n = len(arrs)
    r, c = 1, n
    fig = plt.figure(figsize=(3*n,3))
    for i, (arr, name) in enumerate(zip(arrs,names)):
        if lo is not None and hi is not None and not heat_range:
            arr = np.clip(arr, lo, hi)
        if heat_range:
            heat_for_range(fig, arr, r, c, i+1, name)
        else:
            heat(fig, arr, r, c, i+1)

    plt.show()
