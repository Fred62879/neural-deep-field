import os
import torch
import pickle
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from pathlib import Path
from os.path import join
from astropy.visualization import ZScaleInterval
from wisp.utils.numerical import calculate_sam_spectrum, \
    calculate_precision_recall, calculate_precision_recall_together


def plot_line(x, y, fname, xlabel=None, ylabel=None, x_range=None):
    plt.scatter(x, y)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)
    if x_range is not None:
        lo, hi = x_range
        plt.xlim(lo, hi)
    plt.savefig(fname)
    plt.close()

def _debug(n, grad):
    print(n, grad[0,74:77])
    # permute ids is the order spectra is indexed
    # obtained from nerf::pretrain::idx
    # _permute_ids = torch.tensor([76, 16, 95, 62, 51, 69, 2, 81, 22, 37, 88, 38, 44, 24, 11, 14, 15, 27, 70, 45, 65, 30, 58, 68, 36, 78, 49, 6, 4, 32, 40, 8, 26, 75, 79, 83, 92, 98, 57, 82, 50, 77, 71, 54, 85, 21, 87, 34, 66, 94, 56, 72, 90, 86, 74, 84, 42, 53, 63, 73, 13, 47, 97, 55, 41, 48, 33, 1, 91, 96, 80, 7, 3, 60, 10, 31, 23, 46, 67, 29, 35, 52, 28, 93, 17, 64, 39, 20, 43, 12, 9, 59, 99, 18, 25, 0, 19, 89, 5, 61])

    # gt bin ids is ordered for each spectra
    _gt_bin_ids = torch.argmax( (grad[...,0] != 0).to(torch.long), dim=-1)
    # print('grad_plot:',  _gt_bin_ids[_permute_ids])
    # print('grad_plot:',  _gt_bin_ids)

def plot_grad_flow(named_parameters, gradFileName=None):
    layers, ave_grads = [], []
    for n, p in named_parameters:
        if "grid" not in n and (p.requires_grad) and ("bias" not in n):
            layers.append(n[-32:])
            grad = p.grad.detach().cpu()
            # _debug(n, grad)
            ave_grads.append(p.grad.detach().cpu().abs().mean())

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, fontsize=6, rotation=10)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers");plt.ylabel("average gradient")
    plt.title("Gradient flow");plt.grid(True)

def plot_multiple(n_per_fig, n_per_row, data, fname, x=None,
                  y2=None, vertical_xs=None, hist=False, titles=None):
    n = len(data)
    n_figs = int(np.ceil(n / n_per_fig))

    def _plot(axis, idx, x, y, hist):
        # plot a single subplot
        if x is None:
            if hist:
                bins = np.arange(len(y[idx]) + 1) - 0.5
                axis.hist(bins[:-1], bins, weights=y[idx])
            else: axis.plot(y[idx])
        else:
            if hist: axis.hist(x, weights=y[idx])
            else:    axis.plot(x, y[idx])

    for i in range(n_figs):
        lo = i * n_per_fig
        hi = min(lo + n_per_fig, n)
        cur_n = hi - lo

        ncols = min(n, n_per_row)
        nrows = int(np.ceil(cur_n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows))

        for j in range(cur_n):
            if titles is not None: title = f"{titles[i*n_per_fig + j]:.{3}f}"
            else: title = str(i*n_per_fig + j)

            if nrows == 1: axis = axs if ncols == 1 else axs[j%ncols]
            else:          axis = axs[j//ncols, j%ncols]
            _plot(axis, lo+j, x, data, hist)
            if y2 is not None: _plot(axis, lo+j, x, y2, hist=hist)

            # plot vertical line to indicate e.g. gt location
            if vertical_xs is not None:
                axis.axvline(x=vertical_xs[lo+j], color="red", linewidth=2, linestyle="--")

            axis.set_title(title)

        fig.tight_layout(); plt.savefig(f"{fname}-{i}"); plt.close()

def plot_save(fname, x, y):
    # assert(y.ndim <= 2)
    # if y.ndim == 2:
    #     for sub_y in y:
    #         plt.plot(x, sub_y)
    # else: plt.plot(x, y)
    plt.plot(x, y)
    plt.savefig(fname)
    plt.close()

def plot_precision_recall_individually(logits, gt_redshift, lo, hi, bin_width, n_per_row, fname):
    """ Plot precision recall for each spectra individually.
    """
    n = len(logits)
    ncols = min(n, n_per_row)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows))
    for i, (cur_logits, cur_gt_redshift) in enumerate(zip(logits, gt_redshift)):
        precision, recall = calculate_precision_recall(
            cur_logits, cur_gt_redshift, lo, hi, bin_width
        )
        if nrows == 1: axis = axs if ncols == 1 else axs[i%ncols]
        else:          axis = axs[i//ncols,i%ncols]
        axis.plot(recall, precision)
        axis.set_xlim(xmin=0,xmax=1.2); axis.set_ylim(ymin=-0.05,ymax=1)
        axis.set_xlabel("recall");axis.set_ylabel("precision")
    fig.tight_layout(); plt.savefig(fname); plt.close()

def plot_precision_recall_together(logits, gt_redshifts, lo, hi, bin_width, fname,
                                   num_precision_recall_residuals
):
    """ Plot precision recall combining all spectra together.
        @param
          logits: estimated redshift logits for each spectra [num_spectra,num_bins]
          gt_redshifts: gt redshift value for each spectra [num_spectra,]
          num_precision_recall_residuals: num residual levels to calculate the two stats
    """
    n = len(logits)
    residuals, precision, recall = calculate_precision_recall_together(
        logits, gt_redshifts, lo, hi, bin_width,
        num_precision_recall_residuals
    )
    if residuals is None: return # residuals all 0

    # plt.plot(recall, precision)
    # plt.xlim(xmin=0,xmax=1.2);plt.ylim(ymin=0,ymax=1.2)
    # plt.xlabel("recall");plt.ylabel("precision")
    # plt.title(f"Precision Recall over {n} spectra")
    # plt.tight_layout(); plt.savefig(fname + ".png"); plt.close()

    # plt.plot(residuals, precision)
    # plt.xlabel("residual"); plt.ylabel("precision")
    # plt.title(f"Precision under different residual levels")
    # plt.tight_layout(); plt.savefig(fname + "_precision.png"); plt.close()

    # plt.plot(residuals, recall)
    # plt.xlabel("residual"); plt.ylabel("recall")
    # plt.title(f"Recall under different residual levels")
    # plt.tight_layout(); plt.savefig(fname + "_recall.png"); plt.close()

    plt.plot(residuals, recall)
    plt.xlabel("residual"); plt.ylabel("accuracy")
    plt.title(f"Accuracy under different residual levels")
    plt.tight_layout(); plt.savefig(fname + "_accuracy.png"); plt.close()

def plot_latent_embed(latents, embed, fname, out_dir, plot_latent_only=False):
    """ Plot latent variable distributions and each codebook embedding.
    """
    if type(latents) is list:
        latents = torch.stack(latents)
    if type(latents).__module__ == "torch":
        if latents.device != "cpu":
            latents = latents.detach().cpu()
        latents = latents.numpy()

    latents = latents.reshape((-1,latents.shape[-1]))

    # plot latent variables only
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(latents[:,0],latents[:,1],latents[:,2],marker='v',color='orange')
    latent_dir = join(out_dir, "latents")
    Path(latent_dir).mkdir(parents=True, exist_ok=True)
    cur_fname = join(latent_dir, f"{fname}")
    np.save(cur_fname, latents)
    plt.savefig(cur_fname)
    plt.close()

    if plot_latent_only: return

    # plot embeddings only
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embed[0],embed[1],embed[2],marker='o',color='blue')
    embed_dir = join(out_dir, "embed")
    Path(embed_dir).mkdir(parents=True, exist_ok=True)
    cur_fname = join(embed_dir, f"{fname}")
    np.save(cur_fname, embed)
    plt.savefig(cur_fname)
    plt.close()

    # plot embeddings with latent variables
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embed[0],embed[1],embed[2],marker='o',color='blue')
    ax.scatter(latents[:,0],latents[:,1],latents[:,2],marker='v',color='orange')
    cur_fname = join(out_dir, f"{fname}")
    plt.savefig(cur_fname)
    plt.close()

def plot_embed_map(coords, embed_ids, embed_map_fn, fits_id):
    """ Plot embed map whose pixel value is the embed id of the corresponding pixel.
        Note, the embed map is plotted for one tile at a time.
        @Param
          coords:    coord of pixel to log embed id
          embed_ids: embed id array [num_rows,num_cols]
          fits_id:   fits id of current tile
    """
    import logging as log

    if embed_ids.ndim == 3:
        embed_ids = embed_ids[0]
    embed_ids = np.clip(embed_ids, 0, 255)

    pos = []
    for (r,c,cur_fits_id) in coords:
        if cur_fits_id != fits_id: continue
        cur_embed_id = embed_ids[r, c]
        log.info(f"embed id of {fits_id}_{r}_{c} is: {cur_embed_id}")

    plt.imshow(embed_ids, cmap='gray',origin='lower')
    plt.axis("off")
    plt.savefig(embed_map_fn, bbox_inches="tight")
    plt.close()

def plot_zscale(ax, data, vmin, vmax):
    ax.axis('off')
    ax.imshow(data, cmap='gray', interpolation='none', vmin=vmin,
              vmax=vmax, alpha=1.0, aspect='equal',origin='lower')

def plot_img_distrib_one_band(ax, data, bins=10, prob=True):
    """ Plot pixel vlaue distribution for one band.
    """
    lo, hi = np.min(data), np.max(data)
    rg = hi - lo
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_edges = np.array([ (a + b) / 2 for a, b in zip(bin_edges[:-1], bin_edges[1:]) ])
    if prob: hist = hist/np.sum(hist)
    # ax.set_xlim(lo - rg/10, hi + rg/10)
    # ax.set_ylim(0, 30)
    ax.plot(bin_edges, hist)

def plot_one_row(fig, r, c, lo, img, num_bands, plot_option, vmins, vmaxs, cal_z_range=False, **kwargs):
    """ Plot current img (multiband) in one row based on plot_option
        @Param
           fig: global figure
           r/c: size of fig
           lo: starting subfigure position
    """
    if plot_option == "plot_img" and cal_z_range:
        for i in range(num_bands):
            vmin, vmax = ZScaleInterval(contrast=.25).get_limits(img[i])
            vmins.append(vmin);vmaxs.append(vmax)

    for i in range(num_bands):
        ax = fig.add_subplot(r, c, lo+i+1)
        if plot_option == "plot_img":
            plot_zscale(ax, img[i], vmins[i], vmaxs[i])
        elif plot_option == "plot_distrib":
            plot_img_distrib_one_band(ax, img[i])
        elif plot_option == "plot_heat_map":
            # ax.imshow(img[i])
            heat(ax, img[i], **kwargs)
        else: raise ValueError()

    if cal_z_range:
        return vmins,vmaxs

def plot_horizontally(img, png_fname, plot_option="plot_img", zscale_ranges=None, save_close=True, **kwargs):
    """ Plot multiband image horizontally.
        Currently only supports plot one row.
        @Param
          img: multiband image [nbands,sz,sz]
          zscale_ranges: min and max value for zscaling [2,nbands]
    """
    if plot_option == "plot_img":
        if zscale_ranges is None:
            vmins, vmaxs = [], []
            cal_z_range = True
        else:
            (vmins, vmaxs) = zscale_ranges
            cal_z_range = False
    else: vmins, vmaxs, cal_z_range = None, None, False

    num_bands = img.shape[0]
    fig = plt.figure(figsize=(3*num_bands + 1,3))
    plot_one_row(fig, 1, num_bands, 0, img, num_bands,
                 plot_option, vmins, vmaxs, cal_z_range=cal_z_range,
                 **kwargs)
    fig.tight_layout()

    if save_close:
        plt.savefig(png_fname)
        plt.close()

def mark_on_img(png_fname, img, coords, markers, zscale=True):
    if img.ndim == 3: img = img[0:1]

    plot_horizontally(img, "", "plot_img", save_close=False)
    for i, coord in enumerate(coords):
        plt.scatter(coord[1], coord[0], marker=markers[i])
    plt.savefig(png_fname)
    plt.close()

def plot_simple(img, png_fname):
    if img.ndim == 3:
        for band in img:
            plt.imshow(band)
    else:
        assert(img.ndim == 2)
        plt.imshow(img)
    plt.savefig(png_fname)
    plt.close()

############
# heatmap plot
############

def heat_range(arr,lo,hi):
    ''' only show heats within range '''
    my_cmap = copy(plt.cm.YlGnBu)
    my_cmap.set_over("blue")
    my_cmap.set_under("white")
    g = sns.heatmap(arr, vmin=lo, vmax=hi, xticklabels=False, yticklabels=False, cmap=my_cmap) #, linewidths=1.0, linecolor="grey")
    #g.set_xticklabels(labels, rotation=rotation)
    g.tick_params(left=False, bottom=False)
    #g.set_title("Semantic Textual Similarity")

def heat(ax, arr, resid_lo=None, resid_hi=None):
    if resid_lo is not None and resid_hi is not None:
        arr = np.clip(arr, resid_lo, resid_hi)
    ax.axis('off')
    img=ax.imshow(arr, cmap='viridis', origin='lower')
    plt.colorbar(img,ax=ax)

def heat_all(data, fig=None, fn=None, los=None, his=None):
    nbands = len(data)
    if fig is None:
        fig = plt.figure(figsize=(20,5))
    r, c = 1, nbands
    for i, band in enumerate(data):
        if los is not None and his is not None:
            lo, hi = los[i], his[i]
        else: lo, hi = None, None
        ax = fig.add_subplot(r, c, i+1)
        heat(ax, band)
    fig.tight_layout()
    if fn is not None:
        plt.savefig(fn)
        plt.close()

def annotated_heat(coords, markers, data, fn, fits_id, los=None, his=None):
    """ Plot heat map with markers for given coordinate positions.
        Currently only used to plot heatmap for redshift.
        @Param
          coords:  [n,2/3]: r,c(,fits id)
          markers: markers choices, different for each coord
          data:    data to heat [1,num_rwos,num_cols]
          fits_id: fits id of current tile, only draw coord with same fits id
    """
    import logging as log

    nbands = len(data)
    fig = plt.figure(figsize=(20,5))
    r, c = 1, nbands
    for i, band in enumerate(data):
        if los is not None and his is not None:
            lo, hi = los[i], his[i]
        else: lo, hi = None, None
        ax = fig.add_subplot(r, c, i+1)
        heat(ax, band, lo, hi)
    fig.tight_layout()

    # for (y, x, cur_fits_id), marker in zip(coords, markers):
    for (y, x), marker in zip(coords, markers):
        # if cur_fits_id != fits_id: continue
        plt.scatter(x, y, marker=marker)
        cur_redshift = data[0,y,x]
        # log.info(f"redshift value of {fits_id}_{y}_{x} is: {cur_redshift}")

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

###########
# not used
###########
'''
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

############
# Histogram plot
############

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
'''
