import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import healpy as hp


from classtools.users.djw.tools import get_TT, get_TE, get_EE, get_BB


def mollview_pol(IQU, mask=None, coord='G', polcolor='white', polamp=True,
        axes=None, length=3,
        vecwidth=0.015, polsmooth=4, polnside=16, cbar=True, unit='', title='', **plt_dict):
    I, Q, U = IQU
    P = np.sqrt(hp.ma(Q)**2 + hp.ma(U)**2)
    if polamp:
        map_back = np.copy(P)
    else:
        map_back = np.copy(I)
    IQU_smooth = hp.smoothing(IQU, fwhm=polsmooth*np.pi/180, verbose=False)
    I, Q, U = hp.ud_grade(IQU_smooth, polnside)
    P = np.sqrt(hp.ma(Q)**2 + hp.ma(U)**2)

    ang = np.rad2deg(np.arctan(hp.ma(U)/hp.ma(Q))*0.5) + np.sign(hp.ma(U))*90*(hp.ma(Q)<0)
    if mask is not None:
        P[mask==0] = 0
    if type(map_back) == np.ndarray:
        m = map_back
    else:
        m = map_back.data
    m[m==0] = hp.UNSEEN
    m[~np.isfinite(m)]= hp.UNSEEN
    projmap = hp.mollview(m, return_projected_map=True,)
    plt.close()
    Gprojector = hp.projector.MollweideProj(coord=coord)
    if axes is None:
        fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(8.5, 5.4))
    if plt_dict and 'vmin' in plt_dict.keys():
        im = axes.imshow(projmap, origin='lower', extent=Gprojector.get_extent(), **plt_dict)
    else:
        scale = 5*(np.quantile(map_back, 0.60)-np.quantile(map_back, 0.40))
        im = axes.imshow(projmap, origin='lower', extent=Gprojector.get_extent(), 
                    vmin=-scale, vmax=scale, **plt_dict)
    P_nside = hp.get_nside(P)
    lon, lat = hp.pix2ang(nside=P_nside, ipix=np.arange(hp.nside2npix(P_nside)), lonlat=True)
    xy = Gprojector.ang2xy(lon, lat, lonlat=True)
    _mask = P>0
    # The scale argument is "number of data units per vector length"
    f_pol = P[_mask]/max(np.isfinite(P[_mask]))
    axes.quiver(xy[0][_mask], xy[1][_mask], 
                f_pol*np.sin(np.deg2rad(ang[_mask]))*length, 
                f_pol*np.cos(np.deg2rad(ang[_mask]))*length, 
                pivot='mid', headaxislength=0, color=polcolor, headlength=0,
                units='inches', linestyle='-', 
                width=vecwidth, headwidth=0,
                scale_units='inches', scale=10,
                )
    axes.set_xlim(Gprojector.get_extent()[0], Gprojector.get_extent()[1])
    axes.set_ylim(Gprojector.get_extent()[2], Gprojector.get_extent()[3])
    if plt_dict and 'cmap' in plt_dict.keys():
        cm = mpl.cm.get_cmap(plt_dict['cmap'])
    else:
        cm = mpl.cm.get_cmap('viridis')
    cm.set_bad('white')

    b = im.norm.inverse(np.linspace(0, 1, im.cmap.N + 1))
    v = np.linspace(im.norm.vmin, im.norm.vmax, im.cmap.N)
    nlocs, norm = 2, None
    fig.colorbar(im, orientation='horizontal', label=unit,
                    shrink=0.5,
                    aspect=25,
                    pad=0.05,
                    fraction=0.1,
                    ticks=hp.projaxes.BoundaryLocator(nlocs, norm),
                    boundaries=b,
                    values=v,
    )
    plt.title(title)
    axes.axis('off')
    plt.tight_layout()


def test(nside=256):
    np.random.seed(10)
    lmax = 3*nside - 1
    TT = get_TT(lmax=lmax)
    EE = get_EE(lmax=lmax)
    BB = get_BB(lmax=lmax)
    TE = get_TE(lmax=lmax)

    Cls = np.array([TT, EE, BB, TE])
    m = hp.synfast(Cls, nside, new=True, fwhm=np.pi/180, verbose=False)
    plt.figure(figsize=(16, 8))
    hp.mollview(m[0], min=-250, max=250, cmap='coolwarm', sub=231, title='I')
    hp.mollview(m[1], min=-2, max=2, cmap='coolwarm', sub=232, title='Q')
    hp.mollview(m[2], min=-2, max=2, cmap='coolwarm', sub=233, title='U')
    gamma = np.arctan2(m[2], m[1])*0.5
    hp.mollview(gamma, cmap='twilight_shifted', sub=235, title=r'$\gamma$')
    hp.mollview(np.sqrt(m[1]**2+m[2]**2), min=0, max=2, cmap='viridis', sub=236,
            title=r'$P$')

    mollview_pol(m, polnside=8, polsmooth=0, length=4, polamp=False,
               vmin=-250, vmax=250, cmap='coolwarm', polcolor='k',
               title='Intensity with polarization vectors')

    mollview_pol(m, polnside=16, polamp=False,
               vmin=-250, vmax=250, cmap='coolwarm', polcolor='k',
               title='Intensity with polarization vectors')

    mollview_pol(m, polnside=32, length=1, polamp=False,
               vmin=-250, vmax=250, cmap='coolwarm', polcolor='k',
               title='Intensity with polarization vectors')

    mollview_pol(m, polnside=16, polsmooth=0, polamp=False,
               vmin=-250, vmax=250, cmap='coolwarm', polcolor='k',
               title='Intensity with polarization vectors')

    mollview_pol(m, polnside=32, polsmooth=0, length=1, polamp=False,
               vmin=-250, vmax=250, cmap='coolwarm', polcolor='k',
               title='Intensity with polarization vectors')

    mollview_pol(m, polnside=16, polamp=True,
               vmin=0, vmax=2, cmap='viridis', title='Polarization amplitude')
    plt.show()



    return


if __name__ == '__main__':
    test()
