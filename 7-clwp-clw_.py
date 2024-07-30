
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:21:42 2024

@author: 15872
"""

import xarray as xr
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as crs
from matplotlib.cm import get_cmap
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from datetime import datetime
import pandas as pd 
import string
import datetime
import math
import warnings
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.ticker as mticker
import cartopy.io.shapereader as shpreader
import shapefile
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Times New Roman'



#########MWTS3
def set_map_axes(ax, proj, crs):
    dlon, dlat = 5, 5
    xticks = np.arange(70, 145 + dlon, dlon)
    yticks = np.arange(10, 60 + dlat, dlat)
    
    ax.set_xticks(np.arange(70, 150, 10), crs=proj)
    ax.set_yticks(np.arange(10, 65, 10), crs=crs)
    
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # ax.add_feature(cfeature.LAND, color='lightgray')
    ax.coastlines()

    
    ax.set_extent([95, 150, 10, 60], crs=crs)


def set_map_contour(ax, dlon, dlat, ra_ds1, new_lons, new_lats,vmin,vmax,cmaps):
    
    '''
    输出：平均变量填色图
    '''  
    # 画图范围

    tick_proj = crs.PlateCarree()
    ax.set_xticks(np.arange(70, 150, 10), crs=tick_proj)
    ax.set_yticks(np.arange(10, 70, 10), crs=tick_proj)
    
    tick_length = 8


    ax.tick_params(axis='both', which='major', length=tick_length / 1.5, width=0.8, labelsize = 8)

    # ax.tick_params(bottom=True, left=True, labelbottom=False, labelleft=False)
    ax.tick_params(bottom=True, left=True, labelbottom=True, labelleft=True)

    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.coastlines(lw=0.5,color='black',zorder=4)


    # ax.set_extent([70, 130, 10, 60], crs=crs)
    ax.set_extent([80, 150, 10, 57]) ##设置范围
    
    sc = ax.scatter(new_lons, new_lats, 
                    c=ra_ds1, 
                    # c='lightblue',
                    cmap=cmaps, 
                    vmin=vmin,
                    vmax=vmax,
                    s=0.01)


    

    return sc



# 文件名
filename = r"D:\data\FY3E_MWTS3\output\clwp\clwp_si.txt"
obs = xr.open_dataset(r"D:\data\FY3E_MWTS3\output\new6.9\fy3_5_mwts3_after_QC_BC_4.nc",engine="netcdf4")
obthin = xr.open_dataset(r"D:\data\FY3E_MWTS3\output\new6.9\MOTOR-3DVar_obsThinned_G06.nc",engine="netcdf4")
obs1 = obs.assign_coords(TBB1=obs["org_TBB002"])
tbb1 = obs1["TBB1"].values

# 读取文件
data = pd.read_csv(filename, delim_whitespace=True, header=None, names=['Index', 'lat', 'lon', 'lwp','SI'])

lat = data["lat"]
lon = data["lon"]
clwp = data["lwp"]*10
clwp_land = data["SI"]



fig = plt.figure(figsize=(15,9),dpi=300)
proj = crs.PlateCarree()

ax1 = fig.add_subplot(221,projection=proj)







mask = (clwp < 2)
filtered_lon = lon[mask]
filtered_lat = lat[mask]
filtered_clwp = clwp[mask]

print(filtered_clwp.shape)
print("clwp_max:", np.max(filtered_clwp), "clwp_min:", np.min(filtered_clwp))# 绘制散点图



#########################
vmin = 0
vmax = np.max(filtered_clwp)


dlon = 10
dlat = 10

# vmin = 0
# vmax = 1

#clevs = [0.1, 10., 25., 50., 100.,250.,500.]   #设置颜色的间隔
# clevs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.6,1.8,2]

# clevs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]
clevs = np.arange(0,2.2,0.2)
#clevs = [0.1, 4., 13., 25., 60.,100.,250.]   #设置颜色的间隔 6小时降水
# cdict = ['#FF00FC','#850042', '#DDA0DD', '#0000FF', '#ADD8E6', '#00FF00', '#90EE90']
# cdict = ['#A9F090', '#40B73F', '#63B7FF', '#0000FE', '#FF00FC', '#850042']
cdict = [
          '#540ADF', '#6A2DDD', '#9D78E3', 
           '#2E40EC', '#5D6BF0', '#A7AEF1', 
            '#BBE0F6', '#058364', '#2EB245', 
        '#19FC07', '#7CE074', '#89FC07', 
            '#CFE814', '#F8FF00', '#E7C016', 
            '#F59D09', '#F56309', '#F53009', 
          ]

my_cmap = colors.ListedColormap(cdict)
norm = mpl.colors.BoundaryNorm(clevs, my_cmap.N)

# my_cmap = 'rainbow'
# my_cmap = get_cmap("RdYlBu_r")
contours = set_map_contour(ax1, dlon, dlat, filtered_clwp, filtered_lon, filtered_lat,vmin,vmax,my_cmap)


#%%
#########################################云检测
###fy-4b
# filename_cfr_l2 = "D:\data\A202403250370789027\FY4B-_AGRI--_N_DISK_1330E_L2-_CLM-_MULT_NOM_20230123221500_20230123222959_4000M_V0001.NC"
filename_cfr_l2 = "F:\computer\data\A202403250370789027\FY4B-_AGRI--_N_DISK_1330E_L2-_CLM-_MULT_NOM_20230123210000_20230123211459_4000M_V0001.NC"
# filename_cfr_l2 = "F:\computer\data\A202403250370789027\FY4B-_AGRI--_N_DISK_1330E_L2-_CLM-_MULT_NOM_20230123223000_20230123224459_4000M_V0001.NC"

rawfile = r'F:\computer\data\FullMask_Grid_4000.raw'

dim = 2748
data = np.fromfile(rawfile,dtype=float,count=dim*dim*2)
latlon = np.reshape(data,(dim, dim, 2))
lat = latlon[:,:,0]
lon = latlon[:,:,1]

data =  xr.open_dataset(filename_cfr_l2)
lat = np.ma.masked_equal(lat, 999999.9999) #space value
lon = np.ma.masked_equal(lon, 999999.9999) #space value


cfr = np.ma.masked_greater(data['CLM'], 125)
# colors = np.array([[255, 255, 255], [173, 216, 230],  [128, 128, 128], [0,0, 200]]) / 255.

colors = np.array([[255, 255, 255], [245, 245, 245],  [128, 128, 128], [0,0, 200]]) / 255.
data = data['CLM']

cmap = mpl.colors.ListedColormap(colors)
cmap.set_bad(color='gray')

'''
0:白色 ：有云
1:浅蓝色：可能有云
2：黄色：可能晴空
3：绿色，晴空

'''



ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')



# # # # 绘制图像
cfr = plt.imshow(cfr, 
                  origin='lower', 
                  extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                  cmap=cmap
                  )



tick_locations = [0.755,1.5,2.255,3]  # 假设这些是与标签对应的数值位置  
tick_labels = ['Cloudy', 'P Cloudy', 'P Clear', 'Clear']  # 对应的英文标签  


cax2 = fig.add_axes([
                    0.3,    ##左右距离
                    0.49,    ##上下（越大越近）
                    0.21,   ##大小
                    0.02  #长宽
                      ])
cb1 = fig.colorbar(cfr,   
                   orientation='horizontal',  
                    ticks=tick_locations,
                    cax=cax2,
                   )
cb1.ax.set_xticklabels(tick_labels)
plt.subplots_adjust(
                    top=0.9, 
                    bottom=0.1, 
                    left=0.3, 
                    right=0.75, 
                    hspace=-0.1,
                    wspace=0.2
                    )
# # 设置colorbar的ticks  
# ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]  # 假设这是你想设置的ticks  
# cb1 = fig.colorbar(cfr,   
#                    orientation='horizontal',  
#                    ticks=ticks)  # 使用ticks参数
# cb1 = fig.colorbar(cfr, 
#               orientation='horizontal',
#               # cax=cax1,
#               )
cax1 = fig.add_axes([
                    0.3,    ##左右距离
                    0.44,    ##上下（越大越近）
                    0.21,   ##大小
                    0.02  #长宽
                      ])
cb1 = fig.colorbar(contours, 
              orientation='horizontal',
               cax=cax1,
              )


tick_locator = ticker.MaxNLocator(nbins=5)  # colorbar上的刻度值个数
cb1.locator = tick_locator
cb1.set_ticks(clevs)
cb1.update_ticks()
# plt.show()
# # 序号列表
# labels = ['(a)', '(b)', '(c)','(d)','(e)','(f)']
# labels = ['(a)', '(b)', '(c)','(d)']




'''
bottom: 子图底部边缘位置的坐标，取值范围为 0 到 1，其中 0 表示图的底部边缘。
left: 子图左侧边缘位置的坐标，取值范围为 0 到 1，其中 0 表示图的左侧边缘。
right: 子图右侧边缘位置的坐标，取值范围为 0 到 1，其中 1 表示图的右侧边缘。
hspace: 子图之间的垂直间距，即高度的空白空间，取值为 0 表示无空白，取值范围大于 0。
wspace: 子图之间的水平间距，即宽度的空白空间，取值为 0 表示无空白，取值范围大于 0。
''' 

# cbar.set_label('TBB (K)')
plt.savefig(r'D:\post_graduate\MOTOR\plot\clwp_mwts3_注释掉MWTS3(6.2).png', dpi=900, bbox_inches='tight')
# plt.savefig(r'F:\computer\tangyuan\plot\plot\子图obtihinde&after_qc_21(MWTS3).png', dpi=300, bbox_inches='tight')

plt.show()


